"""Training one model.

The loop is the same for the baseline, for filtering and for GRAM. What differs between
them is the *plan* (gram/routing.py): which batches are trained on and, for each one,
which modules are active and which parameter groups may be updated.

What one optimizer step does (``training_step``)
------------------------------------------------
1. For each micro-batch of the step: run the model with that micro-batch's routing rule
   and backpropagate. Groups that the rule does not update are frozen for that pass, so
   they collect no gradient from it.
2. Clip the gradient of everything that is about to be updated, jointly, to norm 1.
3. Step the optimizer of every group that at least one micro-batch updated, and of no
   other group.

Two mechanisms keep an un-routed group untouched, and it is worth knowing which does what:

* **Selective stepping** (step 3). Each group has its own AdamW optimizer, and a group's
  optimizer is only stepped if the group was routed to. A group that is not stepped is
  left completely alone: no gradient step, no weight decay, no change to its momentum.
  With one micro-batch per optimizer step, which is the paper's Simple Stories setting,
  this alone is enough.
* **Freezing** (step 1, gram/freeze.py). With gradient accumulation, several micro-batches
  with *different* rules share one optimizer step, so a group can be stepped because of
  one micro-batch while another micro-batch, which should not have touched it, has
  already added to its gradient. Freezing stops that gradient from ever being computed.
  It only matters when ``batch_size > micro_batch_size``.

``training_step`` checks on every step that the two agree: a group has a gradient if and
only if some micro-batch was routed to it.

A note on "one optimizer per group". What must be avoided is an optimizer step that moves
parameters whose gradient is *zero*: AdamW would still apply weight decay and whatever
momentum is left from earlier steps (tests/test_routing_invariants.py shows this
happening). Separate optimizers make that impossible by construction.

Everything a run writes goes into its run directory; see README.md for the layout.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from gram.checkpoint import (
    final_checkpoint_path,
    latest_checkpoint_path,
    load_checkpoint,
    newest_resumable_checkpoint,
    save_checkpoint,
    step_checkpoint_path,
)
from gram.config import CORE, UNLABELED, OptimConfig, RunConfig, config_to_dict, load_config, save_config
from gram.data import (
    TokenStore,
    TrainingData,
    assign_batches_to_schedule,
    build_probe_batches,
    build_training_data,
    build_validation_sets,
    rng_for,
)
from gram.eval import fingerprint, validation_sweep
from gram.model import Transformer, build_model, next_token_loss
from gram.probes import probe_steps, run_gradient_probes
from gram.routing import Plan, Rule, build_dense_plan, build_gram_plan

LEARNING_RATE_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# Small pieces
# --------------------------------------------------------------------------- #


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parameter_dtype(precision: str) -> torch.dtype:
    """bf16: the whole model is stored in bfloat16, as in the official code. fp32: ordinary float32."""
    return torch.bfloat16 if precision == "bf16" else torch.float32


def token_tensor(sequences: np.ndarray, device: torch.device) -> torch.Tensor:
    """Token ids as a compact integer tensor on the training device.

    The whole training set is about 1.1 GB this way and lives on the GPU, so making a
    batch is a single indexing operation and needs no data-loader processes.
    """
    assert sequences.max(initial=0) < 2**15, "token ids do not fit in int16"
    return torch.from_numpy(sequences.astype(np.int16)).to(device)


def learning_rate(step: int, total_steps: int, optim: OptimConfig) -> float:
    """Learning rate for optimizer step number ``step`` (counting from 0).

    Linear warm-up over the first 10% of steps, constant for 80%, linear decay over the
    last 10%. Warm-up starts from, and decay ends at, a floor of 1e-8.
    """
    warmup = max(1, round(optim.warmup_fraction * total_steps))
    decay = max(1, round(optim.decay_fraction * total_steps))
    constant = total_steps - warmup - decay
    if step < warmup:
        return LEARNING_RATE_FLOOR + (optim.lr - LEARNING_RATE_FLOOR) * step / warmup
    if step < warmup + constant:
        return optim.lr
    return optim.lr - (optim.lr - LEARNING_RATE_FLOOR) * (step - warmup - constant) / decay


def make_optimizers(model: Transformer, optim: OptimConfig) -> dict[str, torch.optim.AdamW]:
    """One AdamW optimizer per parameter group (see the module docstring for why)."""
    return {
        name: torch.optim.AdamW(
            parameters, lr=optim.lr, betas=tuple(optim.betas), weight_decay=optim.weight_decay, fused=True
        )
        for name, parameters in model.parameter_groups().items()
    }


def spaced_checkpoint_steps(total_steps: int, count: int) -> list[int]:
    """Steps of the checkpoints kept for analysis: 0%, 1/count, 2/count, ... of training."""
    return sorted({round(index * total_steps / count) for index in range(count)})


def curve_steps(total_steps: int, count: int) -> list[int]:
    """Steps after which a validation sweep is run: evenly spaced, ending with the last step."""
    return sorted({max(1, round(point * total_steps / count)) for point in range(1, count + 1)})


# --------------------------------------------------------------------------- #
# One optimizer step
# --------------------------------------------------------------------------- #


def accumulate_gradients(
    model: Transformer,
    micro_batches: Sequence[tuple[torch.Tensor, Rule]],
    forward: Callable | None = None,
) -> list[torch.Tensor]:
    """Backpropagate every micro-batch of one optimizer step, each under its own routing rule.

    Afterwards the ``.grad`` of a parameter group is the sum of the gradients of exactly
    those micro-batches whose rule updates the group (each divided by the number of
    micro-batches). A micro-batch contributes nothing to a group its rule does not
    update, because the group is frozen for that forward pass. Returns the losses.
    """
    forward = model if forward is None else forward
    losses = []
    for tokens, rule in micro_batches:
        logits = forward(tokens[:, :-1], active=rule.active, update=rule.update)
        loss = next_token_loss(logits, tokens[:, 1:])
        (loss / len(micro_batches)).backward()
        losses.append(loss.detach())
    return losses


def training_step(
    model: Transformer,
    optimizers: dict[str, torch.optim.Optimizer],
    micro_batches: Sequence[tuple[torch.Tensor, Rule]],
    lr: float,
    grad_clip: float,
    forward: Callable | None = None,
) -> dict:
    """Take one optimizer step from one or more micro-batches, each with its own routing rule.

    ``micro_batches`` is a list of ``(tokens, rule)``; ``tokens`` has shape
    (micro_batch_size, context_length + 1). ``forward`` is the callable used for the
    forward pass (the compiled model, if there is one); it defaults to ``model``.
    """
    groups = model.parameter_groups()

    # 1. Accumulate gradients. Each micro-batch only reaches the groups its rule updates.
    losses = accumulate_gradients(model, micro_batches, forward)
    updated = [name for name in optimizers if any(name in rule.update for _, rule in micro_batches)]

    # The invariant gradient routing rests on: a group has a gradient if and only if some
    # micro-batch was routed to it. If this ever fails, routing is broken, so stop.
    for name, parameters in groups.items():
        has_gradient = any(parameter.grad is not None for parameter in parameters)
        if has_gradient != (name in updated):
            raise RuntimeError(f"routing violated: group {name!r} gradient={has_gradient}, routed={name in updated}")

    # 2. Clip everything that is about to be updated, as one vector.
    grad_norm = nn.utils.clip_grad_norm_([p for name in updated for p in groups[name]], grad_clip)

    # 3. Step only the groups that were routed to.
    for name in updated:
        for param_group in optimizers[name].param_groups:
            param_group["lr"] = lr
        optimizers[name].step()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    return {"losses": [float(loss) for loss in losses], "grad_norm": float(grad_norm), "updated": updated}


# --------------------------------------------------------------------------- #
# Deciding what a run trains on
# --------------------------------------------------------------------------- #


@dataclass
class TrainingSetup:
    data: TrainingData
    plan: Plan
    # For every micro-batch in training order: the rows of data.sequences it contains.
    batch_rows: np.ndarray

    @property
    def total_steps(self) -> int:
        return self.plan.num_steps


def set_up_training(config: RunConfig, store: TokenStore, model: Transformer) -> TrainingSetup:
    """Build the training data, the routing plan and the batch assignment for a run.

    This is a pure function of the configuration: calling it again (for instance when a
    run is resumed) gives exactly the same sequences, rules and batches.
    """
    data = build_training_data(store, config.data, config.labels, config.model.context_length, config.seed)
    micro = config.optim.micro_batch_size
    assembly = config.data.batch_assembly
    pool_sizes = {pool: data.pool_num_batches(pool, micro, assembly) for pool in data.pool_names()}
    accumulation = config.optim.accumulation_steps
    rng = rng_for(config.seed, "plan")

    if config.method == "baseline":
        # The baseline ignores labels: it trains on every pool.
        plan = build_dense_plan(pool_sizes, data.pool_names(), accumulation, rng)
    elif config.method == "filtering":
        # Core-only filtering removes everything *given* an auxiliary label. Unlabeled data
        # cannot be filtered, so it stays in, whatever its true topic.
        plan = build_dense_plan(pool_sizes, [CORE, UNLABELED], accumulation, rng)
    else:
        dense = build_model("baseline", config.model, config.data.aux_topics)
        baseline_params = sum(parameter.numel() for parameter in dense.parameters())
        plan = build_gram_plan(
            pool_sizes, config.data.aux_topics, config.routing, model.num_parameters(), baseline_params, accumulation, rng
        )

    pools_in_order = [plan.rules[index].pool for index in plan.schedule]
    batch_rows = assign_batches_to_schedule(pools_in_order, data, micro, assembly, config.seed)
    return TrainingSetup(data=data, plan=plan, batch_rows=batch_rows)


def describe_plan(setup: TrainingSetup, config: RunConfig, checkpoint_at: list[int], probe_at: list[int]) -> dict:
    plan = setup.plan
    return {
        "method": config.method,
        "total_steps": plan.num_steps,
        "micro_batches": int(len(plan.schedule)),
        "accumulation_steps": plan.accumulation_steps,
        "rules": [
            {"name": rule.name, "pool": rule.pool, "active": list(rule.active), "update": list(rule.update), "count": count}
            for rule, count in zip(plan.rules, plan.counts)
        ],
        "count_by_rule_name": plan.count_by_rule_name(),
        "pools": {
            pool: {
                "sequences": setup.data.pool_num_sequences(pool),
                "micro_batches_per_pass": setup.data.pool_num_batches(
                    pool, config.optim.micro_batch_size, config.data.batch_assembly
                ),
            }
            for pool in setup.data.pool_names()
        },
        "notes": plan.notes,
        "checkpoint_steps": checkpoint_at,
        "probe_steps": probe_at,
    }


def write_run_inputs(run_dir: Path, config: RunConfig, setup: TrainingSetup, plan_description: dict) -> None:
    """Save what the run was given: its configuration, its split manifest and its batch log."""
    save_config(config, run_dir / "config.yaml")
    setup.data.manifest.to_parquet(run_dir / "manifest.parquet", compression="zstd", index=False)
    (run_dir / "plan.json").write_text(json.dumps(plan_description, indent=2))
    pd.DataFrame(
        [(cell.given_label, cell.topic, cell.first_sequence, cell.num_sequences, cell.num_stories) for cell in setup.data.cells],
        columns=["given_label", "true_topic", "first_sequence", "num_sequences", "num_stories"],
    ).to_csv(run_dir / "cells.csv", index=False)
    # The batch log: for micro-batch m (step m // accumulation_steps), the rule it was trained
    # under and the rows of the sequence table it contained. Together with the manifest this
    # says exactly when, and under which rule, every story was seen.
    np.savez_compressed(
        run_dir / "batch_log.npz",
        rule_index=setup.plan.schedule.astype(np.int16),
        sequence_rows=setup.batch_rows.astype(np.int32),
        rule_names=np.array([rule.name for rule in setup.plan.rules]),
        rule_pools=np.array([rule.pool for rule in setup.plan.rules]),
        rule_active=np.array(["+".join(rule.active) for rule in setup.plan.rules]),
        rule_update=np.array(["+".join(rule.update) for rule in setup.plan.rules]),
    )


# Settings that may differ between two invocations of the same run without changing its results.
IRRELEVANT_TO_RESULTS = {
    "device",
    "output_dir",
    "data.store_dir",
    "eval.batch_size",
    "optim.compile",
    "checkpoints.resume_every_steps",
    "probes.chunk_size",
}


def _flatten(value, prefix: str = "") -> dict:
    """{"optim": {"lr": 1}} -> {"optim.lr": 1}"""
    if not isinstance(value, dict):
        return {prefix.rstrip("."): value}
    flat = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}{key}."))
    return flat


def require_same_config(run_dir: Path, config: RunConfig) -> None:
    """Refuse to continue a run directory that was created with different settings.

    Without this, editing a configuration file and running it again under the same name
    would resume the old run, or report its old results, as if they belonged to the new
    settings.
    """
    saved_path = run_dir / "config.yaml"
    if not saved_path.exists():
        return
    saved = _flatten(config_to_dict(load_config(saved_path)))
    current = _flatten(config_to_dict(config))
    differing = sorted(
        key
        for key in saved.keys() | current.keys()
        if key not in IRRELEVANT_TO_RESULTS and saved.get(key) != current.get(key)
    )
    if differing:
        raise ValueError(
            f"{run_dir} was created with a different configuration (differs in: {', '.join(differing)}). "
            "Give the new run a different name, or delete the old run directory."
        )


def small_cell_warnings(data: TrainingData, micro_batch_size: int, assembly: str) -> list[str]:
    """Label cells that lose much of their data to incomplete batches.

    With ``single_topic`` batches, every (given label, true topic) cell is batched on its
    own, and a cell smaller than one batch contributes nothing at all. That is harmless
    for the 48 large cells of perfectly labeled data, and fatal for small cells such as
    "core stories of topic X that were wrongly given an auxiliary label".
    """
    if assembly != "single_topic":
        return []
    warnings = []
    for cell in data.cells:
        unused = cell.num_sequences % micro_batch_size
        if cell.num_sequences > 0 and unused / cell.num_sequences > 0.25:
            warnings.append(
                f"given label {cell.given_label!r}, true topic {cell.topic!r}: {cell.num_sequences} sequences, "
                f"{unused} of them ({unused / cell.num_sequences:.0%}) never trained on"
            )
    return warnings


# --------------------------------------------------------------------------- #
# Bookkeeping that survives interruptions
# --------------------------------------------------------------------------- #


class MetricTable:
    """Rows of one metrics file, kept in memory and written out whenever a checkpoint is.

    Writing metrics and checkpoints together means the files on disk always describe the
    same moment. When a run resumes, rows from after the checkpoint (if any) are dropped.
    """

    def __init__(self, path: Path, resume_step: int | None) -> None:
        self.path = path
        self.rows: list[dict] = []
        if resume_step is not None and path.exists():
            saved = pd.read_parquet(path)
            self.rows = saved[saved["step"] <= resume_step].to_dict("records")

    def extend(self, rows: Sequence[dict]) -> None:
        self.rows.extend(rows)

    def write(self) -> None:
        if self.rows:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_parquet(self.path, compression="zstd", index=False)


class StopSignal:
    """Turns SIGTERM / SIGINT into a request to stop cleanly after the current step."""

    def __init__(self) -> None:
        self.requested = False
        self._previous = {}

    def __enter__(self) -> "StopSignal":
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[signum] = signal.signal(signum, self._handle)
            except ValueError:  # not in the main thread (for example inside a test runner)
                pass
        return self

    def __exit__(self, *exc) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def _handle(self, signum, frame) -> None:
        self.requested = True


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gram.{run_dir.name}.{id(run_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "log.txt"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


# --------------------------------------------------------------------------- #
# The training loop
# --------------------------------------------------------------------------- #


def train(config: RunConfig, stop_after_steps: int | None = None) -> str:
    """Train the model described by ``config``. Returns "finished" or "interrupted".

    Calling this again on a run that was interrupted continues from its newest
    checkpoint and produces the same result as an uninterrupted run. ``stop_after_steps``
    makes the call stop, as if interrupted, after that many optimizer steps; it exists
    for testing resumption.
    """
    config.validate()
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    require_same_config(run_dir, config)
    if final_checkpoint_path(run_dir).exists():
        return "finished"

    logger = make_logger(run_dir)
    try:
        with StopSignal() as stop:
            return _train(config, run_dir, logger, stop, stop_after_steps)
    finally:
        close_logger(logger)


def _train(config: RunConfig, run_dir: Path, logger: logging.Logger, stop: StopSignal, stop_after_steps: int | None) -> str:
    device = resolve_device(config.device)
    # TF32 is deliberately left off. The official code switches it on, but it only affects
    # float32 matrix products, of which a bfloat16 model has none; it would however lower the
    # precision of the gradient probes, which do run in float32.

    # ----- model, data, plan. All of it follows from the configuration and the seed.
    torch.manual_seed(config.seed)
    model = build_model(config.method, config.model, config.data.aux_topics)  # initialised on the CPU, in float32
    store = TokenStore(config.data.store_dir)
    setup = set_up_training(config, store, model)
    total_steps = setup.total_steps
    if total_steps == 0:
        raise ValueError("this configuration leaves nothing to train on (not a single full batch)")
    accumulation = config.optim.accumulation_steps
    rules = setup.plan.rules

    checkpoint_at = spaced_checkpoint_steps(total_steps, config.checkpoints.num_spaced)
    probe_at = probe_steps(total_steps, config.probes.num_points) if config.probes.enabled else []
    sweep_at = curve_steps(total_steps, config.eval.curve_points) if config.eval.curve_points > 0 else []
    plan_description = describe_plan(setup, config, checkpoint_at, probe_at)
    plan_description["warnings"] = small_cell_warnings(setup.data, config.optim.micro_batch_size, config.data.batch_assembly)
    write_run_inputs(run_dir, config, setup, plan_description)
    if plan_description["warnings"]:
        logger.warning(
            f"WARNING: {len(plan_description['warnings'])} label cell(s) lose more than a quarter of their data to "
            'incomplete single-topic batches. Consider data.batch_assembly: "mixed". The first few:'
        )
        for line in plan_description["warnings"][:5]:
            logger.warning(f"  {line}")

    model.to(device=device, dtype=parameter_dtype(config.optim.precision))
    optimizers = make_optimizers(model, config.optim)

    forward = model
    if config.optim.compile:
        # Each distinct routing rule is compiled once; allow for all of them.
        torch._dynamo.config.recompile_limit = 64
        forward = torch.compile(model)

    # ----- resume, or start from the initialisation
    resume = newest_resumable_checkpoint(run_dir)
    if resume is not None:
        path, step = resume
        state = load_checkpoint(path, device)
        model.load_state_dict(state["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        logger.info(f"resuming {config.name} from {path.name} at step {step} of {total_steps}")
    else:
        step = 0
        logger.info(f"starting {config.name}: {config.method}, seed {config.seed}, {total_steps} steps on {device}")
        logger.info(f"micro-batches per rule: {setup.plan.count_by_rule_name()}")
        logger.info(f"parameters per group: {model.num_parameters()}")
        if 0 in checkpoint_at:
            save_checkpoint(step_checkpoint_path(run_dir, 0), 0, total_steps, model, optimizers, config)

    resume_step = step if resume is not None else None
    train_metrics = MetricTable(run_dir / "metrics" / "train.parquet", resume_step)
    curve_metrics = MetricTable(run_dir / "metrics" / "validation_curve.parquet", resume_step)
    probe_metrics = MetricTable(run_dir / "probes" / "gradient_probes.parquet", resume_step)

    # ----- move the data to the device
    sequences = token_tensor(setup.data.sequences, device)
    batch_rows = torch.from_numpy(setup.batch_rows).to(device)
    validation_sets = probe_batches = None
    if sweep_at:
        validation_sets = {
            name: token_tensor(tokens, device)
            for name, tokens in build_validation_sets(
                store, config.data.aux_topics, config.model.context_length, config.data.val_fraction
            ).items()
        }
    if probe_at:
        probe_batches = {
            name: (token_tensor(first, device), token_tensor(second, device))
            for name, (first, second) in build_probe_batches(
                store, config.data.aux_topics, config.model.context_length, config.optim.micro_batch_size
            ).items()
        }

    def save(path: Path) -> None:
        for table in (train_metrics, curve_metrics, probe_metrics):
            table.write()
        save_checkpoint(path, step, total_steps, model, optimizers, config)

    # ----- the loop
    model.train()
    started = time.time()
    first_step_of_this_call = step
    seconds_training = seconds_measuring = 0.0
    recent_losses: list[float] = []

    while step < total_steps:
        tick = time.time()
        lr = learning_rate(step, total_steps, config.optim)
        micro_indices = range(step * accumulation, (step + 1) * accumulation)
        micro_batches = [
            (sequences[batch_rows[index]].long(), rules[setup.plan.schedule[index]]) for index in micro_indices
        ]
        stats = training_step(model, optimizers, micro_batches, lr, config.optim.grad_clip, forward)
        step += 1

        for position, ((_, rule), loss) in enumerate(zip(micro_batches, stats["losses"])):
            train_metrics.rows.append(
                {
                    "step": step,
                    "micro_batch": position,
                    "rule": rule.name,
                    "pool": rule.pool,
                    "module": rule.module or "",
                    "loss": loss,
                    "lr": lr,
                    "grad_norm": stats["grad_norm"],
                }
            )
        recent_losses = (recent_losses + stats["losses"])[-100:]
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds_training += time.time() - tick

        # Measurements. None of these changes the model, the optimizers or any random state.
        tick = time.time()
        if step in sweep_at:
            curve_metrics.extend(
                validation_sweep(
                    model, validation_sets, step, config.seed,
                    core_sequences=config.eval.curve_core_batches * config.optim.batch_size,
                    batch_size=config.eval.batch_size,
                )
            )
        if step in probe_at:
            rows = run_gradient_probes(model, probe_batches, step, config.probes.chunk_size)
            probe_metrics.extend({"run": config.name, **row} for row in rows)
        seconds_measuring += time.time() - tick

        # Checkpoints.
        finished = step == total_steps
        interrupted = not finished and (
            stop.requested or (stop_after_steps is not None and step - first_step_of_this_call >= stop_after_steps)
        )
        if finished:
            # The summary is written first, so that a final checkpoint never exists without it.
            fingerprint_sets = validation_sets or {
                name: token_tensor(tokens, device)
                for name, tokens in build_validation_sets(
                    store, config.data.aux_topics, config.model.context_length, config.data.val_fraction
                ).items()
            }
            summary = {
                "total_steps": total_steps,
                "steps_in_this_call": step - first_step_of_this_call,
                "seconds_training": seconds_training,
                "seconds_measuring": seconds_measuring,
                "steps_per_second": (step - first_step_of_this_call) / max(seconds_training, 1e-9),
                "device": str(device),
                "torch_version": torch.__version__,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                "fingerprint": fingerprint(model, fingerprint_sets, config.eval.batch_size),
            }
            (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
            save(final_checkpoint_path(run_dir))
        elif step in checkpoint_at:
            save(step_checkpoint_path(run_dir, step))
        elif interrupted or step % config.checkpoints.resume_every_steps == 0:
            save(latest_checkpoint_path(run_dir))

        if step % 100 == 0 or finished or interrupted:
            done = step - first_step_of_this_call
            rate = done / max(seconds_training, 1e-9)
            remaining = (total_steps - step) / max(done / max(time.time() - started, 1e-9), 1e-9)
            logger.info(
                f"step {step:>6}/{total_steps}  loss {np.mean(recent_losses):.4f}  lr {lr:.2e}  "
                f"{rate:.1f} steps/s  about {remaining / 60:.1f} min left"
            )
        if interrupted:
            logger.info(f"interrupted at step {step}; run the same command again to continue")
            return "interrupted"

    latest_checkpoint_path(run_dir).unlink(missing_ok=True)
    logger.info(
        f"finished training in {(time.time() - started) / 60:.1f} min "
        f"({seconds_training:.0f}s of training steps, {seconds_measuring:.0f}s of measurements)"
    )
    return "finished"
