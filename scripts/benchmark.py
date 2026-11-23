"""Measure how fast this machine runs the experiment, and estimate what a full set costs.

    python scripts/benchmark.py                      # the real model, as configured for the paper
    python scripts/benchmark.py --price-per-hour 0.35
    python scripts/benchmark.py --compile            # also try torch.compile

Run this before launching anything long. It needs no data: it times the real-size model
on random tokens. It measures

* optimizer steps per second for each kind of routing rule,
* forward-only sequences per second (validation),
* the time of one set of gradient probes,
* the size and write time of one checkpoint,

and turns them into an estimated wall-clock time, cost and disk use for one experiment
set (baseline + core-only filtering + GRAM), using the real step counts.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch

from gram.checkpoint import save_checkpoint
from gram.config import AUX_TOPICS, CORE, RunConfig
from gram.eval import mean_loss
from gram.model import build_model
from gram.probes import run_gradient_probes
from gram.routing import Rule
from gram.train import make_optimizers, parameter_dtype, resolve_device, training_step

# Step counts of the code-exact condition (they match the official training logs).
BASELINE_STEPS = 16544
FILTERING_STEPS = 15143
GRAM_RULE_COUNTS = {"core_plain": 7332, "core_robust": 7332, "aux_spread": 406, "aux_isolated": 951}
# Validation data: sequences of 256 tokens.
CORE_VAL, AUX_VAL = 205_967, 19_050  # all core sequences; the four auxiliary topics together
SWEEPS, SWEEP_CORE_SEQUENCES = 200, 50 * 128
ELICIT_VAL_PER_TOPIC = AUX_VAL // 4


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def random_tokens(config: RunConfig, count: int, device: torch.device) -> torch.Tensor:
    return torch.randint(2, config.model.vocab_size, (count, config.model.context_length + 1), device=device)


def time_steps(model, optimizers, rule: Rule, config: RunConfig, device, steps: int, forward=None) -> float:
    """Optimizer steps per second under one routing rule (after two warm-up steps)."""
    batch = random_tokens(config, config.optim.micro_batch_size, device)
    for _ in range(2):
        training_step(model, optimizers, [(batch, rule)], 1e-5, 1.0, forward)
    synchronize(device)
    started = time.time()
    for _ in range(steps):
        training_step(model, optimizers, [(batch, rule)], 1e-5, 1.0, forward)
    synchronize(device)
    return steps / (time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=30, help="optimizer steps to time per rule")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="time the model with torch.compile as well")
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--price-per-hour", type=float, default=None, help="GPU rental price, to turn hours into money")
    args = parser.parse_args()

    config = RunConfig(name="benchmark", method="gram")
    config.optim.precision = args.precision
    config.optim.micro_batch_size = config.optim.batch_size = args.micro_batch_size
    device = resolve_device(args.device)
    dtype = parameter_dtype(args.precision)
    module = AUX_TOPICS[0]
    everything = (CORE, *AUX_TOPICS)

    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"torch {torch.__version__}, parameters stored as {args.precision}, "
          f"micro-batch {args.micro_batch_size} x {config.model.context_length} tokens\n")

    # ---------------------------------------------------------------- training speed
    dense = build_model("baseline", config.model, AUX_TOPICS).to(device=device, dtype=dtype)
    gram = build_model("gram", config.model, AUX_TOPICS).to(device=device, dtype=dtype)
    dense_optimizers = make_optimizers(dense, config.optim)
    gram_optimizers = make_optimizers(gram, config.optim)

    rules = {
        "dense (baseline, filtering)": (dense, dense_optimizers, Rule("dense", CORE, (CORE,), (CORE,))),
        "core_plain": (gram, gram_optimizers, Rule("core_plain", CORE, (CORE,), (CORE,))),
        "core_robust / aux_spread": (gram, gram_optimizers, Rule("core_robust", CORE, (CORE, module), (CORE, module))),
        "aux_isolated": (gram, gram_optimizers, Rule("aux_isolated", module, (CORE, module), (module,))),
        "unlabeled": (gram, gram_optimizers, Rule("unlabeled", "unlabeled", everything, everything)),
    }
    speed = {}
    print("optimizer steps per second")
    for name, (model, optimizers, rule) in rules.items():
        speed[name] = time_steps(model, optimizers, rule, config, device, args.steps)
        print(f"  {name:<30} {speed[name]:8.2f}")
    if args.compile:
        torch._dynamo.config.recompile_limit = 64
        compiled_speed = {}
        for name, (model, optimizers, rule) in rules.items():
            compiled_speed[name] = time_steps(model, optimizers, rule, config, device, args.steps, torch.compile(model))
            print(f"  {name:<30} {compiled_speed[name]:8.2f}   with torch.compile ({compiled_speed[name] / speed[name]:.2f}x)")
    if device.type == "cuda":
        print(f"  peak GPU memory so far: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # -------------------------------------------------------------- validation speed
    sequences = random_tokens(config, 4 * config.eval.batch_size, device).to(torch.int16)
    mean_loss(dense, sequences[: config.eval.batch_size], (CORE,), config.eval.batch_size)
    synchronize(device)
    started = time.time()
    mean_loss(dense, sequences, (CORE,), config.eval.batch_size)
    synchronize(device)
    forward_rate = len(sequences) / (time.time() - started)
    print(f"\nforward-only sequences per second    {forward_rate:8.0f}")

    # ------------------------------------------------------------------------ probes
    probe_batches = {
        name: (random_tokens(config, args.micro_batch_size, device), random_tokens(config, args.micro_batch_size, device))
        for name in everything
    }
    started = time.time()
    run_gradient_probes(gram, probe_batches, step=0)
    synchronize(device)
    probe_seconds = time.time() - started
    print(f"one set of gradient probes (GRAM)    {probe_seconds:8.1f} s")

    # ------------------------------------------------------------------- checkpoints
    sizes = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, model, optimizers in (("dense", dense, dense_optimizers), ("GRAM", gram, gram_optimizers)):
            path = Path(directory) / f"{name}.pt"
            started = time.time()
            save_checkpoint(path, 0, 1, model, optimizers, config)
            sizes[name] = path.stat().st_size / 1e9
            print(f"one {name} checkpoint                {sizes[name]:8.3f} GB, written in {time.time() - started:.1f} s")

    # --------------------------------------------------------------------- estimates
    dense_rate = speed["dense (baseline, filtering)"]
    gram_train = (
        GRAM_RULE_COUNTS["core_plain"] / speed["core_plain"]
        + (GRAM_RULE_COUNTS["core_robust"] + GRAM_RULE_COUNTS["aux_spread"]) / speed["core_robust / aux_spread"]
        + GRAM_RULE_COUNTS["aux_isolated"] / speed["aux_isolated"]
    )
    # Elicitation: the "code" protocol typically stops after about 15 full-batch steps over 512
    # sequences; the "paper" protocol always runs 75 steps over 128. Each step is followed by a
    # validation pass over the topic.
    code_steps, paper_steps = 15, 75
    elicit = 4 * (
        (code_steps * 4 + paper_steps) / dense_rate + (code_steps + paper_steps + 2) * ELICIT_VAL_PER_TOPIC / forward_rate
    )
    minutes = {
        "baseline": {
            "training": BASELINE_STEPS / dense_rate,
            "learning-curve sweeps": SWEEPS * (SWEEP_CORE_SEQUENCES + AUX_VAL) / forward_rate,
            "gradient probes": 20 * probe_seconds / 3,  # a dense model has one group and no labeled rule
            "final evaluation": (CORE_VAL + AUX_VAL) / forward_rate,
            "elicitation": elicit,
        },
        "filtering": {
            "training": FILTERING_STEPS / dense_rate,
            "gradient probes": 20 * probe_seconds / 3,
            "final evaluation": (CORE_VAL + AUX_VAL) / forward_rate,
            "elicitation": elicit,
        },
        "GRAM": {
            "training": gram_train,
            "gradient probes": 20 * probe_seconds,
            "final evaluation": 10 * (CORE_VAL + AUX_VAL) / forward_rate,  # ten profiles
            "elicitation": elicit,
        },
    }
    print("\nestimated time for one experiment set, code-exact condition, one seed (minutes)")
    total = 0.0
    for run, parts in minutes.items():
        run_total = sum(parts.values()) / 60
        total += run_total
        detail = ", ".join(f"{name} {seconds / 60:.1f}" for name, seconds in parts.items())
        print(f"  {run:<10} {run_total:6.1f}   ({detail})")
    print(f"  {'total':<10} {total:6.1f}   = {total / 60:.2f} hours")
    if args.price_per_hour is not None:
        print(f"  at ${args.price_per_hour:.2f}/hour: ${total / 60 * args.price_per_hour:.2f} per set")

    checkpoints = config.checkpoints.num_spaced + 1
    disk = checkpoints * (2 * sizes["dense"] + sizes["GRAM"])
    print(f"\ndisk for one set: {checkpoints} checkpoints per run -> "
          f"{checkpoints * sizes['dense']:.1f} GB per dense run, {checkpoints * sizes['GRAM']:.1f} GB per GRAM run, {disk:.1f} GB per set")
    if args.micro_batch_size != 128:
        print("\nNote: the time estimate assumes the paper's batch of 128 sequences; it is not meaningful "
              f"for the micro-batch of {args.micro_batch_size} that was timed here.")
    if device.type != "cuda":
        print("\nNote: these numbers are for this CPU. Run the script on the training GPU for the numbers that matter.")


if __name__ == "__main__":
    main()
