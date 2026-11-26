"""Write the configuration files of an experiment set.

    python scripts/make_configs.py            # list the sets
    python scripts/make_configs.py code_exact # write configs/code_exact/*.yaml

One YAML file per run. The files are committed, so what a set will run can be read
without running anything. Each run writes to runs/<set>/<run name>/.

This file is the statement of the experimental design: every set is a list of runs, and
every run is "the defaults, with these fields changed". The defaults (gram/config.py)
are the code-exact Simple Stories condition.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from gram.config import ElicitProtocol, RunConfig, save_config

ROOT = Path(__file__).resolve().parent.parent


def make(set_name: str, name: str, method: str, seed: int = 1, **changes) -> RunConfig:
    """One run: the defaults with ``changes`` applied. Keys are dotted paths, e.g. ``optim.weight_decay``."""
    config = RunConfig(name=name, method=method, seed=seed, output_dir=f"runs/{set_name}")
    for dotted, value in changes.items():
        target = config
        *parents, leaf = dotted.split(".")
        for parent in parents:
            target = getattr(target, parent)
        if not hasattr(target, leaf):
            raise KeyError(f"unknown configuration field {dotted!r}")
        setattr(target, leaf, copy.deepcopy(value))
    config.validate()
    return config


def three_methods(set_name: str, seed: int, **changes) -> list[RunConfig]:
    """The baseline (with its learning curve), core-only filtering, and GRAM, under one condition."""
    return [
        # Only the baseline needs validation sweeps: they are the compute-ratio reference.
        make(set_name, f"baseline_seed{seed}", "baseline", seed, **{"eval.curve_points": 200, **changes}),
        make(set_name, f"filtering_seed{seed}", "filtering", seed, **changes),
        make(set_name, f"gram_seed{seed}", "gram", seed, **changes),
    ]


# --------------------------------------------------------------------------- #
# Stage 1: smoke tests
# --------------------------------------------------------------------------- #


def smoke() -> list[RunConfig]:
    """The real model on 1% of the data. A few minutes on a GPU; use it to time the pipeline."""
    changes = {
        "data.train_fraction": 0.01,
        "data.val_fraction": 0.05,
        "eval.curve_points": 20,  # the baseline's curve; overridden to 0 for the other two below
    }
    runs = three_methods("smoke", 1, **changes)
    for run in runs[1:]:
        run.eval.curve_points = 0
    return runs


def smoke_compile() -> list[RunConfig]:
    """The smoke set with torch.compile on: checks the whole pipeline with the compiled training step."""
    runs = smoke()
    for run in runs:
        run.output_dir = "runs/smoke_compile"
        run.optim.compile = True
    return runs


def smoke_cpu() -> list[RunConfig]:
    """The same pipeline on 1% of the data, cut down to run on a laptop CPU in half an hour.

    Every part runs on real data (routing, checkpoints, probes, evaluation, elicitation,
    compute ratios). What is cut down: the model (1.5M parameters, stored in float32),
    the number of probe points and validation sweeps, and the elicitation protocols.
    Use it to check that everything works, not to learn anything about GRAM.
    """
    changes = {
        "data.train_fraction": 0.01,
        "data.val_fraction": 0.02,
        "eval.curve_points": 10,
        "eval.batch_size": 128,
        "probes.num_points": 5,
        "elicit.protocols": [
            ElicitProtocol(name="code", num_sequences=128, max_steps=30, patience=5),
            ElicitProtocol(name="paper", num_sequences=128, max_steps=20, patience=0),
        ],
        "optim.precision": "fp32",
        "model.n_layers": 2,
        "model.n_heads": 4,
        "model.d_model": 128,
        "model.dense_mlp_dim": 512,
        "model.core_mlp_dim": 512,
        "model.aux_mlp_dim": 48,
        "device": "cpu",
    }
    runs = three_methods("smoke_cpu", 1, **changes)
    for run in runs[1:]:
        run.eval.curve_points = 0
    return runs


# --------------------------------------------------------------------------- #
# Stage 2: replicate the paper with complete, correct labels
# --------------------------------------------------------------------------- #

# The official runs used torch.compile, and on the RTX 4090 it nearly doubles training
# speed (5.6 -> 10.6 steps per second). It changes speed, not what is computed.
STAGE_2 = {"optim.compile": True}



def code_exact(seeds=(1,)) -> list[RunConfig]:
    """The settings recorded in the official repository for the runs behind Table 1.

    These are the defaults: core MLP width 2048, Appendix I length rescaling, weight
    decay 0.01 (PyTorch's default; the official code never sets it), topics capped at
    the mean size of their class.
    """
    return [run for seed in seeds for run in three_methods("code_exact", seed, **STAGE_2)]


def paper_exact(seeds=(1,)) -> list[RunConfig]:
    """The settings the paper's text states (Section 4 and Appendix A), where they differ.

    Core MLP width 1856 (so that core + one module is the size of the baseline), the same
    number of training steps for GRAM as for the baseline, weight decay 0.1, and all
    available data. See DEVIATIONS.md section 1.
    """
    changes = {
        "model.core_mlp_dim": 1856,
        "routing.equal_compute": False,
        "optim.weight_decay": 0.1,
        "data.cap_topics_at_class_mean": False,
        **STAGE_2,
    }
    return [run for seed in seeds for run in three_methods("paper_exact", seed, **changes)]


SETS = {
    "smoke": smoke,
    "smoke_compile": smoke_compile,
    "smoke_cpu": smoke_cpu,
    "code_exact": code_exact,
    "paper_exact": paper_exact,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", nargs="?", choices=sorted(SETS), help="the experiment set to write")
    args = parser.parse_args()

    if args.set is None:
        for name, builder in SETS.items():
            print(f"{name:<12} {len(builder())} runs   {builder.__doc__.strip().splitlines()[0]}")
        return

    directory = ROOT / "configs" / args.set
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.yaml"):
        stale.unlink()
    for config in SETS[args.set]():
        save_config(config, directory / f"{config.name}.yaml")
        print(f"wrote {(directory / config.name).relative_to(ROOT)}.yaml")


if __name__ == "__main__":
    main()
