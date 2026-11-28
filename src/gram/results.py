"""From run directories to result tables.

A *set* of runs is a directory holding one sub-directory per run. All runs of a set
share one training condition (model size, optimizer, data), so they are compared against
the same reference: the baseline runs found in the set, with all their seeds pooled
(gram/compute_ratio.py).

Three tables are produced:

* ``results``: one row per run, with its settings, its raw validation losses and the
  compute ratios derived from them.
* ``losses``: one row per (run, profile, dataset): every raw loss, with its compute ratio.
* ``reference``: the fitted baseline curve and denominator of each dataset.

What the per-run columns mean (the inference profile of this project is core-only):

=========================  ===========================================================
``loss_core``              all modules ablated (or the dense model), on core data
``loss_forget_<topic>``    all modules ablated (or the dense model), on that topic
``loss_retain_<topic>``    core + that topic's module, on that topic (GRAM only)
``loss_elicit_<p>_<topic>`` forget loss after adversarial finetuning, protocol ``p``
``cr_...``                 the compute ratio of the corresponding loss
``cr_forget``, ``cr_retain``, ``cr_elicit_<p>``  the mean over the auxiliary topics
=========================  ===========================================================

Following the paper, an average is always an average of compute ratios (the ratio of
each topic first, then the mean), never the ratio of an averaged loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gram.compute_ratio import Reference, build_reference
from gram.config import CORE, RunConfig, load_config


# A module narrower than this share of the dense model's MLP counts as "small" in the tables.
SMALL_MODULE_SHARE = 0.05


def label_setting(config: RunConfig) -> str:
    """A short name for a run's label condition, for tables and plots."""
    if config.labels.noise != "none":
        return f"{config.labels.noise}={config.labels.noise_rate:g}"
    setting = f"labeled={config.labels.labeled_fraction:g}"
    # Stage 3's control run: GRAM with unlabeled data, trained for as long as the baseline.
    if config.method == "gram" and config.labels.labeled_fraction < 1 and not config.routing.equal_compute:
        setting += ", same steps"
    # Stage 3's other control: the small modules of the paper's partial-labeling experiment
    # (1.25% of the dense MLP's width, where the standard modules are 9.4%).
    if config.method == "gram" and config.model.aux_mlp_dim < SMALL_MODULE_SHARE * config.model.dense_mlp_dim:
        setting += ", small modules"
    return setting


def read_run(run_dir: Path) -> dict | None:
    """Everything a finished run recorded, or None if the run is not finished."""
    if not (run_dir / "DONE").exists():
        return None
    config = load_config(run_dir / "config.yaml")
    run = {
        "dir": run_dir,
        "config": config,
        "losses": pd.read_csv(run_dir / "eval" / "losses.csv"),
        "plan": json.loads((run_dir / "plan.json").read_text()),
        "elicitation": None,
        "curve": None,
        "checked": False,
    }
    if (run_dir / "eval" / "elicitation.csv").exists():
        run["elicitation"] = pd.read_csv(run_dir / "eval" / "elicitation.csv")
    if (run_dir / "metrics" / "validation_curve.parquet").exists():
        run["curve"] = pd.read_parquet(run_dir / "metrics" / "validation_curve.parquet")
    if (run_dir / "eval" / "check.json").exists():
        run["checked"] = bool(json.loads((run_dir / "eval" / "check.json").read_text())["ok"])
    return run


def read_runs(runs_dir: str | Path) -> list[dict]:
    runs = [read_run(path) for path in sorted(Path(runs_dir).iterdir()) if (path / "config.yaml").exists()]
    return [run for run in runs if run is not None]


def reference_from_baselines(runs: list[dict]) -> Reference:
    """Pool the set's baseline runs into the compute-ratio reference."""
    baselines = [run for run in runs if run["config"].method == "baseline"]
    if not baselines:
        raise ValueError("this set of runs has no finished baseline, so compute ratios cannot be formed")
    curve_rows, final_rows = [], []
    for run in baselines:
        if run["curve"] is None:
            raise ValueError(f"baseline {run['dir'].name} recorded no learning curve (eval.curve_points was 0)")
        seed = run["config"].seed
        curve_rows.append(run["curve"].assign(seed=seed)[["seed", "dataset", "step", "loss"]])
        final_rows.append(run["losses"].assign(seed=seed)[["seed", "dataset", "loss"]])
    return build_reference(pd.concat(curve_rows, ignore_index=True), pd.concat(final_rows, ignore_index=True))


def summarise_run(run: dict, reference: Reference) -> dict:
    """The one-row summary of a run: settings, raw losses, compute ratios."""
    config: RunConfig = run["config"]
    aux = list(config.data.aux_topics)
    loss = {(row.profile, row.dataset): row.loss for row in run["losses"].itertuples()}
    is_gram = config.method == "gram"
    ablated = "core_only" if is_gram else "dense"  # the profile in which nothing auxiliary is switched on

    row = {
        "run": run["dir"].name,
        "method": config.method,
        "seed": config.seed,
        "label_setting": label_setting(config),
        "labeled_fraction": config.labels.labeled_fraction,
        "noise": config.labels.noise,
        "noise_rate": config.labels.noise_rate,
        "batch_assembly": config.data.batch_assembly,
        "core_mlp_dim": config.model.core_mlp_dim if is_gram else config.model.dense_mlp_dim,
        "weight_decay": config.optim.weight_decay,
        "total_steps": run["plan"]["total_steps"],
        "checked_against_checkpoint": run["checked"],
    }

    losses = {"core": loss[(ablated, CORE)]}
    datasets = {"core": CORE}
    for topic in aux:
        losses[f"forget_{topic}"] = loss[(ablated, topic)]
        datasets[f"forget_{topic}"] = topic
        if is_gram:
            losses[f"retain_{topic}"] = loss[(f"only:{topic}", topic)]
            datasets[f"retain_{topic}"] = topic
    if run["elicitation"] is not None:
        for record in run["elicitation"].itertuples():
            losses[f"elicit_{record.protocol}_{record.topic}"] = record.loss_elicited
            datasets[f"elicit_{record.protocol}_{record.topic}"] = record.topic

    ratios = {name: reference.compute_ratio(value, datasets[name]) for name, value in losses.items()}
    row.update({f"loss_{name}": value for name, value in losses.items()})
    row.update({f"cr_{name}": value for name, value in ratios.items()})

    # Class averages: the mean of the per-topic compute ratios.
    def mean_ratio(prefix: str) -> float:
        values = [ratios[f"{prefix}_{topic}"] for topic in aux if f"{prefix}_{topic}" in ratios]
        return float(np.mean(values)) if values else float("nan")

    row["cr_forget"] = mean_ratio("forget")
    row["cr_retain"] = mean_ratio("retain")
    for protocol in sorted({name.split("_")[1] for name in ratios if name.startswith("elicit_")}):
        row[f"cr_elicit_{protocol}"] = mean_ratio(f"elicit_{protocol}")
    return row


def build_tables(runs_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, Reference]:
    """Read a set of runs and return (results, losses, reference). See the module docstring."""
    runs = read_runs(runs_dir)
    reference = reference_from_baselines(runs)

    results = pd.DataFrame([summarise_run(run, reference) for run in runs])
    leading = ["run", "method", "seed", "label_setting", "cr_core", "cr_retain", "cr_forget"]
    leading += sorted(column for column in results.columns if column.startswith("cr_elicit_") and column.count("_") == 2)
    results = results[leading + [column for column in results.columns if column not in leading]]

    long_rows = []
    for run in runs:
        table = run["losses"].assign(
            run=run["dir"].name,
            method=run["config"].method,
            seed=run["config"].seed,
            label_setting=label_setting(run["config"]),
        )
        table["compute_ratio"] = [reference.compute_ratio(row.loss, row.dataset) for row in table.itertuples()]
        long_rows.append(table)
    losses = pd.concat(long_rows, ignore_index=True)
    losses = losses[["run", "method", "seed", "label_setting", "profile", "active", "dataset", "num_sequences", "loss", "compute_ratio"]]
    return results, losses, reference


def per_module_ablation(losses: pd.DataFrame, aux_topics: list[str]) -> pd.DataFrame:
    """For the wrong-module experiments: what does ablating a single module do to each topic?

    One row per (GRAM run, topic), with the compute ratio on that topic when

    * nothing is ablated (``all_modules``),
    * only the topic's *own* module is ablated (``own_ablated``),
    * only one *other* module is ablated (one column per other module, and their mean),
    * every module is ablated (``all_ablated``; this is the forget number).

    If labels are right, ablating a topic's own module is what removes it. If some of its
    stories were labeled as another topic, part of it lives in that other module.
    """
    rows = []
    for (run, seed, setting), table in losses[losses.method == "gram"].groupby(["run", "seed", "label_setting"], sort=False):
        ratio = {(row.profile, row.dataset): row.compute_ratio for row in table.itertuples()}
        for topic in aux_topics:
            others = {other: ratio[(f"without:{other}", topic)] for other in aux_topics if other != topic}
            rows.append(
                {
                    "run": run,
                    "seed": seed,
                    "label_setting": setting,
                    "topic": topic,
                    "all_modules": ratio[("all_modules", topic)],
                    "own_ablated": ratio[(f"without:{topic}", topic)],
                    "other_ablated_mean": float(np.mean(list(others.values()))),
                    **{f"ablated:{other}": value for other, value in others.items()},
                    "all_ablated": ratio[("core_only", topic)],
                }
            )
    return pd.DataFrame(rows)
