"""Turn a set of finished runs into result tables.

    python scripts/make_results.py runs/code_exact --out results/code_exact --check

Reads every finished run in the given directory, pools its baseline runs into the
compute-ratio reference, and writes:

    results.csv             one row per run: settings, raw losses, compute ratios
    losses.csv              one row per (run, profile, dataset): every raw loss and its ratio
    reference.csv           the fitted baseline curve of each dataset
    per_module_ablation.csv for GRAM runs: the effect of ablating single modules
    gradient_probes.parquet the gradient probes of all runs: one row per (run, step, topic, group)
    summary.csv             mean and 90% CI over seeds, per method and label setting
    runs/<run>/             the small files of every run (configuration, plan, training
                            metrics, learning curve, gradient probes, losses, elicitation
                            curves), so that the results directory can be committed and
                            analysed without the checkpoints

``--check`` first re-evaluates every run from its saved checkpoint and stops if any
stored loss is not reproduced. Runs that have never passed this check are listed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from gram.compute_ratio import mean_and_ci90
from gram.results import build_tables, label_setting, per_module_ablation, read_runs

HEADLINE = ["cr_core", "cr_retain", "cr_forget", "cr_elicit_code", "cr_elicit_paper"]

# Everything a run writes except its checkpoints, its split manifest and its batch log,
# which are large and stay in the run directory.
SMALL_FILES = [
    "config.yaml", "plan.json", "cells.csv", "train_summary.json", "log.txt",
    "metrics/train.parquet", "metrics/validation_curve.parquet", "probes/gradient_probes.parquet",
    "eval/losses.csv", "eval/elicitation.csv", "eval/elicitation_curves.csv", "eval/check.json",
]


def export_small_files(runs_dir: Path, out: Path) -> None:
    for run in read_runs(runs_dir):
        for relative in SMALL_FILES:
            source = run["dir"] / relative
            if source.exists():
                target = out / "runs" / run["dir"].name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


def summarise_over_seeds(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, setting), group in results.groupby(["method", "label_setting"], sort=False):
        row = {"method": method, "label_setting": setting, "seeds": len(group)}
        for column in HEADLINE:
            if column in group:
                row[column], row[f"{column}_ci90"] = mean_and_ci90(group[column])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="re-evaluate every run from its checkpoint first")
    parser.add_argument(
        "--check-new", action="store_true",
        help="like --check, but skip runs that already passed it (their eval/check.json exists)",
    )
    args = parser.parse_args()

    if args.check or args.check_new:
        from gram.run import evaluate_run  # needs PyTorch and the data; the tables themselves do not

        for run in read_runs(args.runs_dir):
            config = run["config"]
            if args.check_new and (config.run_dir / "eval" / "check.json").exists():
                print(f"checked  {config.name}: passed the re-evaluation earlier")
                continue
            evaluate_run(config, check=True)
        print()

    results, losses, reference = build_tables(args.runs_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "results.csv", index=False)
    losses.to_csv(args.out / "losses.csv", index=False)
    reference.describe().to_csv(args.out / "reference.csv", index=False)
    summary = summarise_over_seeds(results)
    summary.to_csv(args.out / "summary.csv", index=False)
    if (losses.method == "gram").any():
        aux = [column[len("cr_forget_"):] for column in results.columns if column.startswith("cr_forget_")]
        per_module_ablation(losses, aux).to_csv(args.out / "per_module_ablation.csv", index=False)
    export_small_files(args.runs_dir, args.out)

    probe_tables = []
    for run in read_runs(args.runs_dir):
        path = run["dir"] / "probes" / "gradient_probes.parquet"
        if path.exists():
            table = pd.read_parquet(path)
            table["run"] = run["dir"].name
            table.insert(1, "method", run["config"].method)
            table.insert(2, "seed", run["config"].seed)
            table.insert(3, "label_setting", label_setting(run["config"]))
            probe_tables.append(table[["run"] + [column for column in table.columns if column != "run"]])
    if probe_tables:
        pd.concat(probe_tables, ignore_index=True).to_parquet(args.out / "gradient_probes.parquet", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"{len(results)} finished runs in {args.runs_dir}; reference pooled over "
              f"{reference.num_baseline_seeds} baseline seed(s)\n")
        print(reference.describe().round(4).to_string(index=False), "\n")
        shown = ["run", "total_steps"] + [column for column in HEADLINE if column in results]
        print(results[shown].round(4).to_string(index=False), "\n")
        print(summary.round(4).to_string(index=False))

    unchecked = results.loc[~results.checked_against_checkpoint, "run"].tolist()
    if unchecked:
        print(f"\nNOT YET RE-EVALUATED FROM THEIR CHECKPOINT ({len(unchecked)}): {', '.join(unchecked)}")
        print("Run this script with --check before reporting these numbers.")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
