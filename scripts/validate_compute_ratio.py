"""Check our compute-ratio code against the numbers the paper's authors published.

The official repository contains, for the Simple Stories experiment, the baseline
learning curves, every model's raw validation losses, and the compute ratios the authors
derived from them (the ones behind Table 1). This script feeds the *raw* losses and
curves through our own implementation (gram/compute_ratio.py) and compares the results
with their published compute ratios, row by row, and then with Table 1 itself.

    git clone https://github.com/agencyenterprise/modular-pretraining /path/to/clone
    python scripts/validate_compute_ratio.py --official-repo /path/to/clone

Nothing from the official repository is copied into this one. The learning curves are
stored there as Python pickles; they are read with an unpickler that accepts NumPy
arrays and nothing else, so loading them cannot run code.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from gram.compute_ratio import build_reference, mean_and_ci90

SEEDS = (1, 2, 3)


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Loads pickles made of plain NumPy arrays, dicts, lists and numbers; refuses anything else."""

    ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def baseline_run_dir(seed_dir: Path) -> Path:
    """The run directory holding this seed's baseline (the earliest one with a learning curve)."""
    candidates = [d for d in seed_dir.iterdir() if (d / "baseline" / "losses.pkl").exists()]
    return sorted(candidates, key=lambda d: d.name)[0]


def load_official_baselines(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows, final_rows = [], []
    for seed in SEEDS:
        run_dir = baseline_run_dir(results / f"seed_{seed}")
        with open(run_dir / "baseline" / "losses.pkl", "rb") as handle:
            curves = NumpyOnlyUnpickler(handle).load()["val"]
        for dataset, points in curves.items():
            for step, loss in np.asarray(points, dtype=float):
                curve_rows.append({"seed": seed, "dataset": dataset, "step": step, "loss": loss})
        with open(run_dir / "stats.jsonl") as handle:
            for line in handle:
                record = json.loads(line)
                if (record.get("stage") or {}).get("name") == "baseline" and not record.get("elicited"):
                    final_rows.append({"seed": seed, "dataset": record["data_label"], "loss": record["loss"]})
    return pd.DataFrame(curve_rows), pd.DataFrame(final_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official-repo", type=Path, required=True)
    args = parser.parse_args()

    curve_points, final_losses = load_official_baselines(args.official_repo / "results" / "stories")
    reference = build_reference(curve_points, final_losses)
    print(f"baseline learning curves: {len(curve_points)} points "
          f"({curve_points.groupby(['seed', 'dataset']).size().iloc[0]} per seed and dataset, pooled over {len(SEEDS)} seeds)\n")
    print(reference.describe().round(4).to_string(index=False), "\n")

    published = pd.read_csv(args.official_repo / "analysis" / "stories" / "stories.csv")
    published["ours"] = [reference.compute_ratio(loss, dataset) for loss, dataset in zip(published.loss, published.data_label)]
    published["difference"] = (published.ours - published.compute_ratio).abs()
    worst = published.difference.max()
    print(f"{len(published)} published (model, seed, profile, dataset) compute ratios recomputed from raw losses")
    print(f"largest |ours - published| = {worst:.2e}")
    print(published.groupby("method").difference.max().rename("max difference per method").round(8).to_string(), "\n")

    print("Table 1, recomputed from the raw losses with our code (mean over 3 seeds, 90% CI half-width):")
    table1 = {
        ("GRAM", "core"): (0.938, 0.029), ("GRAM", "retain"): (0.952, 0.030),
        ("GRAM", "forget"): (0.766, 0.017), ("GRAM", "elicited_forget"): (0.855, 0.024),
        ("Filtering", "core"): (0.961, 0.015), ("Filtering", "retain"): (0.962, 0.024),
        ("Filtering", "forget"): (0.780, 0.009), ("Filtering", "elicited_forget"): (0.870, 0.012),
    }
    method_name = {"GRAM": "grmoe", "Filtering": "filtering"}
    table_ok = True
    for (method, label_class), (paper_mean, paper_ci) in table1.items():
        rows = published[(published.method == method_name[method]) & (published.label_class == label_class)]
        mean, ci = mean_and_ci90(rows.groupby("seed").ours.mean())
        ok = abs(mean - paper_mean) < 6e-4 and abs(ci - paper_ci) < 6e-4
        table_ok &= ok
        print(f"  {'ok  ' if ok else 'FAIL'} {method:<10} {label_class:<16} ours {mean:.3f} ({ci:.3f})   paper {paper_mean:.3f} ({paper_ci:.3f})")

    print()
    if worst > 1e-6 or not table_ok:
        raise SystemExit("MISMATCH between our compute ratios and the published ones")
    print("Our compute-ratio implementation reproduces the published values.")


if __name__ == "__main__":
    main()
