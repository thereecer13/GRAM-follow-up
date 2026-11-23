"""Compare a set of our runs with the numbers published for the paper's Simple Stories experiment.

    python scripts/compare_with_paper.py results/code_exact

Needs the tables written by scripts/make_results.py. The reference numbers
(configs/official_targets.json) are derived from the authors' published per-run results.
Two versions of them are shown:

* **like for like**: the authors' numbers restricted to this project's profiles (every
  module ablated; for filtering, the one model trained on core data only). This is the
  comparison that should match.
* **Table 1**: as printed in the paper, which pools five capability profiles. For
  filtering this includes four models that this project does not train, so a gap of
  about 0.015 is expected there.

The Stage 2 gate is GRAM's forget ratio: within 0.05 of the paper's 0.766.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gram.compute_ratio import mean_and_ci90

TARGETS = Path(__file__).resolve().parent.parent / "configs" / "official_targets.json"
GATE = 0.05
OUR_COLUMN = {"core": "cr_core", "retain": "cr_retain", "forget": "cr_forget", "elicit": "cr_elicit_code"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    results = pd.read_csv(args.results_dir / "results.csv")
    targets = json.loads(TARGETS.read_text())
    like_for_like, table_1 = targets["core_only_profile"], targets["table_1"]
    aux = [column[len("cr_forget_"):] for column in results.columns if column.startswith("cr_forget_")]

    print("Compute ratios. Ours: mean over our seeds (90% CI if more than one).")
    print("Authors: mean over their 3 seeds (90% CI) and their per-seed range.\n")
    header = f"{'':<22}{'ours':>16}{'authors, like for like':>40}{'diff':>8}{'Table 1':>10}"
    for method in ("gram", "filtering"):
        ours = results[results.method == method]
        if ours.empty:
            continue
        print(f"{method.upper()}  ({len(ours)} seed{'s' if len(ours) > 1 else ''}: {sorted(ours.seed)})")
        print(header)
        for metric, column in OUR_COLUMN.items():
            if metric not in like_for_like[method]:
                continue
            mean, ci = mean_and_ci90(ours[column])
            theirs = like_for_like[method][metric]["compute_ratio"]
            ours_text = f"{mean:.3f}" + (f" ({ci:.3f})" if len(ours) > 1 else "")
            theirs_text = f"{theirs['mean']:.3f} ({theirs['ci90']:.3f})  [{min(theirs['per_seed']):.3f} .. {max(theirs['per_seed']):.3f}]"
            printed = table_1[method][metric]["compute_ratio"]["mean"]
            print(f"  {metric:<20}{ours_text:>16}{theirs_text:>40}{mean - theirs['mean']:>+8.3f}{printed:>10.3f}")
        if "cr_elicit_paper" in ours:
            mean, _ = mean_and_ci90(ours["cr_elicit_paper"])
            print(f"  {'elicit (paper protocol)':<20}{mean:>13.3f}   (128 sequences, 75 steps; the authors' published numbers use the other protocol)")

        print(f"\n  per topic{'':<24}{'forget: ours':>14}{'authors':>10}{'elicit: ours':>16}{'authors':>10}")
        for topic in aux:
            forget, _ = mean_and_ci90(ours[f"cr_forget_{topic}"])
            elicit, _ = mean_and_ci90(ours[f"cr_elicit_code_{topic}"])
            their_forget = like_for_like[method]["forget"]["per_topic"][topic]["compute_ratio"]["mean"]
            their_elicit = like_for_like[method]["elicit"]["per_topic"][topic]["compute_ratio"]["mean"]
            print(f"    {topic:<30}{forget:>13.3f}{their_forget:>10.3f}{elicit:>16.3f}{their_elicit:>10.3f}")
        print()

    print("Raw validation losses (nats per token). One hundredth of a nat is about 4.5% in compute ratio.")
    print(f"  {'':<34}{'ours':>10}{'authors':>10}{'diff':>9}")
    baseline = results[results.method == "baseline"]
    if not baseline.empty:
        for name, column in [("core", "loss_core")] + [(topic, f"loss_forget_{topic}") for topic in aux]:
            theirs = targets["baseline_final_loss"][name]
            print(f"  baseline on {name:<22}{baseline[column].mean():>10.4f}{theirs:>10.4f}{baseline[column].mean() - theirs:>+9.4f}")
    for method in ("gram", "filtering"):
        ours = results[results.method == method]
        if ours.empty:
            continue
        rows = [("core", ours.loss_core.mean(), like_for_like[method]["core"]["loss"]["mean"])]
        rows.append(("forget (mean of 4)", ours[[f"loss_forget_{t}" for t in aux]].mean().mean(), like_for_like[method]["forget"]["loss"]["mean"]))
        if method == "gram":
            rows.append(("retain (mean of 4)", ours[[f"loss_retain_{t}" for t in aux]].mean().mean(), like_for_like[method]["retain"]["loss"]["mean"]))
        for name, mine, theirs in rows:
            print(f"  {method + ' ' + name:<34}{mine:>10.4f}{theirs:>10.4f}{mine - theirs:>+9.4f}")

    gram = results[results.method == "gram"]
    if not gram.empty:
        forget = gram.cr_forget.mean()
        printed = table_1["gram"]["forget"]["compute_ratio"]["mean"]
        passed = abs(forget - printed) <= GATE
        print(f"\nStage 2 gate: GRAM forget {forget:.3f} against the paper's {printed:.3f}: "
              f"difference {forget - printed:+.3f}, tolerance {GATE} -> {'PASS' if passed else 'OUTSIDE TOLERANCE, DIAGNOSE'}")

    unchecked = results.loc[~results.checked_against_checkpoint, "run"].tolist()
    if unchecked:
        print(f"\nWARNING: not yet re-evaluated from their checkpoints: {', '.join(unchecked)}")
        print("Run scripts/make_results.py with --check before reporting these numbers.")


if __name__ == "__main__":
    main()
