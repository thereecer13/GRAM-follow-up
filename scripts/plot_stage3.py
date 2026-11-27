"""Plots for Stage 3: what happens as labels are removed.

    python scripts/plot_stage3.py results/stage3

reads ``results.csv`` and ``gradient_probes.parquet`` from the results directory (written
by ``scripts/make_results.py``) and writes, into ``<results>/plots/``:

* ``forget_vs_fraction.png``       forget and elicited forget against the labeled fraction
* ``core_retain_vs_fraction.png``  core and retain against the labeled fraction
* ``selectivity_vs_fraction.png``  is the forgetting selective? core against the topics, for both methods
* ``probes_labeled_<f>.png``       one per labeled fraction: the gradient probes of the GRAM run(s)

In the first two, every seed is drawn as a point and the line joins the means over seeds.
The dashed line at 1 is the baseline, which is the reference of the compute ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gram.config import CORE

# One colour per method everywhere (blue, orange), and per comparison in the probe plots.
COLORS = {"filtering": "#257de0", "gram": "#f47230"}
NAMES = {"filtering": "Filtering", "gram": "GRAM"}
PROBE_COLORS = {"matched": "#257de0", "core": "#f47230", "mismatched": "#00b894"}
GRAY = "#666666"
SAME_STEPS = ", same steps"


def style(axis, title: str, ylabel: str | None = None) -> None:
    axis.set_title(title, fontsize=10, loc="left")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# Metrics against the labeled fraction
# --------------------------------------------------------------------------- #


def plot_metric(axis, results: pd.DataFrame, column: str, methods: tuple[str, ...]) -> None:
    """One metric against the labeled fraction: a point per seed, a line through the seed means."""
    fractions = sorted(results.labeled_fraction.unique(), reverse=True)
    position = {fraction: index for index, fraction in enumerate(fractions)}  # evenly spaced, most labels first

    for method in methods:
        runs = results[(results.method == method) & results[column].notna()]
        main = runs[~runs.label_setting.str.endswith(SAME_STEPS)]
        control = runs[runs.label_setting.str.endswith(SAME_STEPS)]
        if main.empty:
            continue
        means = main.groupby("labeled_fraction")[column].mean()
        xs = [position[fraction] for fraction in means.index]
        order = sorted(range(len(xs)), key=lambda index: xs[index])
        axis.plot([xs[i] for i in order], [means.iloc[i] for i in order], color=COLORS[method], linewidth=2, label=NAMES[method])
        axis.scatter(
            [position[fraction] for fraction in main.labeled_fraction], main[column],
            s=36, color=COLORS[method], edgecolor="white", linewidth=1, zorder=3,
        )
        if not control.empty:
            axis.scatter(
                [position[fraction] for fraction in control.labeled_fraction], control[column],
                s=44, facecolor="white", edgecolor=COLORS[method], linewidth=1.8, zorder=3,
                label=f"{NAMES[method]}, same steps as the baseline",
            )

    axis.axhline(1.0, color=GRAY, linewidth=1, linestyle="--", label="Baseline")
    axis.set_xticks(range(len(fractions)), [f"{fraction:g}" for fraction in fractions])
    axis.set_xlim(-0.3, len(fractions) - 0.7)
    axis.set_xlabel("labeled fraction")


def metric_figure(results: pd.DataFrame, panels: list[tuple[str, str, tuple[str, ...]]], title: str, path: Path) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.9), sharey=True)
    for axis, (column, panel_title, methods) in zip(axes, panels):
        plot_metric(axis, results, column, methods)
        style(axis, panel_title)
    axes[0].set_ylabel("compute ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    seeds = sorted(results.seed.unique())
    figure.suptitle(f"{title}   (seeds: {', '.join(map(str, seeds))}; points are seeds, lines are means)", fontsize=10, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(path, dpi=160)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Is the forgetting selective?
# --------------------------------------------------------------------------- #

MEASURE_COLORS = {"core": "#257de0", "topics": "#f47230", "full": "#00b894"}


def selectivity_figure(results: pd.DataFrame, losses: pd.DataFrame, path: Path) -> None:
    """Core data against the auxiliary topics, as labels are removed.

    A method forgets selectively when the topics (orange) sit well below core data
    (blue). Three panels: the filtered model; GRAM with every module removed; GRAM with
    only one module removed, scored on that module's topic and on the other three.
    """
    fractions = sorted(results.labeled_fraction.unique(), reverse=True)
    position = {fraction: index for index, fraction in enumerate(fractions)}
    main = results[~results.label_setting.str.endswith(SAME_STEPS)]

    def draw(axis, table: pd.DataFrame, column: str, color: str, label: str) -> None:
        means = table.groupby("labeled_fraction")[column].mean().reindex(fractions).dropna()
        xs = [position[fraction] for fraction in means.index]
        axis.plot(xs, means.values, color=color, linewidth=2, label=label)
        axis.scatter([position[f] for f in table.labeled_fraction], table[column], s=36, color=color, edgecolor="white", linewidth=1, zorder=3)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)

    filtering = main[main.method == "filtering"]
    draw(axes[0], filtering, "cr_core", MEASURE_COLORS["core"], "core data")
    draw(axes[0], filtering, "cr_forget", MEASURE_COLORS["topics"], "the 4 topics")
    style(axes[0], "Filtering", "compute ratio")

    gram = main[main.method == "gram"]
    gram_losses = losses[(losses.method == "gram") & ~losses.label_setting.str.endswith(SAME_STEPS)]
    gram_losses = gram_losses.merge(results[["run", "labeled_fraction"]], on="run")
    full = gram_losses[(gram_losses.profile == "all_modules") & (gram_losses.dataset == CORE)].rename(columns={"compute_ratio": "cr_full"})
    draw(axes[1], full, "cr_full", MEASURE_COLORS["full"], "core data, all modules on")
    draw(axes[1], gram, "cr_core", MEASURE_COLORS["core"], "core data")
    draw(axes[1], gram, "cr_forget", MEASURE_COLORS["topics"], "the 4 topics")
    style(axes[1], "GRAM, every module removed")

    single = gram_losses[gram_losses.profile.str.startswith("without:") & (gram_losses.dataset != CORE)]
    removed = single.profile.str.removeprefix("without:")
    own = single[single.dataset == removed].groupby(["run", "labeled_fraction"], as_index=False).compute_ratio.mean()
    other = single[single.dataset != removed].groupby(["run", "labeled_fraction"], as_index=False).compute_ratio.mean()
    draw(axes[2], other, "compute_ratio", MEASURE_COLORS["core"], "the other 3 topics")
    draw(axes[2], own, "compute_ratio", MEASURE_COLORS["topics"], "the removed module's topic")
    style(axes[2], "GRAM, one module removed")

    for axis in axes:
        axis.axhline(1.0, color=GRAY, linewidth=1, linestyle="--")
        axis.set_xticks(range(len(fractions)), [f"{fraction:g}" for fraction in fractions])
        axis.set_xlim(-0.3, len(fractions) - 0.7)
        axis.set_xlabel("labeled fraction")
        axis.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    axes[0].annotate("baseline", (0, 1.0), xytext=(0, 4), textcoords="offset points", fontsize=8, color=GRAY)

    seeds = sorted(results.seed.unique())
    figure.suptitle(
        "Is the forgetting selective? A gap between blue and orange means the topics are removed and the rest is kept"
        f"   (seeds: {', '.join(map(str, seeds))})",
        fontsize=10, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Gradient probes
# --------------------------------------------------------------------------- #


def probe_figure(probes: pd.DataFrame, setting: str, path: Path) -> None:
    """The gradient probes of the GRAM runs with one label setting, averaged over topics and seeds.

    Left: does unlabeled data of topic i push each group the way labeled topic-i data
    does? Cosine between the two gradients, restricted to module i ("matched"), to the
    core, and to another module ("mismatched"). The gray line is the ceiling: the cosine
    between two unlabeled batches of the same topic on module i.

    Right: how hard does unlabeled data pull on a module? Gradient norm relative to the
    parameter norm, for a module on its own topic's data, on another auxiliary topic's
    data, and on core data.
    """
    aux_rows = probes[probes.topic != CORE]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    labels = {"matched": "module i (matched)", "core": "core", "mismatched": "another module (mismatched)"}
    for kind, label in labels.items():
        curve = aux_rows[aux_rows.cosine_kind == kind].groupby("step").cos_unlabeled_vs_labeled.mean()
        left.plot(curve.index, curve.values, color=PROBE_COLORS[kind], linewidth=2, label=label)
    ceiling = aux_rows[aux_rows.cosine_kind == "matched"].groupby("step").cos_unlabeled_a_vs_b.mean()
    left.plot(ceiling.index, ceiling.values, color=GRAY, linewidth=1, linestyle="--", label="ceiling: two unlabeled batches, module i")
    left.axhline(0, color="#b3b3b3", linewidth=0.8)
    style(left, "Cosine: unlabeled-rule vs labeled-rule gradient of topic i", "cosine similarity")
    left.set_xlabel("training step")
    left.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    modules = probes[probes.group != CORE]
    relative = modules.assign(relative=(modules.relative_grad_norm_a + modules.relative_grad_norm_b) / 2)
    sources = {
        "its own topic": relative[relative.topic == relative.group],
        "core": relative[relative.topic == CORE],
        "another auxiliary topic": relative[(relative.topic != CORE) & (relative.topic != relative.group)],
    }
    for (label, rows), color in zip(sources.items(), PROBE_COLORS.values()):
        curve = rows.groupby("step").relative.mean()
        right.plot(curve.index, curve.values, color=color, linewidth=2, label=f"data: {label}")
    right.set_yscale("log")
    style(right, "Pull on a module from unlabeled data, by the data's topic", "gradient norm / parameter norm")
    right.set_xlabel("training step")
    right.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    seeds = sorted(probes.seed.unique())
    figure.suptitle(f"Gradient probes, GRAM, {setting}   (mean over 4 topics; seeds: {', '.join(map(str, seeds))})", fontsize=10, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    plots = args.results_dir / "plots"
    plots.mkdir(exist_ok=True)

    results = pd.read_csv(args.results_dir / "results.csv")
    metric_figure(
        results,
        [
            ("cr_forget", "Forget", ("filtering", "gram")),
            ("cr_elicit_code", "Elicited forget (512 sequences, early stopping)", ("filtering", "gram")),
            ("cr_elicit_paper", "Elicited forget (128 sequences, 75 steps)", ("filtering", "gram")),
        ],
        "Forget: lower is better",
        plots / "forget_vs_fraction.png",
    )
    metric_figure(
        results,
        [("cr_core", "Core", ("filtering", "gram")), ("cr_retain", "Retain (GRAM: core + the topic's module)", ("gram",))],
        "Core and retain: higher is better",
        plots / "core_retain_vs_fraction.png",
    )

    losses = pd.read_csv(args.results_dir / "losses.csv")
    selectivity_figure(results, losses, plots / "selectivity_vs_fraction.png")

    probes = pd.read_parquet(args.results_dir / "gradient_probes.parquet")
    probes = probes[probes.method == "gram"]
    for setting, rows in probes.groupby("label_setting", sort=False):
        name = setting.replace("=", "_").replace(", ", "_").replace(" ", "_")
        probe_figure(rows, setting, plots / f"probes_{name}.png")
    print(f"wrote {len(list(plots.glob('*.png')))} plots to {plots}")


if __name__ == "__main__":
    main()
