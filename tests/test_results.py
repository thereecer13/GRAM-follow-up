"""From run directories to result tables."""

import dataclasses
import numpy as np
import pandas as pd
import pytest
from helpers import AUX, make_store, tiny_run_config

from gram.config import CORE
from gram.results import build_tables, label_setting, per_module_ablation, read_runs
from gram.run import run


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    """Two seeds of each method, plus one run that was interrupted and so must be ignored."""
    path = tmp_path_factory.mktemp("results")
    make_store(path / "store")
    directory = path / "runs" / "toy"
    for seed in (1, 2):
        for method in ("baseline", "filtering", "gram"):
            extra = {"eval__curve_points": 12} if method == "baseline" else {}
            config = tiny_run_config(path, method, seed=seed, **extra)
            config.name, config.output_dir = f"{method}_seed{seed}", str(directory)
            assert run(config) == "finished"
    unfinished = tiny_run_config(path, "gram", seed=3)
    unfinished.name, unfinished.output_dir = "gram_seed3", str(directory)
    assert run(unfinished, stop_after_steps=10) == "interrupted"
    return directory


@pytest.fixture(scope="module")
def tables(runs_dir):
    return build_tables(runs_dir)


def test_only_finished_runs_are_read(runs_dir):
    names = [run["dir"].name for run in read_runs(runs_dir)]
    assert len(names) == 6 and "gram_seed3" not in names


def test_one_row_per_run_with_settings_losses_and_ratios(tables):
    results, _, _ = tables
    assert len(results) == 6 and results.run.is_unique
    assert list(results.columns[:7]) == ["run", "method", "seed", "label_setting", "cr_core", "cr_retain", "cr_forget"]
    for topic in AUX:
        for kind in ("forget", "elicit_code", "elicit_paper"):
            assert f"loss_{kind}_{topic}" in results and f"cr_{kind}_{topic}" in results
    gram, dense = results[results.method == "gram"], results[results.method != "gram"]
    assert gram[[f"loss_retain_{topic}" for topic in AUX]].notna().all().all()
    assert dense.cr_retain.isna().all(), "a dense model has no module to switch on, so no retain number"
    assert (results.label_setting == "labeled=1").all() and not results.checked_against_checkpoint.any()


def test_class_ratios_are_means_of_per_topic_ratios(tables):
    results, _, reference = tables
    for _, row in results.iterrows():
        per_topic = [row[f"cr_forget_{topic}"] for topic in AUX]
        assert row.cr_forget == pytest.approx(np.mean(per_topic))
        # ... which is not the ratio of the mean loss: the averaging is done on ratios.
        assert row.cr_core == pytest.approx(reference.compute_ratio(row.loss_core, CORE))
    elicit = results[[f"cr_elicit_code_{topic}" for topic in AUX]].mean(axis=1)
    assert np.allclose(elicit, results.cr_elicit_code)


def test_the_pooled_baselines_average_to_one(tables):
    results, _, reference = tables
    baselines = results[results.method == "baseline"]
    assert reference.num_baseline_seeds == 2
    assert baselines.cr_core.mean() == pytest.approx(1.0, abs=1e-9)
    for topic in AUX:
        assert baselines[f"cr_forget_{topic}"].mean() == pytest.approx(1.0, abs=1e-9)
    assert set(reference.describe().columns) >= {"dataset", "A", "alpha", "s0", "fit_rms_log_error", "denominator_steps"}


def test_the_toy_result_has_the_expected_shape(tables):
    results, _, _ = tables
    mean = results.groupby("method")[["cr_core", "cr_forget", "cr_retain", "cr_elicit_code"]].mean()
    assert mean.loc["filtering", "cr_forget"] < 0.2 and mean.loc["gram", "cr_forget"] < 0.4, "removed topics are removed"
    assert mean.loc["gram", "cr_retain"] > mean.loc["gram", "cr_forget"] + 0.2, "a module brings its topic back"
    assert mean.loc["filtering", "cr_core"] > 0.8, "core performance is kept"
    assert (results.loc[results.method != "baseline", "cr_elicit_code"] >= results.loc[results.method != "baseline", "cr_forget"]).all()


def test_long_table_has_every_loss_with_its_ratio(tables):
    results, losses, reference = tables
    assert len(losses) == 2 * (5 + 5 + 50)
    row = losses[(losses.run == "gram_seed1") & (losses.profile == "core_only") & (losses.dataset == AUX[1])].iloc[0]
    summary = results.set_index("run").loc["gram_seed1"]
    assert row.loss == summary[f"loss_forget_{AUX[1]}"] and row.compute_ratio == pytest.approx(summary[f"cr_forget_{AUX[1]}"])
    assert row.compute_ratio == pytest.approx(reference.compute_ratio(row.loss, AUX[1]))


def test_per_module_ablation_table(tables):
    _, losses, _ = tables
    table = per_module_ablation(losses, list(AUX))
    assert len(table) == 2 * len(AUX) and set(table.run) == {"gram_seed1", "gram_seed2"}
    for _, row in table.iterrows():
        others = [row[f"ablated:{other}"] for other in AUX if other != row.topic]
        assert row.other_ablated_mean == pytest.approx(np.mean(others))
        assert np.isnan(row[f"ablated:{row.topic}"]), "a topic's own module is reported under own_ablated instead"
    # With correct labels, a topic is hurt by ablating its own module and not by ablating another.
    assert (table.own_ablated < table.other_ablated_mean).all()
    assert (table.all_ablated <= table.own_ablated + 0.15).all()


def test_label_setting_names(tmp_path):
    assert label_setting(tiny_run_config(tmp_path)) == "labeled=1"
    assert label_setting(tiny_run_config(tmp_path, labels__labeled_fraction=0.01)) == "labeled=0.01"
    noisy = tiny_run_config(tmp_path, labels__noise="false_negative", labels__noise_rate=0.2)
    assert label_setting(noisy) == "false_negative=0.2"
    # The two control runs of Stage 3 are told apart from the main runs by a suffix.
    same_steps = tiny_run_config(tmp_path, labels__labeled_fraction=0.01, routing__equal_compute=False)
    assert label_setting(same_steps) == "labeled=0.01, same steps"
    small = tiny_run_config(tmp_path, labels__labeled_fraction=0.1)
    small.model = dataclasses.replace(small.model, aux_mlp_dim=1)  # under 5% of the dense MLP's width
    assert label_setting(small) == "labeled=0.1, small modules"
