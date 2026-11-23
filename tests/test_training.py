"""The training loop: schedule, checkpoints, resuming, and that measurements do not disturb training."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import AUX, make_store, tiny_run_config

from gram.analysis import load_model, story_exposures
from gram.checkpoint import final_checkpoint_path, latest_checkpoint_path, load_checkpoint, model_from_checkpoint
from gram.config import CORE, UNLABELED, OptimConfig, load_config
from gram.probes import probe_steps
from gram.train import curve_steps, learning_rate, small_cell_warnings, spaced_checkpoint_steps, train


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    path = tmp_path_factory.mktemp("training")
    make_store(path / "store")
    return path


def config_for(workspace, name, method="gram", **overrides):
    config = tiny_run_config(workspace, method, **overrides)
    config.name = name
    return config


def assert_same_checkpoint(first, second):
    """Two checkpoints hold bit-for-bit the same model and the same optimizer states."""
    assert first["step"] == second["step"]
    assert first["model"].keys() == second["model"].keys()
    for name in first["model"]:
        assert torch.equal(first["model"][name], second["model"][name]), f"parameter {name} differs"
    for group in first["optimizers"]:
        old, new = first["optimizers"][group]["state"], second["optimizers"][group]["state"]
        for index in old:
            for field in old[index]:
                assert torch.equal(torch.as_tensor(old[index][field]), torch.as_tensor(new[index][field]))


# ------------------------------------------------------------------ schedule


def test_learning_rate_schedule_matches_the_official_logs():
    optim = OptimConfig()  # lr 5e-3, 10% warm-up, 10% decay
    # Baseline, 16544 steps: "warmup_steps: 1654, constant_steps: 13236, decay_steps: 1654".
    total = 16544
    assert learning_rate(0, total, optim) == pytest.approx(1e-8)
    assert learning_rate(1653, total, optim) < 5e-3 == learning_rate(1654, total, optim)
    # The decay phase starts at the full rate (progress 0) and falls from the next step on.
    assert learning_rate(1654 + 13236, total, optim) == 5e-3 > learning_rate(1654 + 13236 + 1, total, optim)
    assert 0 < learning_rate(total - 1, total, optim) < 1e-5
    # GRAM, 16021 steps: the log prints "Step: 1, LR: 3.1311e-06" and, at the end, "LR: 1.0000e-08".
    assert learning_rate(1, 16021, optim) == pytest.approx(3.1311e-06, rel=1e-4)
    assert learning_rate(16021, 16021, optim) == pytest.approx(1e-8)
    rates = [learning_rate(step, total, optim) for step in range(total)]
    assert rates[:1655] == sorted(rates[:1655]) and rates[-1654:] == sorted(rates[-1654:], reverse=True)


def test_checkpoint_probe_and_curve_steps_are_evenly_spaced():
    assert spaced_checkpoint_steps(16544, 10) == [0, 1654, 3309, 4963, 6618, 8272, 9926, 11581, 13235, 14890]
    probes = probe_steps(16544, 20)
    assert len(probes) == 20 and probes[-1] == 16544 and probes[0] == 827
    assert set(spaced_checkpoint_steps(16544, 10)[1:]) <= set(probes), "every kept checkpoint has a probe at the same step"
    curve = curve_steps(16544, 200)
    assert len(curve) == 200 and curve[-1] == 16544
    assert np.ptp(np.diff(curve)) <= 1


# ------------------------------------------------------------- what is saved


@pytest.fixture(scope="module")
def finished_gram(workspace):
    config = config_for(workspace, "gram_reference")
    assert train(config) == "finished"
    return config


def test_a_run_saves_everything_needed_later(finished_gram):
    run_dir = finished_gram.run_dir
    plan = json.loads((run_dir / "plan.json").read_text())

    # The configuration, exactly as used.
    assert load_config(run_dir / "config.yaml") == finished_gram

    # Checkpoints: the evenly spaced ones (the first is the initialisation) plus the final one.
    saved = sorted(path.name for path in (run_dir / "checkpoints").iterdir())
    assert saved == sorted([f"step_{step:07d}.pt" for step in plan["checkpoint_steps"]] + ["final.pt"])
    assert plan["checkpoint_steps"][0] == 0 and len(plan["checkpoint_steps"]) == 4
    assert not latest_checkpoint_path(run_dir).exists(), "the rolling checkpoint is removed at the end"

    # Each checkpoint has all parameter groups and all optimizer states.
    checkpoint = load_checkpoint(final_checkpoint_path(run_dir))
    assert set(checkpoint["parameter_groups"]) == set(checkpoint["optimizers"]) == {CORE, *AUX}
    listed = [name for names in checkpoint["parameter_groups"].values() for name in names]
    assert sorted(listed) == sorted(checkpoint["model"]), "the groups partition the saved parameters"
    assert checkpoint["step"] == checkpoint["total_steps"] == plan["total_steps"]

    # The split manifest.
    manifest = pd.read_parquet(run_dir / "manifest.parquet")
    assert {"story_id", "true_topic", "true_label", "given_label", "seed"} <= set(manifest.columns)

    # Per-step training metrics, one row per micro-batch.
    metrics = pd.read_parquet(run_dir / "metrics" / "train.parquet")
    assert metrics.step.tolist() == list(range(1, plan["total_steps"] + 1))
    assert set(metrics.rule) == {"core_plain", "core_robust", "aux_isolated", "aux_spread"}
    assert metrics.loss.iloc[-20:].mean() < metrics.loss.iloc[:20].mean() - 0.5, "the model learns"

    # Gradient probes at the planned steps.
    probes = pd.read_parquet(run_dir / "probes" / "gradient_probes.parquet")
    assert sorted(probes.step.unique()) == plan["probe_steps"]
    # Tidy: exactly one row per (run, step, topic, group).
    assert list(probes.columns[:4]) == ["run", "step", "topic", "group"] and (probes.run == "gram_reference").all()
    assert not probes.duplicated(["run", "step", "topic", "group"]).any()
    assert len(probes) == len(plan["probe_steps"]) * 5 * 5


def test_checkpoints_are_states_along_one_trajectory(finished_gram):
    run_dir = finished_gram.run_dir
    steps = json.loads((run_dir / "plan.json").read_text())["checkpoint_steps"]
    models = [model_from_checkpoint(run_dir / "checkpoints" / f"step_{step:07d}.pt")[0] for step in steps]
    models.append(model_from_checkpoint(final_checkpoint_path(run_dir))[0])
    embeddings = [model.embed.weight for model in models]
    assert all(not torch.equal(a, b) for a, b in zip(embeddings, embeddings[1:])), "every checkpoint is a new state"
    # The first checkpoint is the initialisation: untrained weights, standard deviation 0.02.
    assert embeddings[0].std().item() == pytest.approx(0.02, rel=0.1)


def test_batch_log_and_manifest_say_which_stories_were_trained_without_a_label(workspace):
    config = config_for(workspace, "gram_half_labeled", labels__labeled_fraction=0.5)
    assert train(config) == "finished"
    run_dir = config.run_dir
    log = np.load(run_dir / "batch_log.npz")
    cells = pd.read_csv(run_dir / "cells.csv")
    manifest = pd.read_parquet(run_dir / "manifest.parquet")

    # Which given label does each row of the sequence table carry?
    given_of_row = np.empty(int((cells.first_sequence + cells.num_sequences).max()), dtype=object)
    for cell in cells.itertuples():
        given_of_row[cell.first_sequence : cell.first_sequence + cell.num_sequences] = cell.given_label

    rule_of_batch = log["rule_names"][log["rule_index"]]
    pool_of_batch = log["rule_pools"][log["rule_index"]]
    for rows, rule, pool in zip(log["sequence_rows"], rule_of_batch, pool_of_batch):
        assert (given_of_row[rows] == pool).all(), "a batch only holds sequences of its own pool"
        assert (rule == "unlabeled") == (pool == UNLABELED)

    # Every unlabeled micro-batch ran with all modules active and all groups updated.
    unlabeled_rule = list(log["rule_names"]).index("unlabeled")
    assert log["rule_active"][unlabeled_rule] == log["rule_update"][unlabeled_rule] == "+".join([CORE, *AUX])

    # From sequences back to stories: the stories seen without a label are exactly those whose
    # sequences were trained under the unlabeled rule, and all of them were given no label.
    rows_trained_unlabeled = np.unique(log["sequence_rows"][rule_of_batch == "unlabeled"])
    seen = manifest[
        [np.isin(np.arange(first, last + 1), rows_trained_unlabeled).any() if first >= 0 else False
         for first, last in zip(manifest.first_sequence, manifest.last_sequence)]
    ]
    assert len(seen) > 100 and (seen.given_label == UNLABELED).all()
    assert manifest.given_label.eq(UNLABELED).mean() == pytest.approx(0.5, abs=0.01)


def test_story_exposures_lists_every_time_a_story_was_trained_on(workspace):
    config = config_for(workspace, "gram_exposures", labels__labeled_fraction=0.5, probes__enabled=False)
    assert train(config) == "finished"
    run_dir = config.run_dir
    exposures = story_exposures(run_dir)
    manifest = pd.read_parquet(run_dir / "manifest.parquet").set_index("story_id")
    log = np.load(run_dir / "batch_log.npz")

    # Check a sample of rows against the batch log directly.
    for row in exposures.sample(200, random_state=0).itertuples():
        assert row.sequence_row in log["sequence_rows"][row.micro_batch]
        assert log["rule_names"][log["rule_index"][row.micro_batch]] == row.rule
        assert manifest.first_sequence[row.story_id] <= row.sequence_row <= manifest.last_sequence[row.story_id]
        assert row.step == row.micro_batch + 1  # one micro-batch per step in this run

    # Nothing is missing: every (training sequence, micro-batch) pair appears for every story in that sequence.
    pairs_in_log = log["sequence_rows"].size
    stories_per_row = exposures.groupby(["sequence_row", "micro_batch"]).story_id.nunique()
    assert len(stories_per_row) == pairs_in_log and (stories_per_row >= 1).all()

    # The question the manifest exists to answer: which stories were trained on without a label?
    without_label = exposures[exposures.rule == "unlabeled"]
    assert (without_label.given_label == UNLABELED).all() and (without_label.active == "+".join([CORE, *AUX])).all()
    assert set(exposures[exposures.given_label == UNLABELED].rule) == {"unlabeled"}
    assert 0.35 < without_label.story_id.nunique() / len(manifest) < 0.5  # half are unlabeled; a few fall in unused tails
    never_trained = set(manifest.index) - set(exposures.story_id)
    assert 0 < len(never_trained) < 0.25 * len(manifest)


def test_load_model_returns_any_kept_checkpoint(finished_gram):
    run_dir = finished_gram.run_dir
    final, config = load_model(run_dir)
    assert config == finished_gram and next(final.parameters()).dtype == torch.float32 and not final.training
    initial, _ = load_model(run_dir, step=0)
    assert not torch.equal(initial.embed.weight, final.embed.weight)
    with pytest.raises(FileNotFoundError, match="kept steps are"):
        load_model(run_dir, step=7)


# ------------------------------------------------------------------ resuming


def test_an_interrupted_run_resumes_to_exactly_the_same_result(workspace, finished_gram):
    config = config_for(workspace, "gram_interrupted")
    total = json.loads((finished_gram.run_dir / "plan.json").read_text())["total_steps"]

    # Interrupt twice, at steps that are neither kept checkpoints nor on the rolling-checkpoint schedule.
    assert train(config, stop_after_steps=23) == "interrupted"
    assert load_checkpoint(latest_checkpoint_path(config.run_dir))["step"] == 23
    assert not final_checkpoint_path(config.run_dir).exists()
    assert train(config, stop_after_steps=88) == "interrupted"
    assert train(config) == "finished"
    assert train(config) == "finished", "running a finished run again does nothing"

    assert_same_checkpoint(
        load_checkpoint(final_checkpoint_path(finished_gram.run_dir)), load_checkpoint(final_checkpoint_path(config.run_dir))
    )
    for table in ("metrics/train.parquet", "probes/gradient_probes.parquet"):
        # Identical in everything but the name of the run.
        straight = pd.read_parquet(finished_gram.run_dir / table).drop(columns="run", errors="ignore")
        resumed = pd.read_parquet(config.run_dir / table).drop(columns="run", errors="ignore")
        pd.testing.assert_frame_equal(straight, resumed)
    assert len(pd.read_parquet(config.run_dir / "metrics/train.parquet")) == total
    assert sorted(p.name for p in (config.run_dir / "checkpoints").iterdir()) == sorted(
        p.name for p in (finished_gram.run_dir / "checkpoints").iterdir()
    )


def test_resuming_with_gradient_accumulation(workspace):
    settings = dict(optim__batch_size=12, optim__micro_batch_size=4)
    straight = config_for(workspace, "gram_accumulate_straight", **settings)
    resumed = config_for(workspace, "gram_accumulate_resumed", **settings)
    assert train(straight) == "finished"
    assert train(resumed, stop_after_steps=17) == "interrupted"
    assert train(resumed) == "finished"
    assert_same_checkpoint(
        load_checkpoint(final_checkpoint_path(straight.run_dir)), load_checkpoint(final_checkpoint_path(resumed.run_dir))
    )
    metrics = pd.read_parquet(straight.run_dir / "metrics/train.parquet")
    assert metrics.groupby("step").size().eq(3).all(), "three micro-batches per optimizer step"
    assert metrics.groupby("step").rule.nunique().gt(1).mean() > 0.5, "micro-batches of one step follow different rules"


# ------------------------------------------ measurements do not affect training


def test_gradient_probes_do_not_change_the_training_run(workspace, finished_gram):
    without = config_for(workspace, "gram_no_probes", probes__enabled=False)
    assert train(without) == "finished"
    assert not (without.run_dir / "probes").exists()
    assert_same_checkpoint(
        load_checkpoint(final_checkpoint_path(finished_gram.run_dir)), load_checkpoint(final_checkpoint_path(without.run_dir))
    )


def test_validation_sweeps_do_not_change_the_training_run(workspace):
    quiet = config_for(workspace, "baseline_quiet", method="baseline", probes__enabled=False)
    measured = config_for(workspace, "baseline_measured", method="baseline", eval__curve_points=7)
    assert train(quiet) == "finished" and train(measured) == "finished"
    assert_same_checkpoint(
        load_checkpoint(final_checkpoint_path(quiet.run_dir)), load_checkpoint(final_checkpoint_path(measured.run_dir))
    )
    curve = pd.read_parquet(measured.run_dir / "metrics" / "validation_curve.parquet")
    assert curve.step.nunique() == 7 and set(curve.dataset) == {CORE, *AUX}
    first, last = curve[curve.step == curve.step.min()], curve[curve.step == curve.step.max()]
    assert last.loss.mean() < first.loss.mean(), "validation loss falls during training"


def test_the_same_seed_gives_the_same_run_and_another_seed_a_different_one(workspace, finished_gram):
    same = config_for(workspace, "gram_same_seed")
    other = config_for(workspace, "gram_other_seed", seed=2)
    assert train(same) == "finished" and train(other) == "finished"
    reference = load_checkpoint(final_checkpoint_path(finished_gram.run_dir))
    assert_same_checkpoint(reference, load_checkpoint(final_checkpoint_path(same.run_dir)))
    different = load_checkpoint(final_checkpoint_path(other.run_dir))
    assert not torch.equal(reference["model"]["embed.weight"], different["model"]["embed.weight"])


# ------------------------------------------------- methods see the same data


def test_every_method_gets_the_identical_label_split(workspace):
    settings = dict(labels__labeled_fraction=0.1, probes__enabled=False)
    gram = config_for(workspace, "split_gram", **settings)
    filtering = config_for(workspace, "split_filtering", method="filtering", **settings)
    assert train(gram) == "finished" and train(filtering) == "finished"
    pd.testing.assert_frame_equal(
        pd.read_parquet(gram.run_dir / "manifest.parquet"), pd.read_parquet(filtering.run_dir / "manifest.parquet")
    )


def test_filtering_trains_on_core_and_unlabeled_data_only(workspace):
    config = config_for(workspace, "filtering_pools", method="filtering", labels__labeled_fraction=0.5, probes__enabled=False)
    assert train(config) == "finished"
    plan = json.loads((config.run_dir / "plan.json").read_text())
    assert {rule["pool"] for rule in plan["rules"]} == {CORE, UNLABELED}
    assert all(rule["name"] == "dense" for rule in plan["rules"])

    # With wrong labels, filtering follows the *given* label: false positives leave, false negatives stay.
    noisy = config_for(
        workspace, "filtering_false_positive", method="filtering", probes__enabled=False,
        labels__noise="false_positive", labels__noise_rate=0.2,
    )
    assert train(noisy) == "finished"
    manifest = pd.read_parquet(noisy.run_dir / "manifest.parquet")
    log = np.load(noisy.run_dir / "batch_log.npz")
    trained_rows = np.unique(log["sequence_rows"])
    removed = manifest[(manifest.true_label == CORE) & (manifest.given_label != CORE) & (manifest.first_sequence >= 0)]
    assert len(removed) > 50
    assert not np.isin(removed.first_sequence, trained_rows).any(), "core stories given an auxiliary label are filtered out"


# ------------------------------------------------------------------- safeguards


def test_a_run_directory_cannot_be_reused_with_different_settings(workspace, finished_gram):
    changed = config_for(workspace, "gram_reference", routing__p_as=0.5)
    with pytest.raises(ValueError, match="different configuration.*routing.p_as"):
        train(changed)
    other_seed = config_for(workspace, "gram_reference", seed=7)
    with pytest.raises(ValueError, match="differs in: seed"):
        train(other_seed)

    # Settings that cannot change the results may differ: the run is simply recognised as finished.
    harmless = config_for(workspace, "gram_reference", eval__batch_size=3, checkpoints__resume_every_steps=11)
    harmless.device = "auto"
    assert train(harmless) == "finished"


def test_an_interrupted_run_cannot_be_continued_with_different_settings(workspace):
    config = config_for(workspace, "gram_changed_midway")
    assert train(config, stop_after_steps=20) == "interrupted"
    changed = config_for(workspace, "gram_changed_midway", optim__weight_decay=0.1)
    with pytest.raises(ValueError, match="optim.weight_decay"):
        train(changed)
    assert train(config) == "finished"


def test_single_topic_batches_warn_when_small_label_cells_lose_their_data(workspace):
    """At a low false-positive rate every (auxiliary label, core topic) cell is smaller than one
    batch, so single-topic batching would silently train on none of the mislabeled stories."""
    settings = dict(labels__noise="false_positive", labels__noise_rate=0.05, probes__enabled=False)
    single = config_for(workspace, "warn_single_topic", data__batch_assembly="single_topic", **settings)
    mixed = config_for(workspace, "warn_mixed", data__batch_assembly="mixed", **settings)
    assert train(single) == "finished" and train(mixed) == "finished"

    warnings = json.loads((single.run_dir / "plan.json").read_text())["warnings"]
    assert len(warnings) >= 4 and all("never trained on" in line for line in warnings)
    assert "WARNING" in (single.run_dir / "log.txt").read_text()
    assert json.loads((mixed.run_dir / "plan.json").read_text())["warnings"] == []

    def mislabeled_stories_trained(config):
        manifest = pd.read_parquet(config.run_dir / "manifest.parquet")
        trained = np.unique(np.load(config.run_dir / "batch_log.npz")["sequence_rows"])
        wrong = manifest[(manifest.true_label == CORE) & (manifest.given_label != CORE) & (manifest.first_sequence >= 0)]
        return np.isin(wrong.first_sequence, trained).mean()

    assert mislabeled_stories_trained(mixed) > 0.9
    assert mislabeled_stories_trained(single) < 0.5


# ------------------------------------------------------- bfloat16, as the real runs


def test_bfloat16_run_checkpoints_resumes_and_re_evaluates_exactly(workspace):
    """The real runs keep parameters, gradients and optimizer state in bfloat16. Everything that
    was tested above in float32 must also hold there: checkpoints round-trip, an interrupted run
    resumes to the identical result, and the saved model reproduces the trained one."""
    from gram.run import evaluate_run, run

    settings = dict(optim__precision="bf16", elicit__enabled=False)
    straight = config_for(workspace, "bf16_straight", **settings)
    resumed = config_for(workspace, "bf16_resumed", **settings)
    assert run(straight) == "finished"
    assert run(resumed, stop_after_steps=41) == "interrupted"
    assert run(resumed) == "finished"

    first = load_checkpoint(final_checkpoint_path(straight.run_dir))
    assert all(value.dtype == torch.bfloat16 for value in first["model"].values())
    for group in first["optimizers"].values():
        assert all(state["exp_avg"].dtype == torch.bfloat16 for state in group["state"].values())
    assert_same_checkpoint(first, load_checkpoint(final_checkpoint_path(resumed.run_dir)))

    model, _, _ = model_from_checkpoint(final_checkpoint_path(straight.run_dir))
    assert next(model.parameters()).dtype == torch.bfloat16
    evaluate_run(straight, check=True)  # raises if the checkpoint does not reproduce the stored losses
    assert json.loads((straight.run_dir / "eval" / "check.json").read_text())["largest_difference"] == 0.0

    metrics = pd.read_parquet(straight.run_dir / "metrics" / "train.parquet")
    assert metrics.loss.iloc[-20:].mean() < metrics.loss.iloc[:20].mean() - 0.5, "the bfloat16 model learns"
    probes = pd.read_parquet(straight.run_dir / "probes" / "gradient_probes.parquet")
    assert np.isfinite(probes.relative_grad_norm_a).all() and (probes.relative_grad_norm_a > 0).all()
