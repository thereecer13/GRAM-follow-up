"""Evaluation, elicitation, and a run from start to finish."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from helpers import AUX, CONTEXT, TINY_MODEL, make_store, tiny_run_config

from gram.checkpoint import final_checkpoint_path, load_checkpoint, model_from_checkpoint
from gram.config import CORE, ElicitProtocol, save_config
from gram.data import TokenStore, build_elicitation_sequences, build_validation_sets
from gram.elicit import elicit_topic, summarise_elicitation
from gram.eval import evaluate_profiles, evaluation_profiles, fingerprint, mean_loss, validation_sweep
from gram.model import build_model
from gram.run import evaluate_run, run
from gram.train import token_tensor


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    path = tmp_path_factory.mktemp("runs")
    make_store(path / "store")
    return path


@pytest.fixture(scope="module")
def validation_sets(workspace):
    store = TokenStore(workspace / "store")
    return {name: token_tensor(tokens, "cpu") for name, tokens in build_validation_sets(store, AUX, CONTEXT).items()}


@pytest.fixture(scope="module")
def gram_model():
    torch.manual_seed(0)
    return build_model("gram", TINY_MODEL, AUX)


def run_config(workspace, name, method="gram", **overrides):
    config = tiny_run_config(workspace, method, **overrides)
    config.name = name
    return config


# ---------------------------------------------------------------- evaluation


def test_mean_loss_is_the_plain_mean_over_all_predictions(gram_model, validation_sets):
    sequences = validation_sets[AUX[0]][:21]
    per_token = []
    with torch.no_grad():
        for sequence in sequences.long():
            logits = gram_model(sequence[None, :-1], active=(CORE, AUX[0]))
            per_token.append(F.cross_entropy(logits[0], sequence[1:], reduction="none"))
    expected = torch.cat(per_token).double().mean().item()
    for batch_size in (1, 4, 21, 64):  # including a last batch that is not full
        assert mean_loss(gram_model, sequences, (CORE, AUX[0]), batch_size) == pytest.approx(expected, rel=1e-6)
    assert np.isnan(mean_loss(gram_model, sequences[:0], (CORE,), 4))


def test_mean_loss_does_not_disturb_the_model(gram_model, validation_sets):
    gram_model.train()
    before = {name: value.clone() for name, value in gram_model.state_dict().items()}
    mean_loss(gram_model, validation_sets[CORE][:8], (CORE,), 4)
    assert gram_model.training
    assert all(torch.equal(before[name], value) for name, value in gram_model.state_dict().items())
    assert all(parameter.grad is None for parameter in gram_model.parameters())


def test_profiles_of_a_gram_model_and_of_a_dense_model(gram_model):
    profiles = evaluation_profiles(gram_model)
    assert profiles["core_only"] == (CORE,)
    assert profiles["all_modules"] == (CORE, *AUX)
    for topic in AUX:
        assert profiles[f"only:{topic}"] == (CORE, topic)
        assert set(profiles[f"without:{topic}"]) == {CORE, *AUX} - {topic}
    assert len(profiles) == 2 + 2 * len(AUX)
    assert evaluation_profiles(build_model("filtering", TINY_MODEL, AUX)) == {"dense": (CORE,)}


def test_every_profile_is_scored_on_every_dataset(gram_model, validation_sets):
    small = {name: tokens[:8] for name, tokens in validation_sets.items()}
    table = evaluate_profiles(gram_model, small, batch_size=8)
    assert len(table) == 10 * 5 and not table.loss.isna().any()
    assert set(table.dataset) == {CORE, *AUX}
    assert fingerprint(gram_model, validation_sets, 8).keys() == {f"{r.profile}|{r.dataset}" for r in table.itertuples()}


def test_validation_sweep_scores_all_auxiliary_data_and_a_fresh_sample_of_core(validation_sets):
    dense = build_model("baseline", TINY_MODEL, AUX)
    sweep = lambda step: pd.DataFrame(validation_sweep(dense, validation_sets, step, seed=1, core_sequences=12, batch_size=16))  # noqa: E731
    first, again, later = sweep(10), sweep(10), sweep(20)
    sizes = dict(zip(first.dataset, first.num_sequences))
    assert sizes[CORE] == 12 and all(sizes[topic] == len(validation_sets[topic]) for topic in AUX)
    pd.testing.assert_frame_equal(first, again)
    core = lambda frame: frame.loc[frame.dataset == CORE, "loss"].item()  # noqa: E731
    assert core(first) != core(later), "each sweep draws a new sample of the core validation set"
    aux = lambda frame: frame.loc[frame.dataset == AUX[0], "loss"].item()  # noqa: E731
    assert aux(first) == aux(later), "the auxiliary sets are always scored in full"


# --------------------------------------------------------------- elicitation


@pytest.fixture(scope="module")
def elicitation_data(workspace, validation_sets):
    store = TokenStore(workspace / "store")
    topic = AUX[0]
    train = token_tensor(build_elicitation_sequences(store, topic, 16, CONTEXT, seed=1), "cpu")
    return train, validation_sets[topic]


def test_elicitation_recovers_a_topic_the_model_never_saw(gram_model, elicitation_data):
    train, validation = elicitation_data
    protocol = ElicitProtocol(name="t", num_sequences=16, max_steps=12, patience=0)
    before = {name: value.clone() for name, value in gram_model.state_dict().items()}
    rows = pd.DataFrame(elicit_topic(gram_model, train, validation, (CORE,), 5e-3, protocol, 4, 16))

    assert rows.step.tolist() == list(range(13)), "step 0 is the untouched model, then one row per step"
    assert rows.val_loss[0] == pytest.approx(mean_loss(gram_model, validation, (CORE,), 16))
    assert rows.val_loss.iloc[-1] < rows.val_loss[0] - 0.3, "finetuning on the topic lowers its loss"
    assert all(torch.equal(before[name], value) for name, value in gram_model.state_dict().items()), "the model itself is not modified"


def test_elicitation_stops_early_when_patience_runs_out(gram_model, elicitation_data):
    train, validation = elicitation_data
    diverging = dict(lr=5.0, micro_batch_size=4, eval_batch_size=16)  # a rate so high that the loss only gets worse
    patient = ElicitProtocol(name="t", num_sequences=16, max_steps=30, patience=3)
    rows = pd.DataFrame(elicit_topic(gram_model, train, validation, (CORE,), protocol=patient, **diverging))
    assert rows.step.max() < 30, "stopped before max_steps"
    best = rows[rows.step >= 1].val_loss.idxmin()
    assert rows.step.max() - rows.step[best] == 4, "stops once more than `patience` steps pass without a new best"

    fixed_length = ElicitProtocol(name="t", num_sequences=16, max_steps=9, patience=0)
    assert len(elicit_topic(gram_model, train, validation, (CORE,), protocol=fixed_length, **diverging)) == 10


def test_an_ablated_module_plays_no_part_in_elicitation(gram_model, elicitation_data):
    train, validation = elicitation_data
    protocol = ElicitProtocol(name="t", num_sequences=16, max_steps=4, patience=0)
    reference = elicit_topic(gram_model, train, validation, (CORE,), 5e-3, protocol, 4, 16)
    scrambled = build_model("gram", TINY_MODEL, AUX)
    scrambled.load_state_dict(gram_model.state_dict())
    with torch.no_grad():
        for topic in AUX:
            for parameter in scrambled.parameter_groups()[topic]:
                parameter.normal_()
    with_scrambled_modules = elicit_topic(scrambled, train, validation, (CORE,), 5e-3, protocol, 4, 16)
    pd.testing.assert_frame_equal(pd.DataFrame(with_scrambled_modules), pd.DataFrame(reference))


def test_elicited_loss_is_the_best_finetuned_checkpoint():
    curve = pd.DataFrame(
        {
            "protocol": "code",
            "topic": ["a"] * 5 + ["b"] * 3,
            "step": [0, 1, 2, 3, 4, 0, 1, 2],
            "val_loss": [2.0, 1.8, 1.5, 1.6, 1.7, 1.0, 1.2, 1.3],
        }
    )
    summary = summarise_elicitation(curve).set_index("topic")
    assert summary.loc["a", ["loss_before", "loss_elicited", "best_step", "loss_at_last_step", "steps_run"]].tolist() == [2.0, 1.5, 2, 1.7, 4]
    # As in the official code, the untouched model (step 0) is not a candidate, even when
    # finetuning only made things worse.
    assert summary.loc["b", "loss_elicited"] == 1.2 and summary.loc["b", "best_step"] == 1


# ------------------------------------------------------- a whole run, three ways


@pytest.fixture(scope="module")
def finished_runs(workspace):
    """One finished run per method, trained on the same data with the same seed."""
    configs = {
        "baseline": run_config(workspace, "toy_baseline", "baseline", eval__curve_points=6),
        "filtering": run_config(workspace, "toy_filtering", "filtering"),
        "gram": run_config(workspace, "toy_gram", "gram"),
    }
    for config in configs.values():
        assert run(config) == "finished"
    return configs


def losses_of(config) -> pd.DataFrame:
    return pd.read_csv(config.run_dir / "eval" / "losses.csv").pivot(index="profile", columns="dataset", values="loss")


def test_a_finished_run_has_all_its_results(finished_runs):
    for method, config in finished_runs.items():
        run_dir = config.run_dir
        assert (run_dir / "DONE").exists()
        losses = pd.read_csv(run_dir / "eval" / "losses.csv")
        assert len(losses) == (50 if method == "gram" else 5) and not losses.loss.isna().any()
        summary = pd.read_csv(run_dir / "eval" / "elicitation.csv")
        assert sorted(zip(summary.protocol, summary.topic)) == sorted((p, t) for p in ("code", "paper") for t in AUX)
        curves = pd.read_csv(run_dir / "eval" / "elicitation_curves.csv")
        assert curves[curves.protocol == "paper"].groupby("topic").step.max().eq(3).all(), "no early stopping: all 3 steps"
        assert curves[curves.protocol == "code"].groupby("topic").step.max().le(6).all()
        # Elicitation starts from the forget loss that evaluation reported.
        core_profile = "core_only" if method == "gram" else "dense"
        for row in summary.itertuples():
            reported = losses[(losses.profile == core_profile) & (losses.dataset == row.topic)].loss.item()
            assert row.loss_before == pytest.approx(reported, abs=1e-9)
    assert (finished_runs["baseline"].run_dir / "metrics" / "validation_curve.parquet").exists()


def test_gradient_routing_localises_each_topic_in_its_own_module(finished_runs):
    """The method works on the toy problem: a topic's knowledge sits in its module and nowhere else."""
    losses = losses_of(finished_runs["gram"])
    for topic in AUX:
        ablated = losses.loc["core_only", topic]
        own_module = losses.loc[f"only:{topic}", topic]
        assert ablated - own_module > 1.0, f"switching on {topic}'s module should restore {topic}"
        for other in AUX:
            if other != topic:
                assert losses.loc[f"only:{other}", topic] - own_module > 0.8, "another topic's module does not help"
                assert losses.loc[f"without:{other}", topic] < losses.loc[f"without:{topic}", topic] - 0.5
        # Only removing a topic's own module costs it much.
        assert losses.loc[f"without:{topic}", topic] - losses.loc["all_modules", topic] > 0.5
    assert losses.loc["core_only", CORE] < 1.6, "core performance survives ablating every module"


def test_filtering_never_learns_the_removed_topics(finished_runs):
    baseline, filtering = losses_of(finished_runs["baseline"]), losses_of(finished_runs["filtering"])
    for topic in AUX:
        assert filtering.loc["dense", topic] - baseline.loc["dense", topic] > 1.0
    assert abs(filtering.loc["dense", CORE] - baseline.loc["dense", CORE]) < 0.4


def test_elicitation_brings_back_some_of_what_was_removed(finished_runs):
    for config in (finished_runs["gram"], finished_runs["filtering"]):
        summary = pd.read_csv(config.run_dir / "eval" / "elicitation.csv")
        assert (summary.loss_elicited < summary.loss_before).all()


def test_running_a_finished_run_again_does_nothing(finished_runs):
    config = finished_runs["gram"]
    checkpoint = final_checkpoint_path(config.run_dir)
    modified = checkpoint.stat().st_mtime_ns
    assert run(config) == "finished"
    assert checkpoint.stat().st_mtime_ns == modified


def test_an_interrupted_run_can_be_continued(workspace):
    config = run_config(workspace, "toy_interrupted")
    assert run(config, stop_after_steps=30) == "interrupted"
    assert not (config.run_dir / "DONE").exists() and not (config.run_dir / "eval").exists()
    assert run(config) == "finished"
    assert (config.run_dir / "DONE").exists()


# ----------------------------------------- numbers come from the saved checkpoint


def test_re_evaluating_the_saved_checkpoint_reproduces_the_stored_losses(finished_runs):
    for config in finished_runs.values():
        evaluate_run(config, check=True)
        result = json.loads((config.run_dir / "eval" / "check.json").read_text())
        assert result["ok"] and result["largest_difference"] <= 1e-9


def test_the_checkpoint_reproduces_the_model_that_training_ended_with(finished_runs):
    config = finished_runs["gram"]
    recorded = json.loads((config.run_dir / "train_summary.json").read_text())["fingerprint"]
    store = TokenStore(config.data.store_dir)
    sets = {name: token_tensor(tokens, "cpu") for name, tokens in build_validation_sets(store, AUX, CONTEXT).items()}
    model, _, _ = model_from_checkpoint(final_checkpoint_path(config.run_dir))
    recomputed = fingerprint(model, sets, config.eval.batch_size)
    assert recorded.keys() == recomputed.keys() and len(recorded) == 50
    assert max(abs(recorded[key] - recomputed[key]) for key in recorded) < 1e-9


def test_a_checkpoint_that_differs_from_the_trained_model_is_rejected(workspace):
    config = run_config(workspace, "toy_tampered", "filtering", elicit__enabled=False)
    assert run(config) == "finished"
    path = final_checkpoint_path(config.run_dir)
    checkpoint = load_checkpoint(path)
    # As if a tensor had been saved or restored wrongly. (Noise, not a constant shift: adding the
    # same number to every output bias would leave the predicted probabilities unchanged.)
    noise = torch.randn(checkpoint["model"]["unembed.weight"].shape, generator=torch.Generator().manual_seed(0))
    checkpoint["model"]["unembed.weight"] += 0.05 * noise
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="does not reproduce the trained model"):
        evaluate_run(config, check=True)


def test_stored_losses_that_differ_from_the_checkpoint_are_rejected(workspace):
    config = run_config(workspace, "toy_edited", "filtering", elicit__enabled=False)
    assert run(config) == "finished"
    path = config.run_dir / "eval" / "losses.csv"
    losses = pd.read_csv(path)
    losses.loc[0, "loss"] += 0.001
    losses.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="does not match the stored losses"):
        evaluate_run(config, check=True)
    assert not json.loads((config.run_dir / "eval" / "check.json").read_text())["ok"]


def test_command_line(workspace):
    config = run_config(workspace, "toy_cli", "filtering", elicit__enabled=False)
    path = workspace / "toy_cli.yaml"
    save_config(config, path)
    finished = subprocess.run([sys.executable, "-m", "gram.run", str(path)], capture_output=True, text=True)
    assert finished.returncode == 0 and "toy_cli: finished" in finished.stdout, finished.stderr[-2000:]
    checked = subprocess.run([sys.executable, "-m", "gram.run", str(path), "--check"], capture_output=True, text=True)
    assert checked.returncode == 0 and "ok" in checked.stdout, checked.stderr[-2000:]
