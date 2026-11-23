"""Gradient probes: they measure the right thing, and they leave everything exactly as it was."""

import copy
import random

import numpy as np
import pytest
import torch
from helpers import AUX, CONTEXT, TINY_MODEL

from gram.config import CORE, OptimConfig
from gram.data import build_probe_batches
from gram.model import build_model, next_token_loss
from gram.probes import cosine, group_gradients, norm, probe_steps, run_gradient_probes
from gram.routing import Rule
from gram.train import make_optimizers, token_tensor, training_step

ALL = (CORE, *AUX)


@pytest.fixture(scope="module")
def probe_batches(store):
    batches = build_probe_batches(store, AUX, CONTEXT, batch_size=4)
    return {name: (token_tensor(a, "cpu"), token_tensor(b, "cpu")) for name, (a, b) in batches.items()}


def trained_model(method="gram", dtype=torch.float32, steps=6):
    """A model that has taken a few steps, so that its optimizers hold state worth protecting."""
    torch.manual_seed(0)
    model = build_model(method, TINY_MODEL, AUX).to(dtype)
    optimizers = make_optimizers(model, OptimConfig(lr=1e-2, precision="fp32"))
    everything = model.module_names
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        tokens = torch.randint(2, TINY_MODEL.vocab_size, (4, CONTEXT + 1), generator=generator)
        training_step(model, optimizers, [(tokens, Rule("unlabeled", "unlabeled", everything, everything))], 1e-2, 1.0)
    return model, optimizers


def random_states():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def same_random_states(before, after) -> bool:
    numpy_same = all(np.array_equal(a, b) for a, b in zip(before["numpy"], after["numpy"]))
    return before["python"] == after["python"] and numpy_same and torch.equal(before["torch"], after["torch"])


# ----------------------------------------------------------------------------
# (e) Running the probes leaves parameters, optimizer state and RNG state unchanged.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gram", "baseline"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_probes_leave_parameters_optimizers_and_random_state_unchanged(probe_batches, method, dtype):
    model, optimizers = trained_model(method, dtype)
    model.train()
    parameters_before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizers_before = {name: copy.deepcopy(optimizer.state_dict()) for name, optimizer in optimizers.items()}
    random_before = random_states()

    rows = run_gradient_probes(model, probe_batches, step=6)
    assert len(rows) > 0

    for name, value in model.state_dict().items():
        assert torch.equal(value, parameters_before[name]) and value.dtype == dtype, f"{name} changed"
    for name, optimizer in optimizers.items():
        before, after = optimizers_before[name], optimizer.state_dict()
        assert before["param_groups"] == after["param_groups"]
        for index in before["state"]:
            for field in before["state"][index]:
                assert torch.equal(torch.as_tensor(before["state"][index][field]), torch.as_tensor(after["state"][index][field]))
    assert same_random_states(random_before, random_states()), "a probe advanced a random number generator"
    assert all(parameter.grad is None for parameter in model.parameters()), "a probe left gradients behind"
    assert model.training, "a probe changed the model's training mode"


def test_the_next_training_step_is_identical_with_and_without_a_probe(probe_batches):
    """The strongest form of 'no effect on training': probe, then step, and compare with just stepping."""
    tokens = torch.randint(2, TINY_MODEL.vocab_size, (4, CONTEXT + 1), generator=torch.Generator().manual_seed(7))
    rule = Rule("aux_spread", AUX[0], (CORE, AUX[0]), (CORE, AUX[0]))
    results = []
    for probe in (False, True):
        model, optimizers = trained_model()
        if probe:
            run_gradient_probes(model, probe_batches, step=6)
        training_step(model, optimizers, [(tokens, rule)], 1e-2, 1.0)
        results.append({name: value.clone() for name, value in model.state_dict().items()})
    assert all(torch.equal(results[0][name], results[1][name]) for name in results[0])


# ----------------------------------------------------------------------------
# What the probes record
# ----------------------------------------------------------------------------


def test_one_row_per_topic_and_group(probe_batches):
    model, _ = trained_model()
    rows = run_gradient_probes(model, probe_batches, step=6)
    assert sorted((row["topic"], row["group"]) for row in rows) == sorted((topic, group) for topic in ALL for group in ALL)
    assert all(row["step"] == 6 for row in rows)

    dense, _ = trained_model("baseline")
    dense_rows = run_gradient_probes(dense, probe_batches, step=6)
    assert sorted((row["topic"], row["group"]) for row in dense_rows) == sorted((topic, CORE) for topic in ALL)
    assert all(row["cosine_kind"] == "" and np.isnan(row["cos_unlabeled_vs_labeled"]) for row in dense_rows)


def reference_gradients(model, tokens, active):
    """Gradients computed here, independently of gram.probes, on a float32 copy of the model."""
    clone = copy.deepcopy(model).float()
    tokens = tokens.long()
    next_token_loss(clone(tokens[:, :-1], active=active), tokens[:, 1:]).backward()
    return {name: [p.grad for p in parameters] for name, parameters in clone.parameter_groups().items() if name in active}


def test_relative_gradient_norm_is_the_unlabeled_rule_gradient_over_the_parameter_norm(probe_batches):
    model, _ = trained_model()
    rows = {(row["topic"], row["group"]): row for row in run_gradient_probes(model, probe_batches, step=6)}
    for topic in (CORE, AUX[1]):
        batch_a, batch_b = probe_batches[topic]
        from_a = reference_gradients(model, batch_a, ALL)  # the unlabeled rule: every module active
        from_b = reference_gradients(model, batch_b, ALL)
        for group in ALL:
            row = rows[(topic, group)]
            parameter_norm = norm(model.parameter_groups()[group])
            assert row["param_norm"] == pytest.approx(parameter_norm, rel=1e-6)
            assert row["grad_norm_a"] == pytest.approx(norm(from_a[group]), rel=1e-5)
            assert row["relative_grad_norm_a"] == pytest.approx(norm(from_a[group]) / parameter_norm, rel=1e-5)
            assert row["relative_grad_norm_b"] == pytest.approx(norm(from_b[group]) / parameter_norm, rel=1e-5)
            assert row["cos_unlabeled_a_vs_b"] == pytest.approx(cosine(from_a[group], from_b[group]), abs=1e-5)


def test_cosines_compare_the_unlabeled_rule_with_the_labeled_rule(probe_batches):
    model, _ = trained_model()
    rows = {(row["topic"], row["group"]): row for row in run_gradient_probes(model, probe_batches, step=6)}
    topic, other = AUX[0], AUX[2]
    batch_a, batch_b = probe_batches[topic]
    unlabeled_a = reference_gradients(model, batch_a, ALL)
    labeled_b = reference_gradients(model, batch_b, (CORE, topic))

    # (b) restricted to the topic's own module.
    matched = rows[(topic, topic)]
    assert matched["cosine_kind"] == "matched"
    assert matched["cos_unlabeled_vs_labeled"] == pytest.approx(cosine(unlabeled_a[topic], labeled_b[topic]), abs=1e-5)

    # (c) the same comparison restricted to the core parameters.
    core = rows[(topic, CORE)]
    assert core["cosine_kind"] == "core"
    assert core["cos_unlabeled_vs_labeled"] == pytest.approx(cosine(unlabeled_a[CORE], labeled_b[CORE]), abs=1e-5)

    # (c) the mismatched-module control: topic i's batch, module j's parameters.
    mismatched = rows[(topic, other)]
    assert mismatched["cosine_kind"] == "mismatched"
    as_other = reference_gradients(model, batch_b, (CORE, other))  # topic i's batch B, labeled as module j
    assert mismatched["cos_unlabeled_vs_labeled"] == pytest.approx(cosine(unlabeled_a[other], as_other[other]), abs=1e-5)
    own = reference_gradients(model, probe_batches[other][1], (CORE, other))  # module j's own topic, batch B
    assert mismatched["cos_unlabeled_vs_own_topic_labeled"] == pytest.approx(cosine(unlabeled_a[other], own[other]), abs=1e-5)

    # The core dataset has no module of its own, so it carries no labeled-rule comparison.
    assert all(rows[(CORE, group)]["cosine_kind"] == "" for group in ALL)


def test_probing_a_bfloat16_model_measures_the_same_weights_in_float32(probe_batches):
    model, _ = trained_model(dtype=torch.bfloat16)
    rows = {(row["topic"], row["group"]): row for row in run_gradient_probes(model, probe_batches, step=6)}
    reference = reference_gradients(model, probe_batches[AUX[0]][0], ALL)
    assert all(g.dtype == torch.float32 for gradients in reference.values() for g in gradients)
    for group in ALL:
        assert rows[(AUX[0], group)]["grad_norm_a"] == pytest.approx(norm(reference[group]), rel=1e-5)


def test_cosine_and_norm_helpers():
    a = [torch.tensor([3.0, 0.0]), torch.tensor([[4.0]])]
    assert norm(a) == pytest.approx(5.0)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, [-t for t in a]) == pytest.approx(-1.0)
    assert cosine(a, [torch.tensor([0.0, 1.0]), torch.tensor([[0.0]])]) == pytest.approx(0.0)
    assert np.isnan(cosine(a, [torch.zeros(2), torch.zeros(1, 1)]))


def test_group_gradients_only_returns_active_groups(probe_batches):
    model, _ = trained_model()
    gradients, loss = group_gradients(model, probe_batches[AUX[0]][0], (CORE, AUX[0]))
    assert set(gradients) == {CORE, AUX[0]} and np.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_probe_steps_end_with_the_last_step():
    assert probe_steps(1000, 20) == list(range(50, 1001, 50))
    assert probe_steps(7, 20) == [1, 2, 3, 4, 5, 6, 7], "never more probe points than steps, never step 0"


def test_chunking_a_probe_batch_does_not_change_the_gradient(probe_batches):
    """Chunks only bound memory: the accumulated gradient is that of the whole batch."""
    model, _ = trained_model()
    batch = probe_batches[AUX[0]][0]
    whole, whole_loss = group_gradients(model, batch, ALL)
    for chunk_size in (1, 3, 4):  # 3 leaves a last chunk of a different size
        chunked, chunked_loss = group_gradients(model, batch, ALL, chunk_size)
        assert chunked_loss == pytest.approx(whole_loss, rel=1e-6)
        for group in ALL:
            assert cosine(whole[group], chunked[group]) == pytest.approx(1.0, abs=1e-6)
            assert norm(chunked[group]) == pytest.approx(norm(whole[group]), rel=1e-5)

    rows = run_gradient_probes(model, probe_batches, step=6)
    chunked_rows = run_gradient_probes(model, probe_batches, step=6, chunk_size=3)
    for row, chunked_row in zip(rows, chunked_rows):
        for key, value in row.items():
            if isinstance(value, float) and not np.isnan(value):
                assert chunked_row[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key
