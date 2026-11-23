"""The guarantees gradient routing has to give, tested on the real training step.

These are requirements (a), (b) and (c) of the project brief, plus the property that
makes heterogeneous gradient accumulation correct (paper Appendix H).

Every test first *primes* the optimizers: a few steps in which all groups are updated, so
that every group has non-zero Adam momentum and has been subject to weight decay. Without
that the tests would be too easy. A group whose optimizer has never run would not move
even if it were stepped by mistake with a zero gradient; a primed one would, through its
leftover momentum and through weight decay. "Unchanged" below always means bit-for-bit
identical parameters *and* an identical optimizer state.
"""

import copy

import pytest
import torch
from helpers import AUX, CONTEXT, EOS, TINY_MODEL, VOCAB

from gram.config import CORE, OptimConfig
from gram.model import build_model, next_token_loss
from gram.routing import Rule
from gram.train import accumulate_gradients, make_optimizers, training_step

ALL = (CORE, *AUX)
OPTIM = OptimConfig(lr=1e-2, weight_decay=0.1, precision="fp32")  # a large weight decay, so a stray step would show


def batch(seed: int, size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(2, VOCAB, (size, CONTEXT + 1), generator=generator)
    tokens[:, 9] = EOS
    return tokens


def rule(name: str, module: str | None = None) -> Rule:
    return {
        "core_plain": lambda: Rule("core_plain", CORE, (CORE,), (CORE,)),
        "core_robust": lambda: Rule("core_robust", CORE, (CORE, module), (CORE, module)),
        "aux_isolated": lambda: Rule("aux_isolated", module, (CORE, module), (module,)),
        "aux_spread": lambda: Rule("aux_spread", module, (CORE, module), (CORE, module)),
        "unlabeled": lambda: Rule("unlabeled", "unlabeled", ALL, ALL),
    }[name]()


@pytest.fixture
def primed():
    """A GRAM model and its optimizers after a few steps that updated every group."""
    torch.manual_seed(0)
    model = build_model("gram", TINY_MODEL, AUX)
    optimizers = make_optimizers(model, OPTIM)
    for seed in range(5):
        training_step(model, optimizers, [(batch(100 + seed), rule("unlabeled"))], lr=OPTIM.lr, grad_clip=1.0)
    for name, optimizer in optimizers.items():
        momentum = [state["exp_avg"].abs().sum() for state in optimizer.state.values()]
        assert len(momentum) > 0 and all(value > 0 for value in momentum), f"{name} should have momentum"
    return model, optimizers


def snapshot(model, optimizers):
    """Deep copies of every group's parameters and of every optimizer's state."""
    return {
        name: {
            "parameters": [parameter.detach().clone() for parameter in model.parameter_groups()[name]],
            "optimizer": copy.deepcopy(optimizers[name].state_dict()),
        }
        for name in ALL
    }


def is_unchanged(before, model, optimizers, name: str) -> bool:
    same_parameters = all(
        torch.equal(old, new) for old, new in zip(before[name]["parameters"], model.parameter_groups()[name])
    )
    old_state, new_state = before[name]["optimizer"]["state"], optimizers[name].state_dict()["state"]
    same_optimizer = old_state.keys() == new_state.keys() and all(
        torch.equal(torch.as_tensor(old_state[key][field]), torch.as_tensor(new_state[key][field]))
        for key in old_state
        for field in old_state[key]
    )
    return same_parameters and same_optimizer


def changed_groups(before, model, optimizers) -> set[str]:
    return {name for name in ALL if not is_unchanged(before, model, optimizers, name)}


# ----------------------------------------------------------------------------
# (a) An auxiliary batch never changes other modules' parameters.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("module", AUX)
@pytest.mark.parametrize("kind", ["aux_isolated", "aux_spread"])
def test_an_auxiliary_batch_never_changes_other_modules(primed, module, kind):
    model, optimizers = primed
    before = snapshot(model, optimizers)
    training_step(model, optimizers, [(batch(1), rule(kind, module))], lr=OPTIM.lr, grad_clip=1.0)

    for other in AUX:
        if other != module:
            assert is_unchanged(before, model, optimizers, other), f"module {other} moved on a {module} batch"
    assert not is_unchanged(before, model, optimizers, module), "the batch's own module is always updated"


# ----------------------------------------------------------------------------
# (b) A core batch with no module sampled changes no auxiliary parameters.
# ----------------------------------------------------------------------------


def test_a_core_batch_without_a_module_changes_no_auxiliary_parameters(primed):
    model, optimizers = primed
    before = snapshot(model, optimizers)
    training_step(model, optimizers, [(batch(2), rule("core_plain"))], lr=OPTIM.lr, grad_clip=1.0)
    assert changed_groups(before, model, optimizers) == {CORE}


# ----------------------------------------------------------------------------
# (c) The core group is untouched on auxiliary batches where the p_as draw fails.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("module", AUX)
def test_the_core_is_untouched_when_auxiliary_spread_does_not_fire(primed, module):
    model, optimizers = primed
    before = snapshot(model, optimizers)
    training_step(model, optimizers, [(batch(3), rule("aux_isolated", module))], lr=OPTIM.lr, grad_clip=1.0)
    assert is_unchanged(before, model, optimizers, CORE)
    assert changed_groups(before, model, optimizers) == {module}


# ----------------------------------------------------------------------------
# The remaining rules, for completeness.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, module, expected",
    [
        ("aux_spread", AUX[1], {CORE, AUX[1]}),  # the p_as draw succeeds: core is updated too
        ("core_robust", AUX[2], {CORE, AUX[2]}),  # core batch with one module sampled: both updated
        ("unlabeled", None, set(ALL)),  # no label: everything takes part
    ],
)
def test_each_rule_changes_exactly_the_groups_it_updates(primed, kind, module, expected):
    model, optimizers = primed
    before = snapshot(model, optimizers)
    stats = training_step(model, optimizers, [(batch(4), rule(kind, module))], lr=OPTIM.lr, grad_clip=1.0)
    assert changed_groups(before, model, optimizers) == expected == set(stats["updated"])


def test_a_group_that_is_not_routed_keeps_its_adam_step_count(primed):
    model, optimizers = primed
    steps = lambda name: {int(state["step"]) for state in optimizers[name].state.values()}  # noqa: E731
    assert all(steps(name) == {5} for name in ALL)
    training_step(model, optimizers, [(batch(5), rule("aux_isolated", AUX[0]))], lr=OPTIM.lr, grad_clip=1.0)
    assert steps(AUX[0]) == {6} and steps(CORE) == {5} and steps(AUX[1]) == {5}


def test_a_single_optimizer_over_everything_would_break_isolation(primed):
    """Why there is one optimizer per group. With a single optimizer, a group that received no
    gradient is still moved by weight decay and by the momentum left from earlier steps."""
    model, _ = primed
    single = torch.optim.AdamW(model.parameters(), lr=OPTIM.lr, weight_decay=OPTIM.weight_decay)
    for seed in range(3):  # give the single optimizer momentum for every parameter
        next_token_loss(model(batch(seed)[:, :-1], active=ALL), batch(seed)[:, 1:]).backward()
        single.step()
        single.zero_grad()

    untouched = [parameter.detach().clone() for parameter in model.parameter_groups()[AUX[3]]]
    tokens = batch(6)
    next_token_loss(model(tokens[:, :-1], active=(CORE, AUX[0]), update=(AUX[0],)), tokens[:, 1:]).backward()
    assert all(parameter.grad is None for parameter in model.parameter_groups()[AUX[3]])
    # AdamW skips parameters without a gradient, so give them the zero gradient a mask-based
    # implementation would produce.
    for parameter in model.parameter_groups()[AUX[3]]:
        parameter.grad = torch.zeros_like(parameter)
    single.step()
    moved = any(not torch.equal(old, new) for old, new in zip(untouched, model.parameter_groups()[AUX[3]]))
    assert moved, "expected the shared optimizer to move a module that received a zero gradient"


def test_routing_violation_is_detected(primed):
    """If a group ends up with a gradient that no micro-batch routed to it, the step refuses to run."""
    model, optimizers = primed
    for parameter in model.parameter_groups()[AUX[3]]:
        parameter.grad = torch.ones_like(parameter)  # a stray gradient, as a routing bug would leave behind
    with pytest.raises(RuntimeError, match="routing violated"):
        training_step(model, optimizers, [(batch(7), rule("core_plain"))], lr=OPTIM.lr, grad_clip=1.0)


# ----------------------------------------------------------------------------
# Heterogeneous accumulation (Appendix H): several micro-batches with different
# routing rules inside one optimizer step.
# ----------------------------------------------------------------------------


def full_gradient(model, tokens, active):
    """Reference: the ordinary gradient of one micro-batch's loss with nothing frozen, by group."""
    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    next_token_loss(reference(tokens[:, :-1], active=active), tokens[:, 1:]).backward()
    return {
        name: [parameter.grad.clone() for parameter in parameters]
        for name, parameters in reference.parameter_groups().items()
        if name in active
    }


def test_each_group_accumulates_only_the_micro_batches_routed_to_it(primed):
    model, _ = primed
    first, second = AUX[0], AUX[2]
    micro_batches = [
        (batch(10), rule("aux_isolated", first)),  # updates module `first` only
        (batch(11), rule("core_plain")),  # updates core only
        (batch(12), rule("aux_spread", second)),  # updates core and module `second`
    ]
    accumulate_gradients(model, micro_batches)
    accumulated = {
        name: [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
        for name, parameters in model.parameter_groups().items()
    }
    model.zero_grad(set_to_none=True)

    from_first = full_gradient(model, batch(10), (CORE, first))
    from_core = full_gradient(model, batch(11), (CORE,))
    from_second = full_gradient(model, batch(12), (CORE, second))
    count = len(micro_batches)

    def assert_close(actual, expected):
        for a, e in zip(actual, expected):
            assert torch.allclose(a, e, atol=1e-7, rtol=1e-4)

    # Each module: only its own micro-batch.
    assert_close(accumulated[first], [g / count for g in from_first[first]])
    assert_close(accumulated[second], [g / count for g in from_second[second]])
    # The core: the core batch and the spread batch, but NOT the isolated auxiliary batch.
    assert_close(accumulated[CORE], [(a + b) / count for a, b in zip(from_core[CORE], from_second[CORE])])
    # Modules that were not active at all: nothing.
    for other in (AUX[1], AUX[3]):
        assert all(gradient is None for gradient in accumulated[other])

    # The test is sensitive to the leak it guards against: had the isolated batch not frozen the
    # core, the core's accumulated gradient would be measurably different.
    leaked = [(a + b + c) / count for a, b, c in zip(from_core[CORE], from_second[CORE], from_first[CORE])]
    assert not all(torch.allclose(a, l, atol=1e-7, rtol=1e-4) for a, l in zip(accumulated[CORE], leaked))


def test_a_mixed_step_updates_the_union_and_nothing_else(primed):
    model, optimizers = primed
    before = snapshot(model, optimizers)
    micro_batches = [
        (batch(20), rule("aux_isolated", AUX[0])),
        (batch(21), rule("aux_isolated", AUX[1])),
        (batch(22), rule("core_robust", AUX[1])),
    ]
    stats = training_step(model, optimizers, micro_batches, lr=OPTIM.lr, grad_clip=1.0)
    assert changed_groups(before, model, optimizers) == {CORE, AUX[0], AUX[1]} == set(stats["updated"])
    assert len(stats["losses"]) == 3


def test_gradients_are_clipped_jointly_over_the_updated_groups(primed):
    model, optimizers = primed
    micro_batches = [(batch(30), rule("aux_spread", AUX[0]))]
    accumulate_gradients(model, micro_batches)
    joint = torch.sqrt(
        sum(parameter.grad.pow(2).sum() for name in (CORE, AUX[0]) for parameter in model.parameter_groups()[name])
    )
    model.zero_grad(set_to_none=True)
    stats = training_step(model, optimizers, micro_batches, lr=OPTIM.lr, grad_clip=1.0)
    assert stats["grad_norm"] == pytest.approx(float(joint), rel=1e-5)
