"""The model: sizes, parameter groups, ablation, the story mask, and freezing."""

import pytest
import torch
from helpers import AUX, CONTEXT, EOS, TINY_MODEL, VOCAB

from gram.config import AUX_TOPICS, CORE, ModelConfig
from gram.model import RMS_NORM_EPS, Transformer, build_model, next_token_loss, story_causal_mask


def tokens(batch=3, seed=0, with_eos=True):
    generator = torch.Generator().manual_seed(seed)
    out = torch.randint(2, VOCAB, (batch, CONTEXT + 1), generator=generator)
    if with_eos:
        out[:, 5] = EOS
        out[0, 11] = EOS
    return out


@pytest.fixture
def gram():
    torch.manual_seed(0)
    return build_model("gram", TINY_MODEL, AUX)


# --------------------------------------------------------------------- sizes


def test_parameter_counts_match_the_official_models():
    config = ModelConfig()
    dense = build_model("baseline", config, AUX_TOPICS)
    assert sum(p.numel() for p in dense.parameters()) == 26_257_920  # official num_base_params

    gram = build_model("gram", config, AUX_TOPICS).num_parameters()  # code-exact: core width 2048
    assert gram[CORE] == 26_257_920 and all(gram[topic] == 1_578_496 for topic in AUX_TOPICS)
    assert sum(gram.values()) == 32_571_904  # "Initialized MoETransformer with 32,571,904 total parameters"

    paper = build_model("gram", ModelConfig(core_mlp_dim=1856), AUX_TOPICS).num_parameters()
    # The paper's width of 1856 is the one at which core + one module is the size of the baseline.
    assert paper[CORE] + paper[AUX_TOPICS[0]] == 26_262_016
    assert abs(paper[CORE] + paper[AUX_TOPICS[0]] - 26_257_920) / 26_257_920 < 2e-4


def test_filtering_and_baseline_are_the_same_dense_architecture():
    assert build_model("filtering", TINY_MODEL, AUX).module_names == (CORE,)
    assert build_model("baseline", TINY_MODEL, AUX).aux_names == ()


# ---------------------------------------------------------- parameter groups


def test_groups_partition_the_parameters(gram):
    groups = gram.parameter_groups()
    assert list(groups) == [CORE, *AUX]
    ids = [id(p) for parameters in groups.values() for p in parameters]
    assert len(ids) == len(set(ids)) == len(list(gram.parameters()))


def test_core_group_is_everything_shared_and_each_module_group_is_only_its_mlps(gram):
    names = gram.parameter_group_names()
    for topic in AUX:
        assert all(f".mlps.{topic}." in name for name in names[topic])
        assert len(names[topic]) == 4 * TINY_MODEL.n_layers  # two weights and two biases per block
    core = names[CORE]
    assert not any(f".mlps.{topic}." in name for name in core for topic in AUX)
    for needle in ("embed.weight", "unembed.weight", "unembed.bias", "norm_out.weight",
                   "blocks.0.attention.query.weight", "blocks.1.attention.key_value.bias",
                   "blocks.0.norm_attention.weight", "blocks.1.norm_mlp.weight",
                   f"blocks.0.mlps.{CORE}.fc_in.weight", f"blocks.1.mlps.{CORE}.fc_out.bias"):
        assert needle in core, f"{needle} should be in the core group"


# ------------------------------------------------------------------ ablation


def test_ablating_every_module_equals_a_model_with_only_the_core_mlp(gram):
    """Requirement (d). Build a separate dense model, give it the GRAM model's shared weights,
    and compare with the GRAM model run with every auxiliary module switched off."""
    core_only = Transformer(TINY_MODEL, {CORE: TINY_MODEL.core_mlp_dim})
    shared = {name: value for name, value in gram.state_dict().items()
              if not any(f".mlps.{topic}." in name for topic in AUX)}
    assert set(shared) == set(core_only.state_dict()), "the dense model has exactly the core group's tensors"
    core_only.load_state_dict(shared)

    x = tokens()[:, :-1]
    with torch.no_grad():
        ablated = gram(x, active=(CORE,))
        assert torch.equal(ablated, core_only(x))
        # The auxiliary modules do change the output when they are on, so the equality above is not vacuous.
        assert not torch.allclose(ablated, gram(x, active=(CORE, AUX[0])))
        assert not torch.allclose(gram(x, active=(CORE, AUX[0])), gram(x, active=(CORE, AUX[1])))


def test_ablated_output_ignores_the_auxiliary_weights(gram):
    x = tokens()[:, :-1]
    with torch.no_grad():
        before = gram(x, active=(CORE, AUX[0]))
        for topic in AUX[1:]:
            for parameter in gram.parameter_groups()[topic]:
                parameter.add_(torch.randn_like(parameter))
        assert torch.equal(gram(x, active=(CORE, AUX[0])), before)


def test_block_output_is_the_sum_of_the_active_modules(gram):
    block = gram.blocks[0]
    x = torch.randn(2, CONTEXT, TINY_MODEL.d_model)
    mask = story_causal_mask(tokens(2)[:, :-1], EOS)
    with torch.no_grad():
        core_only = block(x, mask, (CORE,), (CORE,))
        with_two = block(x, mask, (CORE, AUX[0], AUX[2]), (CORE, AUX[0], AUX[2]))
        after_attention = x + block.attention(block.norm_attention(x), mask)
        normed = block.norm_mlp(after_attention)
        expected = core_only + block.mlps[AUX[0]](normed) + block.mlps[AUX[2]](normed)
    assert torch.allclose(with_two, expected, atol=1e-6)


def test_forward_rejects_impossible_routing(gram):
    x = tokens()[:, :-1]
    with pytest.raises(ValueError):
        gram(x, active=(AUX[0],))  # the core module is always active
    with pytest.raises(ValueError):
        gram(x, active=(CORE,), update=(CORE, AUX[0]))  # cannot update a module that did not run
    with pytest.raises(ValueError):
        gram(x, active=(CORE, "no-such-module"))


# ---------------------------------------------------------------- story mask


def test_mask_is_causal_and_confined_to_one_story():
    row = torch.tensor([[7, 8, EOS, 9, 10, EOS, 11]])
    mask = story_causal_mask(row, EOS)[0, 0]
    story = [0, 0, 0, 1, 1, 1, 2]  # the end-of-story token belongs to the story it ends
    for query in range(7):
        for key in range(7):
            assert bool(mask[query, key]) == (key <= query and story[key] == story[query])


def test_a_story_is_unaffected_by_the_story_before_it(gram):
    x = tokens(with_eos=False)[:, :-1]
    x[:, 6] = EOS
    changed = x.clone()
    changed[:, :6] = torch.randint(2, VOCAB, (x.shape[0], 6), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        a, b = gram(x), gram(changed)
    assert torch.equal(a[:, 7:], b[:, 7:]), "positions after the end-of-story token must not see across it"
    assert not torch.allclose(a[:, :6], b[:, :6])


def test_predictions_do_not_depend_on_later_tokens(gram):
    x = tokens(with_eos=False)[:, :-1]
    changed = x.clone()
    changed[:, 10:] = 3
    with torch.no_grad():
        assert torch.equal(gram(x)[:, :10], gram(changed)[:, :10])


# ------------------------------------------------------------------ freezing


def gradients(model, x, y, active, update):
    model.zero_grad(set_to_none=True)
    next_token_loss(model(x, active=active, update=update), y).backward()
    return {name: (None if p.grad is None else p.grad.clone()) for name, p in model.named_parameters()}


def test_freezing_does_not_change_the_output(gram):
    x = tokens()[:, :-1]
    with torch.no_grad():
        normal = gram(x, active=(CORE, AUX[0]))
        assert torch.equal(gram(x, active=(CORE, AUX[0]), update=(AUX[0],)), normal)
        assert torch.equal(gram(x, active=(CORE, AUX[0]), update=()), normal)


def test_only_updated_groups_receive_gradients(gram):
    batch = tokens()
    x, y = batch[:, :-1], batch[:, 1:]
    names = gram.parameter_group_names()
    for active, update in [((CORE,), (CORE,)), ((CORE, AUX[1]), (AUX[1],)), ((CORE, AUX[1]), (CORE, AUX[1])),
                           ((CORE, *AUX), (CORE, *AUX)), ((CORE, *AUX), (AUX[0], AUX[3]))]:
        grads = gradients(gram, x, y, active, update)
        for group, members in names.items():
            for name in members:
                if group in update:
                    assert grads[name] is not None and grads[name].abs().sum() > 0, f"{name} should get a gradient"
                else:
                    assert grads[name] is None, f"{name} must not get a gradient under update={update}"


def test_freezing_one_group_leaves_the_other_groups_gradients_unchanged(gram):
    """A frozen core still passes gradient *through* itself: the module's gradient is identical
    whether or not the core is being updated, including for the module's MLP in the first block."""
    batch = tokens()
    x, y = batch[:, :-1], batch[:, 1:]
    both = gradients(gram, x, y, (CORE, AUX[2]), (CORE, AUX[2]))
    module_only = gradients(gram, x, y, (CORE, AUX[2]), (AUX[2],))
    module_names = gram.parameter_group_names()[AUX[2]]
    for name in module_names:
        assert torch.equal(both[name], module_only[name])
    first_block = [name for name in module_names if name.startswith("blocks.0.")]
    assert first_block and all(module_only[name].abs().sum() > 0 for name in first_block)


# ----------------------------------------------------------------- precision


def test_rms_norm_epsilon_is_the_modern_pytorch_default_for_a_bfloat16_model():
    assert RMS_NORM_EPS == torch.finfo(torch.float32).eps
    x = torch.full((1, 8), 1e-4)  # so small that the epsilon dominates the result
    ours = torch.nn.RMSNorm(8, eps=RMS_NORM_EPS).to(torch.bfloat16)(x.bfloat16())
    default = torch.nn.RMSNorm(8).to(torch.bfloat16)(x.bfloat16())
    assert torch.equal(ours, default), "PyTorch's default epsilon for bfloat16 changed; see DEVIATIONS.md 2.3"


def test_bfloat16_model_runs_and_the_loss_is_computed_in_float32(gram):
    gram = gram.to(torch.bfloat16)
    batch = tokens()
    logits = gram(batch[:, :-1], active=(CORE, AUX[0]))
    loss = next_token_loss(logits, batch[:, 1:])
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
    assert torch.isfinite(loss) and 3.0 < loss.item() < 5.5  # about log(64) at initialisation
    loss.backward()
    assert all(p.grad.dtype == torch.bfloat16 for p in gram.parameter_groups()[AUX[0]])
