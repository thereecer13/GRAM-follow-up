"""The analysis forward pass must compute exactly what the model computes."""

import torch

from gram import interp
from gram.config import CORE, ModelConfig
from gram.model import build_model

AUX = ("topic_a", "topic_b")


def small_model():
    config = ModelConfig(vocab_size=50, context_length=16, n_layers=3, n_heads=4, n_kv_heads=2, d_model=32,
                         dense_mlp_dim=64, core_mlp_dim=64, aux_mlp_dim=16)
    torch.manual_seed(0)
    return build_model("gram", config, AUX).eval()


def test_run_matches_the_model_for_every_profile():
    model = small_model()
    tokens = torch.randint(2, 50, (4, 16))
    for active in [(CORE,), (CORE, "topic_a"), (CORE, "topic_a", "topic_b")]:
        ours = interp.run(model, tokens, interp.everywhere(model, active)).logits
        assert torch.allclose(ours, model(tokens, active=active), atol=1e-6)


def test_removing_a_module_in_every_block_is_the_profile_without_it():
    model = small_model()
    tokens = torch.randint(2, 50, (4, 16))
    ours = interp.run(model, tokens, interp.without(model, "topic_a", interp.ALL_LAYERS)).logits
    assert torch.allclose(ours, model(tokens, active=(CORE, "topic_b")), atol=1e-6)


def test_removing_a_module_in_one_block_changes_only_that_block():
    model = small_model()
    plan = interp.without(model, "topic_b", 1)
    assert plan[0] == plan[2] == (CORE, "topic_a", "topic_b")
    assert plan[1] == (CORE, "topic_a")


def test_kept_outputs_add_up_to_the_residual_change():
    model = small_model()
    tokens = torch.randint(2, 50, (2, 16))
    forward = interp.run(model, tokens, interp.everywhere(model, model.module_names), keep=True)
    assert len(forward.mlp_outputs) == 3 and set(forward.mlp_outputs[0]) == set(model.module_names)
    assert forward.residual[-1].shape == (2, 16, 32)
