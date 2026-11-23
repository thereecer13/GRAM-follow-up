"""Check our model against the official GRAM implementation, numerically.

Our implementation is written from scratch, using the official repository
(github.com/agencyenterprise/modular-pretraining) only as a specification. This script
is the evidence that the two compute the same function: it copies one set of weights
into both models and compares logits and gradients, for the dense model and for the
GRAM model under every kind of routing decision.

Nothing from the official repository is vendored here (it carries no licence). You
point the script at your own clone:

    git clone https://github.com/agencyenterprise/modular-pretraining /path/to/clone
    python scripts/crosscheck_official.py --official-repo /path/to/clone

It imports exactly two files from the clone, ``src/model/base.py`` and
``src/model/moe.py`` (pure PyTorch model code), and refuses to run if they differ from
the versions that were read and reviewed for this project (commit d07d62e9).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from gram.config import AUX_TOPICS, ModelConfig
from gram.model import CORE, build_model, next_token_loss

REVIEWED_COMMIT = "d07d62e9869c6a969a2306e755aba7a742c317fd"
REVIEWED_SHA256 = {
    "src/model/base.py": "07ab2f6b536098ac72b3928f7f9ddbdd5a87cb77b76ede1dab78359677570119",
    "src/model/moe.py": "fa2a3aaf61b7d4ca8eb374531082150e1309fb5c8775b96d0cac8ca96e27dad7",
}

# The official model files import their configuration classes from src/model/config.py,
# which pulls in the tokenizer library and the training utilities. The models only need
# the plain fields below, so we give them this stand-in instead.
CONFIG_STUB = '''
from dataclasses import dataclass
from torch import nn

@dataclass
class ModelConfig:
    arch: str = "base"
    ctx_len: int = 1024
    vocab_size: int = 50304
    num_layers: int = 8
    num_heads: int = 8
    num_key_value: int = 2
    attn_bias: bool = True
    eos_token_id: int = -1
    embed_dim: int = 512
    mlp_dim: int = 2048

@dataclass
class RoutedModelConfig(ModelConfig):
    arch: str = "moe"
    core_param_prc: float = 0.95
    aux_param_prc: float = 0.05

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
'''


def load_official_modules(repo: Path):
    """Import the two official model files from a throwaway package directory."""
    for relative, expected in REVIEWED_SHA256.items():
        actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"{relative} in {repo} differs from the version reviewed for this project "
                f"(commit {REVIEWED_COMMIT[:8]}). Read the new version before running it."
            )

    staging = Path(tempfile.mkdtemp(prefix="gram_official_"))
    package = staging / "src" / "model"
    package.mkdir(parents=True)
    (staging / "src" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text(CONFIG_STUB)
    shutil.copy(repo / "src/model/base.py", package / "base.py")
    shutil.copy(repo / "src/model/moe.py", package / "moe.py")

    sys.path.insert(0, str(staging))
    config = importlib.import_module("src.model.config")
    base = importlib.import_module("src.model.base")
    moe = importlib.import_module("src.model.moe")
    return config, base, moe


def official_to_ours_name(official_name: str, module_names: tuple[str, ...]) -> str:
    """Translate an official state-dict key into ours."""
    name = official_name
    name = name.replace(".attn.c_attn_q.", ".attention.query.")
    name = name.replace(".attn.c_attn_kv.", ".attention.key_value.")
    name = name.replace(".attn.c_proj.", ".attention.output.")
    name = name.replace(".norm_1.", ".norm_attention.")
    name = name.replace(".norm_2.", ".norm_mlp.")
    if name.startswith("norm."):
        name = "norm_out." + name[len("norm."):]
    # Dense model: blocks.L.mlp.*   GRAM model: blocks.L.moe.experts.K.*
    name = name.replace(".mlp.c_fc.", f".mlps.{CORE}.fc_in.").replace(".mlp.c_proj.", f".mlps.{CORE}.fc_out.")
    if ".moe.experts." in name:
        prefix, rest = name.split(".moe.experts.")
        index, rest = rest.split(".", 1)
        rest = rest.replace("c_fc.", "fc_in.").replace("c_proj.", "fc_out.")
        name = f"{prefix}.mlps.{module_names[int(index)]}.{rest}"
    return name


def copy_weights(official_model, our_model) -> dict[str, str]:
    """Copy the official model's weights into ours. Returns {official name: our name}."""
    ours = dict(our_model.named_parameters())
    mapping = {}
    with torch.no_grad():
        for official_name, parameter in official_model.named_parameters():
            our_name = official_to_ours_name(official_name, our_model.module_names)
            assert our_name in ours, f"no counterpart for {official_name} (tried {our_name})"
            assert ours[our_name].shape == parameter.shape, f"shape mismatch for {official_name}"
            ours[our_name].copy_(parameter)
            mapping[official_name] = our_name
    assert len(mapping) == len(ours), "some of our parameters have no official counterpart"
    return mapping


def random_tokens(config: ModelConfig, batch: int, generator: torch.Generator) -> torch.Tensor:
    """Random token ids with end-of-story tokens sprinkled in, to exercise the story mask."""
    tokens = torch.randint(2, config.vocab_size, (batch, config.context_length + 1), generator=generator)
    is_eos = torch.rand(tokens.shape, generator=generator) < 0.02
    return torch.where(is_eos, torch.full_like(tokens, config.eos_token_id), tokens)


def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    official_config, official_base, official_moe = load_official_modules(args.official_repo)
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = ModelConfig()
    failures = 0

    def report(label: str, value: float, tolerance: float) -> None:
        nonlocal failures
        ok = value <= tolerance
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} max |difference| = {value:.2e}")

    # ---------------------------------------------------------------- dense
    print("Dense model (baseline / filtering)")
    reference_config = official_config.ModelConfig(
        ctx_len=config.context_length, vocab_size=config.vocab_size, num_layers=config.n_layers,
        num_heads=config.n_heads, num_key_value=config.n_kv_heads, attn_bias=True,
        eos_token_id=config.eos_token_id, embed_dim=config.d_model, mlp_dim=config.dense_mlp_dim,
    )
    reference = official_base.BaseTransformer(reference_config)
    ours = build_model("baseline", config, AUX_TOPICS)
    n_reference = sum(p.numel() for p in reference.parameters())
    n_ours = sum(p.numel() for p in ours.parameters())
    print(f"  parameters: official {n_reference:,} | ours {n_ours:,}")
    failures += n_reference != n_ours
    copy_weights(reference, ours)

    tokens = random_tokens(config, args.batch, generator)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    reference_logits, reference_loss = reference(inputs, targets)
    our_logits = ours(inputs)
    report("logits", max_abs_difference(reference_logits, our_logits), args.tolerance)
    report("loss", abs(float(reference_loss) - float(next_token_loss(our_logits, targets))), 1e-5)

    # ----------------------------------------------------------------- GRAM
    for core_dim in (2048, 1856):
        print(f"GRAM model, core MLP width {core_dim}")
        gram_config = ModelConfig(core_mlp_dim=core_dim)
        reference_config = official_config.RoutedModelConfig(
            ctx_len=config.context_length, vocab_size=config.vocab_size, num_layers=config.n_layers,
            num_heads=config.n_heads, num_key_value=config.n_kv_heads, attn_bias=True,
            eos_token_id=config.eos_token_id, embed_dim=config.d_model, mlp_dim=config.dense_mlp_dim,
            core_param_prc=core_dim / config.dense_mlp_dim, aux_param_prc=0.1,
        )
        labels = [CORE, *AUX_TOPICS]
        reference = official_moe.MoETransformer(reference_config, labels=labels)
        ours = build_model("gram", gram_config, AUX_TOPICS)
        n_reference = sum(p.numel() for p in reference.parameters())
        n_ours = sum(p.numel() for p in ours.parameters())
        print(f"  parameters: official {n_reference:,} | ours {n_ours:,}")
        failures += n_reference != n_ours
        for label in labels:
            official_count = sum(p.numel() for p in reference.get_params(label))
            failures += official_count != ours.num_parameters()[label]
        mapping = copy_weights(reference, ours)
        our_parameters = dict(ours.named_parameters())

        first, second = AUX_TOPICS[0], AUX_TOPICS[2]
        routing_decisions = {
            "core batch, no module": ((CORE,), (CORE,)),
            "core batch + one module (robust)": ((CORE, first), (CORE, first)),
            "auxiliary batch, core frozen": ((CORE, second), (second,)),
            "auxiliary batch, core updated (spread)": ((CORE, second), (CORE, second)),
            "unlabeled batch (everything active)": (tuple(labels), tuple(labels)),
        }
        for description, (active, update) in routing_decisions.items():
            forward_mask = torch.tensor([label in active for label in labels])
            backward_mask = torch.tensor([label in update for label in labels])

            reference.zero_grad(set_to_none=True)
            ours.zero_grad(set_to_none=True)
            reference_logits, reference_loss = reference(inputs, targets, fwd_mask=forward_mask, bck_mask=backward_mask)
            reference_loss.backward()
            our_logits = ours(inputs, active=active, update=update)
            next_token_loss(our_logits, targets).backward()

            print(f" {description}")
            report("logits", max_abs_difference(reference_logits, our_logits), args.tolerance)

            worst_gradient = 0.0
            same_set = True
            for official_name, reference_parameter in reference.named_parameters():
                our_gradient = our_parameters[mapping[official_name]].grad
                if (reference_parameter.grad is None) != (our_gradient is None):
                    same_set = False
                elif reference_parameter.grad is not None:
                    scale = float(reference_parameter.grad.abs().max()) + 1e-12
                    worst_gradient = max(worst_gradient, max_abs_difference(reference_parameter.grad, our_gradient) / scale)
            failures += not same_set
            print(f"  {'ok  ' if same_set else 'FAIL'} the same parameters receive a gradient in both models")
            report("gradients (relative to each tensor's largest entry)", worst_gradient, 1e-3)

    # ------------------------------------------------------- negative control
    # Differences of exactly zero are only convincing if the comparison is able to see a
    # difference at all. Nudge a single weight of our model and confirm that it shows.
    print("Negative control (one of our weights nudged by 1e-3)")
    with torch.no_grad():
        ours.blocks[3].mlps[CORE].fc_in.weight[0, 0] += 1e-3
    forward_mask = torch.tensor([label == CORE for label in labels])
    reference_logits, _ = reference(inputs, targets, fwd_mask=forward_mask, bck_mask=forward_mask)
    difference = max_abs_difference(reference_logits, ours(inputs, active=(CORE,)))
    detected = difference > 0
    failures += not detected
    print(f"  {'ok  ' if detected else 'FAIL'} the comparison detects the change            max |difference| = {difference:.2e}")

    print()
    if failures:
        raise SystemExit(f"{failures} check(s) FAILED")
    print("All checks passed: our model and the official one compute the same function.")


if __name__ == "__main__":
    main()
