"""The Transformer used for every model in this project.

One class covers both kinds of model:

* The **dense** model (baseline and filtering). Every block has a single MLP, which we
  call "core".
* The **GRAM** model. Every block has the core MLP plus one small auxiliary MLP per
  auxiliary topic. The MLP output of a block is the *sum* of the outputs of the modules
  that are active:

      h_out = sum over active modules m of  MLP_m(h_in)

  There is no router and nothing is decided per token. Which modules are active is
  decided outside the model, once per batch, from the label of the batch.

A dense model is therefore just a GRAM model with no auxiliary modules, and "ablating"
a module means leaving its name out of ``active``.

Parameter groups
----------------
Gradient routing works on *groups* of parameters:

* ``"core"``: everything shared. Token embedding, unembedding, every attention layer,
  every normalisation layer, and the core MLP of every block.
* one group per auxiliary topic: that topic's MLP in every block, and nothing else.

Each forward pass is told which modules are ``active`` and which groups to ``update``.
An active group that is not in ``update`` still runs, but is *frozen* for that pass
(see gram/freeze.py).

Architecture
------------
The paper gives 8 layers, 8 heads, width 512, context 256, vocabulary 4096. The rest is
taken from the official code, and is needed to reproduce its exact parameter count
(26,257,920 for the dense model): pre-norm blocks with RMSNorm, rotary positions,
grouped-query attention with 2 key/value heads, biases on every linear layer, a
GELU(tanh) MLP, separate (untied) embedding and unembedding matrices, and an attention
mask that stops a story attending to the previous story in the same sequence.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from gram.config import CORE, ModelConfig
from gram.freeze import run_module

# RMSNorm adds a small epsilon under the square root. The official model leaves it at
# PyTorch's default, and that default depends on the PyTorch version when the model is
# stored in bfloat16 (as it is here): up to 2.6 it was the bfloat16 machine epsilon
# (2**-7, large enough to change how a freshly initialised model normalises), and from
# 2.7 on it is the float32 machine epsilon. We fix the modern value explicitly so the
# model computes the same function on every PyTorch version, on CPU and GPU, and
# whether its parameters are stored in bfloat16 (training) or float32 (gradient probes
# and CPU tests). See DEVIATIONS.md 2.3.
RMS_NORM_EPS = 1.1920928955078125e-07  # float32 machine epsilon

INIT_STD = 0.02


class RotaryEmbedding(nn.Module):
    """Rotary position encoding, applied to attention queries and keys.

    Positions are not added to the residual stream. Instead each query and key vector is
    rotated by an angle proportional to its position, so the dot product between a query
    and a key depends on how far apart they are.

    The head dimension is split into two halves that are paired up into rotation planes.
    Half of the planes get a frequency (geometrically spaced from 1 down to 1/1024); the
    other half get frequency zero, so those dimensions are never rotated and carry
    position-independent content.
    """

    def __init__(self, head_dim: int, max_positions: int) -> None:
        super().__init__()
        quarter = head_dim // 4
        frequencies = (1 / 1024) ** torch.linspace(0, 1, steps=quarter)
        frequencies = torch.cat([frequencies, torch.zeros(quarter)])
        positions = torch.arange(max_positions, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)  # (position, head_dim / 2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (batch, time, heads, head_dim)."""
        time = x.shape[1]
        cos = self.cos[None, :time, None, :]
        sin = self.sin[None, :time, None, :]
        first, second = x.float().chunk(2, dim=-1)
        rotated = torch.cat([first * cos + second * sin, second * cos - first * sin], dim=-1)
        return rotated.type_as(x)


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention.

    There are ``n_heads`` query heads but only ``n_kv_heads`` key/value heads: each
    key/value head is shared by ``n_heads / n_kv_heads`` query heads.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        self.query = nn.Linear(config.d_model, config.d_model)
        self.key_value = nn.Linear(config.d_model, 2 * self.n_kv_heads * self.head_dim)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.rotary = RotaryEmbedding(self.head_dim, config.context_length)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape

        queries = self.query(x).view(batch, time, self.n_heads, self.head_dim)
        keys, values = self.key_value(x).split(self.n_kv_heads * self.head_dim, dim=-1)
        keys = keys.view(batch, time, self.n_kv_heads, self.head_dim)
        values = values.view(batch, time, self.n_kv_heads, self.head_dim)

        queries = self.rotary(queries)
        keys = self.rotary(keys)

        # scaled_dot_product_attention wants (batch, heads, time, head_dim).
        attended = F.scaled_dot_product_attention(
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=attention_mask,
            enable_gqa=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, time, -1)
        return self.output(attended)


class MLP(nn.Module):
    """A two-layer MLP. The core module and every auxiliary module is one of these."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x), approximate="tanh"))


class Block(nn.Module):
    """One Transformer layer: attention, then the sum of the active MLP modules."""

    def __init__(self, config: ModelConfig, mlp_dims: Mapping[str, int]) -> None:
        super().__init__()
        self.norm_attention = nn.RMSNorm(config.d_model, eps=RMS_NORM_EPS)
        self.attention = Attention(config)
        self.norm_mlp = nn.RMSNorm(config.d_model, eps=RMS_NORM_EPS)
        self.mlps = nn.ModuleDict({name: MLP(config.d_model, dim) for name, dim in mlp_dims.items()})

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        active: Sequence[str],
        update: Sequence[str],
    ) -> torch.Tensor:
        # Attention and both normalisation layers belong to the core group.
        freeze_core = CORE not in update

        normed = run_module(self.norm_attention, x, frozen=freeze_core)
        x = x + run_module(self.attention, normed, attention_mask, frozen=freeze_core)

        normed = run_module(self.norm_mlp, x, frozen=freeze_core)
        mlp_output = None
        for name in active:
            output = run_module(self.mlps[name], normed, frozen=name not in update)
            mlp_output = output if mlp_output is None else mlp_output + output
        return x + mlp_output


class Transformer(nn.Module):
    """Decoder-only Transformer whose blocks hold one MLP per name in ``mlp_dims``.

    ``mlp_dims`` maps module name to MLP hidden width. The first entry must be "core".
    A dense model has only that entry; a GRAM model adds one entry per auxiliary topic.
    """

    def __init__(self, config: ModelConfig, mlp_dims: Mapping[str, int]) -> None:
        super().__init__()
        names = tuple(mlp_dims)
        if not names or names[0] != CORE:
            raise ValueError(f'the first module must be "{CORE}", got {names}')

        self.config = config
        self.module_names: tuple[str, ...] = names
        self.aux_names: tuple[str, ...] = names[1:]

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, mlp_dims) for _ in range(config.n_layers))
        self.norm_out = nn.RMSNorm(config.d_model, eps=RMS_NORM_EPS)
        self.unembed = nn.Linear(config.d_model, config.vocab_size)

        self.apply(_init_weights)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(
        self,
        tokens: torch.Tensor,
        active: Sequence[str] = (CORE,),
        update: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return next-token logits of shape (batch, time, vocab).

        tokens : (batch, time) token ids.
        active : names of the MLP modules that take part in the forward pass. Must
                 include "core", which is always active.
        update : names of the parameter groups that should receive gradients from this
                 pass. Every other active group is frozen. ``None`` means "all active
                 groups", which is the ordinary behaviour and the right choice whenever
                 no gradient is needed (evaluation).
        """
        active = tuple(active)
        update = active if update is None else tuple(update)
        if CORE not in active:
            raise ValueError("the core module is always active")
        if not set(active) <= set(self.module_names):
            raise ValueError(f"unknown module(s) in active={active}; this model has {self.module_names}")
        if not set(update) <= set(active):
            raise ValueError(f"update={update} must be a subset of active={active}")

        # The embedding, the final norm and the unembedding belong to the core group.
        freeze_core = CORE not in update

        attention_mask = story_causal_mask(tokens, self.config.eos_token_id)
        x = run_module(self.embed, tokens, frozen=freeze_core)
        for block in self.blocks:
            x = block(x, attention_mask, active, update)
        x = run_module(self.norm_out, x, frozen=freeze_core)
        return run_module(self.unembed, x, frozen=freeze_core)

    # ------------------------------------------------------------------ #
    # Parameter groups
    # ------------------------------------------------------------------ #

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split the parameters into the groups that gradient routing works on.

        "core" holds every shared parameter (embedding, unembedding, attention, all
        normalisation layers, the core MLPs). Each auxiliary name holds that module's
        MLP in every block. Every parameter is in exactly one group.
        """
        groups: dict[str, list[nn.Parameter]] = {name: [] for name in self.module_names}

        groups[CORE] += list(self.embed.parameters())
        groups[CORE] += list(self.unembed.parameters())
        groups[CORE] += list(self.norm_out.parameters())
        for block in self.blocks:
            groups[CORE] += list(block.attention.parameters())
            groups[CORE] += list(block.norm_attention.parameters())
            groups[CORE] += list(block.norm_mlp.parameters())
            for name, mlp in block.mlps.items():
                groups[name] += list(mlp.parameters())

        grouped = sum(len(parameters) for parameters in groups.values())
        assert grouped == len(list(self.parameters())), "parameter groups must cover every parameter exactly once"
        return groups

    def parameter_group_names(self) -> dict[str, list[str]]:
        """The same partition as ``parameter_groups``, as state-dict names."""
        name_of = {id(parameter): name for name, parameter in self.named_parameters()}
        return {
            group: [name_of[id(parameter)] for parameter in parameters]
            for group, parameters in self.parameter_groups().items()
        }

    def num_parameters(self) -> dict[str, int]:
        return {
            group: sum(parameter.numel() for parameter in parameters)
            for group, parameters in self.parameter_groups().items()
        }


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.RMSNorm):
        nn.init.ones_(module.weight)


def story_causal_mask(tokens: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Attention mask of shape (batch, 1, time, time); True means "may attend".

    Training sequences are fixed-length slices of a long stream of stories, so one
    sequence usually contains the end of one story and the start of the next. A position
    may attend to an earlier position only if both belong to the same story, which makes
    each story's predictions independent of whatever happens to precede it.

    The end-of-story token closes its own story: it has the same story index as the
    tokens before it, and the token after it starts a new story.
    """
    is_eos = (tokens == eos_token_id).int()
    story_index = torch.cumsum(is_eos, dim=1) - is_eos
    same_story = story_index[:, :, None] == story_index[:, None, :]
    time = tokens.shape[1]
    causal = torch.ones(time, time, dtype=torch.bool, device=tokens.device).tril()
    return (same_story & causal)[:, None, :, :]


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy per token.

    The loss is computed in float32 even when the model runs in bfloat16: summing tens
    of thousands of per-token losses in bfloat16 would quantise the result to about
    0.008 nats, which is the same size as the effects this project measures.
    """
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))


def mlp_dims_for(method: str, config: ModelConfig, aux_topics: Sequence[str]) -> dict[str, int]:
    """MLP modules, and their hidden widths, for the given training method."""
    if method in ("baseline", "filtering"):
        return {CORE: config.dense_mlp_dim}
    if method == "gram":
        return {CORE: config.core_mlp_dim, **{topic: config.aux_mlp_dim for topic in aux_topics}}
    raise ValueError(f"unknown method {method!r}")


def build_model(method: str, config: ModelConfig, aux_topics: Sequence[str]) -> Transformer:
    return Transformer(config, mlp_dims_for(method, config, aux_topics))
