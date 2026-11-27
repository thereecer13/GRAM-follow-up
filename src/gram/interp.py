"""Looking inside a trained GRAM model: does module i hold topic i?

Everything here is a measurement on a finished model. Nothing is trained except the small
linear probes of ``topic_probe_accuracy``, which never touch the model's weights.

The questions, each answered by one function:

* ``layer_ablation``      Which layers of module i matter? Switch module i off in one block
                          at a time and see how much the loss rises, on its own topic and
                          on other data.
* ``token_effects``       Which words does module i help with? The loss of every predicted
                          token with and without module i, grouped by token.
* ``output_norms``        How loudly does each MLP speak? The size of each module's output
                          on each topic's text, block by block.
* ``topic_probe_accuracy``  Is the topic *readable* from an MLP's output? A linear
                          classifier per block and per MLP.
* ``direct_writes``       Which tokens does module i push for on its own topic? Its mean
                          output there, read through the unembedding (an approximation).

They share one piece of machinery: ``run``, a forward pass that can use a different set
of active modules in every block and returns what each MLP wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gram.config import CORE
from gram.model import Transformer, story_causal_mask

ALL_LAYERS = -1  # stands for "every block" in the tables


@dataclass
class Forward:
    logits: torch.Tensor  # (batch, time, vocab)
    mlp_outputs: list[dict[str, torch.Tensor]]  # per block: {module name: (batch, time, d_model)}
    residual: list[torch.Tensor]  # per block: the residual stream after the block


@torch.no_grad()
def run(model: Transformer, tokens: torch.Tensor, active_per_layer: Sequence[Sequence[str]], keep: bool = False) -> Forward:
    """A forward pass in which block L uses the modules ``active_per_layer[L]``.

    With the same modules in every block this computes exactly what ``model(tokens,
    active=...)`` computes (a test checks it). ``keep=True`` also returns every MLP's
    output and the residual stream, which costs memory.
    """
    mask = story_causal_mask(tokens, model.config.eos_token_id)
    x = model.embed(tokens)
    outputs, residual = [], []
    for block, active in zip(model.blocks, active_per_layer):
        x = x + block.attention(block.norm_attention(x), mask)
        normed = block.norm_mlp(x)
        written = {name: block.mlps[name](normed) for name in active}
        for output in written.values():
            x = x + output
        if keep:
            outputs.append(written)
            residual.append(x)
    return Forward(model.unembed(model.norm_out(x)), outputs, residual)


def everywhere(model: Transformer, active: Sequence[str]) -> list[tuple[str, ...]]:
    """The same active modules in every block."""
    return [tuple(active)] * len(model.blocks)


def without(model: Transformer, module: str, layer: int) -> list[tuple[str, ...]]:
    """Every module on, except ``module`` in block ``layer`` (or in every block, for ALL_LAYERS)."""
    full = tuple(model.module_names)
    reduced = tuple(name for name in full if name != module)
    return [reduced if layer in (ALL_LAYERS, index) else full for index in range(len(model.blocks))]


def batches(sequences: torch.Tensor, batch_size: int):
    for start in range(0, len(sequences), batch_size):
        yield sequences[start : start + batch_size].long()


def token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Loss of every predicted token, shape (batch, time), in float32."""
    return F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")


def mean_loss(model: Transformer, sequences: torch.Tensor, active_per_layer, batch_size: int) -> float:
    total, count = 0.0, 0
    for batch in batches(sequences, batch_size):
        losses = token_losses(run(model, batch[:, :-1], active_per_layer).logits, batch[:, 1:])
        total += losses.sum().item()
        count += losses.numel()
    return total / count


# --------------------------------------------------------------------------- #
# 1. Which layers of a module matter?
# --------------------------------------------------------------------------- #


def layer_ablation(model: Transformer, datasets: dict[str, torch.Tensor], batch_size: int = 256) -> pd.DataFrame:
    """Loss on every dataset with one module switched off in one block (or in all of them).

    The reference is the model with every module on. One row per (module, layer,
    dataset); ``loss_increase`` is how much worse the model gets, in nats per token.
    ``layer`` is -1 for "module off in every block".
    """
    full = everywhere(model, model.module_names)
    reference = {name: mean_loss(model, sequences, full, batch_size) for name, sequences in datasets.items()}
    rows = []
    for module in model.aux_names:
        for layer in [ALL_LAYERS, *range(len(model.blocks))]:
            for name, sequences in datasets.items():
                loss = mean_loss(model, sequences, without(model, module, layer), batch_size)
                rows.append(
                    {
                        "module": module, "layer": layer, "dataset": name, "own_topic": name == module,
                        "loss_all_modules": reference[name], "loss": loss, "loss_increase": loss - reference[name],
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Which words does a module help with?
# --------------------------------------------------------------------------- #


def token_effects(model: Transformer, datasets: dict[str, torch.Tensor], vocabulary: Sequence[str], batch_size: int = 256) -> pd.DataFrame:
    """For every module and every token of the vocabulary: how much removing the module hurts.

    Each predicted token's loss is computed with every module on and with one module
    removed from every block. The differences are averaged per target token, separately
    on the module's own topic and on core data. One row per (module, dataset, token).
    """
    full = everywhere(model, model.module_names)
    vocab_size = model.config.vocab_size
    rows = []
    for module in model.aux_names:
        reduced = without(model, module, ALL_LAYERS)
        for name in (module, CORE):
            count = torch.zeros(vocab_size, dtype=torch.float64)
            loss_sum = torch.zeros(vocab_size, dtype=torch.float64)
            increase_sum = torch.zeros(vocab_size, dtype=torch.float64)
            for batch in batches(datasets[name], batch_size):
                inputs, targets = batch[:, :-1], batch[:, 1:]
                with_module = token_losses(run(model, inputs, full).logits, targets)
                without_module = token_losses(run(model, inputs, reduced).logits, targets)
                flat = targets.reshape(-1).cpu()
                count.index_add_(0, flat, torch.ones(flat.shape, dtype=torch.float64))
                loss_sum.index_add_(0, flat, with_module.reshape(-1).double().cpu())
                increase_sum.index_add_(0, flat, (without_module - with_module).reshape(-1).double().cpu())
            seen = count > 0
            table = pd.DataFrame(
                {
                    "module": module, "dataset": name, "own_topic": name == module,
                    "token_id": torch.nonzero(seen).squeeze(1).numpy(),
                    "token": [vocabulary[index] for index in torch.nonzero(seen).squeeze(1).tolist()],
                    "count": count[seen].numpy().astype(int),
                    "loss_all_modules": (loss_sum[seen] / count[seen]).numpy(),
                    "loss_increase": (increase_sum[seen] / count[seen]).numpy(),
                }
            )
            table["share_of_total_increase"] = table.loss_increase * table["count"] / (table.loss_increase * table["count"]).sum()
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 3. How loudly does each MLP speak, and where?
# --------------------------------------------------------------------------- #


def output_norms(model: Transformer, datasets: dict[str, torch.Tensor], batch_size: int = 256) -> pd.DataFrame:
    """Mean length of each MLP's output vector, per block and per dataset, with every module on.

    ``relative_to_core`` divides by the core MLP's output in the same block on the same
    data, so 0.1 means "a tenth as large as what the core MLP writes".
    """
    full = everywhere(model, model.module_names)
    rows = []
    for name, sequences in datasets.items():
        totals = [dict.fromkeys(model.module_names, 0.0) for _ in model.blocks]
        count = 0
        for batch in batches(sequences, batch_size):
            forward = run(model, batch[:, :-1], full, keep=True)
            count += batch[:, :-1].numel()
            for layer, written in enumerate(forward.mlp_outputs):
                for module, output in written.items():
                    totals[layer][module] += output.float().norm(dim=-1).sum().item()
        for layer, by_module in enumerate(totals):
            for module, total in by_module.items():
                rows.append(
                    {
                        "dataset": name, "layer": layer, "module": module, "own_topic": name == module,
                        "mean_norm": total / count, "relative_to_core": total / by_module[CORE],
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Is the topic readable from an MLP's output?
# --------------------------------------------------------------------------- #


def pooled_outputs(model: Transformer, sequences: torch.Tensor, batch_size: int) -> dict[tuple[int, str], torch.Tensor]:
    """Each sequence summarised by the mean over positions of what each source wrote.

    Returns ``{(layer, source): (n_sequences, d_model)}`` where source is a module name or
    "residual" (the residual stream after the block).
    """
    full = everywhere(model, model.module_names)
    pooled: dict[tuple[int, str], list[torch.Tensor]] = {}
    for batch in batches(sequences, batch_size):
        forward = run(model, batch[:, :-1], full, keep=True)
        for layer, written in enumerate(forward.mlp_outputs):
            for module, output in written.items():
                pooled.setdefault((layer, module), []).append(output.float().mean(dim=1).cpu())
            pooled.setdefault((layer, "residual"), []).append(forward.residual[layer].float().mean(dim=1).cpu())
    return {key: torch.cat(parts) for key, parts in pooled.items()}


def fit_linear_probe(train_x, train_y, test_x, test_y, num_classes: int, steps: int = 300, seed: int = 0) -> float:
    """Test accuracy of a linear classifier (multinomial logistic regression) trained by gradient descent."""
    torch.manual_seed(seed)
    mean, scale = train_x.mean(dim=0), train_x.std(dim=0) + 1e-6
    train_x, test_x = (train_x - mean) / scale, (test_x - mean) / scale
    probe = torch.nn.Linear(train_x.shape[1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            F.cross_entropy(probe(train_x), train_y).backward()
            optimizer.step()
    return (probe(test_x).argmax(dim=1) == test_y).float().mean().item()


def topic_probe_accuracy(
    model: Transformer, datasets: dict[str, torch.Tensor], train_per_class: int, test_per_class: int, batch_size: int = 256
) -> pd.DataFrame:
    """How well a linear probe tells the five datasets apart from each MLP's (pooled) output.

    Classes are the four auxiliary topics and core. Chance is 20%. One row per (layer,
    source). A high accuracy says the topic is *present* in what that MLP writes; it does
    not say the model *uses* it, which is what the ablations measure.
    """
    names = list(datasets)
    features: dict[tuple[int, str], list[torch.Tensor]] = {}
    labels = []
    for index, name in enumerate(names):
        sequences = datasets[name][: train_per_class + test_per_class]
        if len(sequences) < train_per_class + test_per_class:
            raise ValueError(f"{name} has only {len(sequences)} sequences; need {train_per_class + test_per_class}")
        for key, values in pooled_outputs(model, sequences, batch_size).items():
            features.setdefault(key, []).append(values)
        labels.append(torch.full((len(sequences),), index))
    is_train = torch.cat([torch.arange(train_per_class + test_per_class) < train_per_class for _ in names])
    y = torch.cat(labels)
    rows = []
    for (layer, source), parts in features.items():
        x = torch.cat(parts)
        accuracy = fit_linear_probe(x[is_train], y[is_train], x[~is_train], y[~is_train], len(names))
        rows.append({"layer": layer, "source": source, "accuracy": accuracy, "chance": 1 / len(names)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Which tokens does a module push for?
# --------------------------------------------------------------------------- #


def direct_writes(model: Transformer, datasets: dict[str, torch.Tensor], vocabulary: Sequence[str], top: int = 25, batch_size: int = 256) -> pd.DataFrame:
    """The tokens each module's output points towards on its own topic, block by block.

    For module i in block L: the mean of what it writes on topic-i text, minus the mean
    of what it writes on core text, multiplied by the unembedding matrix. This is an
    approximation ("logit lens"): it ignores the later blocks and the final
    normalisation's rescaling, so read the ranking, not the numbers.
    """
    full = everywhere(model, model.module_names)

    def mean_outputs(sequences: torch.Tensor) -> dict[tuple[int, str], torch.Tensor]:
        sums: dict[tuple[int, str], torch.Tensor] = {}
        count = 0
        for batch in batches(sequences, batch_size):
            forward = run(model, batch[:, :-1], full, keep=True)
            count += batch[:, :-1].numel()
            for layer, written in enumerate(forward.mlp_outputs):
                for module in model.aux_names:
                    vector = written[module].float().sum(dim=(0, 1)).cpu()
                    sums[(layer, module)] = sums.get((layer, module), 0) + vector
        return {key: value / count for key, value in sums.items()}

    on_core = mean_outputs(datasets[CORE])
    unembed = model.unembed.weight.float().cpu() * model.norm_out.weight.float().cpu()
    rows = []
    for module in model.aux_names:
        on_topic = mean_outputs(datasets[module])
        for layer in range(len(model.blocks)):
            scores = unembed @ (on_topic[(layer, module)] - on_core[(layer, module)])
            best = torch.topk(scores, top)
            for rank, (score, token_id) in enumerate(zip(best.values.tolist(), best.indices.tolist()), start=1):
                rows.append({"module": module, "layer": layer, "rank": rank, "token_id": token_id, "token": vocabulary[token_id], "score": score})
    return pd.DataFrame(rows)


def sample(sequences: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """A fixed random subset of ``count`` sequences (all of them if there are fewer)."""
    if len(sequences) <= count:
        return sequences
    chosen = np.random.default_rng(seed).choice(len(sequences), size=count, replace=False)
    return sequences[np.sort(chosen)]
