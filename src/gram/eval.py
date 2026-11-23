"""Validation losses.

Everything this project reports starts as a mean next-token loss of some model, run
with some set of modules switched on, on some held-out dataset. This module computes
those raw losses. Turning losses into compute ratios is a separate, torch-free step
(gram/compute_ratio.py), so the raw losses are always saved first.

Profiles
--------
A *profile* is a set of active modules. For a GRAM model we score:

* ``core_only``          all auxiliary modules ablated. This is the inference profile of
                         this project: its loss on the core data is "core", and its loss
                         on each auxiliary topic is "forget".
* ``only:<topic>``       core plus that one module. Its loss on its own topic is "retain".
* ``without:<topic>``    every module except that one. Used by the wrong-module
                         experiments, to ask which module a topic actually ended up in.
* ``all_modules``        everything on.

A dense model has a single profile, ``dense``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gram.config import CORE
from gram.data import rng_for
from gram.model import Transformer


@torch.no_grad()
def mean_loss(
    model: Transformer,
    sequences: torch.Tensor,
    active: Sequence[str],
    batch_size: int,
) -> float:
    """Mean next-token loss, in nats per token, over all the given sequences.

    ``sequences`` has shape (n, context_length + 1). Every sequence contributes the same
    number of predicted tokens, so this is the plain mean over all predictions. The
    result does not depend on ``batch_size`` (up to float rounding) and involves no
    randomness, so evaluating a saved checkpoint again gives the same number.
    """
    if len(sequences) == 0:
        return float("nan")
    was_training = model.training
    model.eval()
    total = 0.0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size].long()
        logits = model(batch[:, :-1], active=active)
        # Summed in float32 (see model.next_token_loss), accumulated across batches in float64.
        total += F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum"
        ).item()
    model.train(was_training)
    return total / (len(sequences) * (sequences.shape[1] - 1))


def evaluation_profiles(model: Transformer) -> dict[str, tuple[str, ...]]:
    """The profiles a model is scored under: ``{profile name: active modules}``."""
    if not model.aux_names:
        return {"dense": (CORE,)}
    profiles = {"core_only": (CORE,)}
    for topic in model.aux_names:
        profiles[f"only:{topic}"] = (CORE, topic)
    for topic in model.aux_names:
        profiles[f"without:{topic}"] = tuple(name for name in model.module_names if name != topic)
    profiles["all_modules"] = model.module_names
    return profiles


def evaluate_profiles(
    model: Transformer,
    validation_sets: dict[str, torch.Tensor],
    batch_size: int,
    profiles: dict[str, tuple[str, ...]] | None = None,
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Loss of every profile on every dataset. One row per (profile, dataset)."""
    profiles = evaluation_profiles(model) if profiles is None else profiles
    datasets = list(validation_sets) if datasets is None else list(datasets)
    rows = []
    for profile, active in profiles.items():
        for dataset in datasets:
            rows.append(
                {
                    "profile": profile,
                    "active": "+".join(active),
                    "dataset": dataset,
                    "loss": mean_loss(model, validation_sets[dataset], active, batch_size),
                    "num_sequences": len(validation_sets[dataset]),
                }
            )
    return pd.DataFrame(rows)


def curve_profiles(model: Transformer) -> list[tuple[str, tuple[str, ...], str | None]]:
    """What a mid-training validation sweep scores: (profile, active modules, dataset or None for all).

    For the dense baseline this is simply the model on every dataset: these sweeps are the
    learning curves that the compute-ratio metric is built on. For a GRAM model (should
    sweeps be switched on) it is the core-only profile on every dataset, plus each module
    on its own topic.
    """
    if not model.aux_names:
        return [("dense", (CORE,), None)]
    return [("core_only", (CORE,), None)] + [(f"only:{topic}", (CORE, topic), topic) for topic in model.aux_names]


def validation_sweep(
    model: Transformer,
    validation_sets: dict[str, torch.Tensor],
    step: int,
    seed: int,
    core_sequences: int,
    batch_size: int,
) -> list[dict]:
    """One point of the learning curve: the loss on every dataset at the current step.

    Auxiliary topics are scored on their whole validation set. The core validation set is
    about forty times larger, so each sweep scores a fresh random sample of it instead (a
    sample redrawn at every sweep keeps the fitted curve unbiased for the full set, which
    a fixed subset would not).
    """
    rows = []
    for profile, active, only_dataset in curve_profiles(model):
        for dataset, sequences in validation_sets.items():
            if only_dataset is not None and dataset != only_dataset:
                continue
            if dataset == CORE and core_sequences < len(sequences):
                sample = rng_for(seed, f"curve:{step}").choice(len(sequences), size=core_sequences, replace=False)
                sequences = sequences[torch.from_numpy(np.sort(sample)).to(sequences.device)]
            rows.append(
                {
                    "step": step,
                    "profile": profile,
                    "dataset": dataset,
                    "loss": mean_loss(model, sequences, active, batch_size),
                    "num_sequences": len(sequences),
                }
            )
    return rows


# How many sequences of each dataset go into a model's fingerprint.
FINGERPRINT_SEQUENCES = 64


def fingerprint(model: Transformer, validation_sets: dict[str, torch.Tensor], batch_size: int) -> dict[str, float]:
    """A handful of losses that identify a model: every profile on the first sequences of every dataset.

    Training records the fingerprint of the model it has in memory; evaluation recomputes
    it from the checkpoint on disk. If the two agree, saving and loading lost nothing.
    """
    small = {name: tokens[:FINGERPRINT_SEQUENCES] for name, tokens in validation_sets.items()}
    table = evaluate_profiles(model, small, batch_size)
    return {f"{row.profile}|{row.dataset}": float(row.loss) for row in table.itertuples()}
