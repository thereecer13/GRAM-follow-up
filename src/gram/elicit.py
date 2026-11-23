"""Elicited forget: can a small amount of finetuning bring a removed capability back?

A capability that was merely suppressed comes back quickly when the model is finetuned
on a little data from the removed topic; one that was never learned does not. So after
measuring the forget loss of a model, an adversary finetunes it on a small sample of the
forget topic, and the loss on that topic is measured again.

The protocol is the official code's (``do_finetune``):

* The adversary gets ``num_sequences`` training sequences of the forget topic.
* Each step is one full pass over those sequences (gradients accumulated over
  micro-batches, then a single optimizer step): full-batch gradient descent.
* AdamW with PyTorch's default settings, a constant learning rate of a quarter of the
  training learning rate, and no gradient clipping.
* The model is finetuned *in its ablated form*: only the modules of the profile are
  active, so only they can change. The removed module stays removed.
* After every step the loss on the topic's validation set is measured, and the adversary
  keeps the best checkpoint: the reported loss is the lowest one seen. Optionally the
  run stops early once ``patience`` steps have passed without a new best.

Two protocols are run on every model (see ElicitConfig): the one the official code runs
(512 sequences, early stopping), and the one the paper's Appendix A describes (128
sequences, 75 steps).
"""

from __future__ import annotations

import copy
from typing import Sequence

import pandas as pd
import torch

from gram.config import CORE, ElicitProtocol, RunConfig
from gram.data import TokenStore, build_elicitation_sequences
from gram.eval import mean_loss
from gram.model import Transformer, next_token_loss


def elicit_topic(
    model: Transformer,
    train_sequences: torch.Tensor,
    validation_sequences: torch.Tensor,
    active: Sequence[str],
    lr: float,
    protocol: ElicitProtocol,
    micro_batch_size: int,
    eval_batch_size: int,
) -> list[dict]:
    """Finetune a copy of ``model`` on one topic. Returns the validation loss after every step.

    The first row (step 0) is the loss before any finetuning. ``model`` is not modified.
    """
    model = copy.deepcopy(model)
    model.train()
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([p for name in active for p in groups[name]], lr=lr, fused=True)

    micro_batches = [
        train_sequences[start : start + micro_batch_size].long()
        for start in range(0, len(train_sequences), micro_batch_size)
    ]
    rows = [{"step": 0, "train_loss": float("nan"), "val_loss": mean_loss(model, validation_sequences, active, eval_batch_size)}]

    best_loss = float("inf")
    steps_without_improvement = 0
    for step in range(1, protocol.max_steps + 1):
        train_loss = 0.0
        for tokens in micro_batches:
            loss = next_token_loss(model(tokens[:, :-1], active=active, update=active), tokens[:, 1:])
            (loss / len(micro_batches)).backward()
            train_loss += loss.item() / len(micro_batches)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        val_loss = mean_loss(model, validation_sequences, active, eval_batch_size)
        rows.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss <= best_loss:
            best_loss = val_loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        if protocol.patience > 0 and steps_without_improvement > protocol.patience:
            break
    return rows


def summarise_elicitation(curve: pd.DataFrame) -> pd.DataFrame:
    """One row per (protocol, topic): the loss before, the best loss the adversary reached, and when.

    As in the official code, the best is taken over the finetuned checkpoints only
    (step 1 onwards), not over the untouched model.
    """
    rows = []
    for (protocol, topic), group in curve.groupby(["protocol", "topic"], sort=False):
        finetuned = group[group.step >= 1]
        best = finetuned.loc[finetuned.val_loss.idxmin()]
        rows.append(
            {
                "protocol": protocol,
                "topic": topic,
                "loss_before": float(group.loc[group.step == 0, "val_loss"].iloc[0]),
                "loss_elicited": float(best.val_loss),
                "best_step": int(best.step),
                "loss_at_last_step": float(finetuned.val_loss.iloc[-1]),
                "steps_run": int(finetuned.step.max()),
            }
        )
    return pd.DataFrame(rows)


def run_elicitation(
    model: Transformer,
    config: RunConfig,
    store: TokenStore,
    validation_sets: dict[str, torch.Tensor],
    device: torch.device,
) -> pd.DataFrame:
    """Elicit every auxiliary topic from the core-only form of the model, under every protocol.

    Returns the full curves: one row per (protocol, topic, step).
    """
    from gram.train import token_tensor  # here to avoid a circular import

    active = (CORE,)  # the project's inference profile: every auxiliary module ablated
    lr = config.optim.lr * config.elicit.lr_factor
    rows = []
    for protocol in config.elicit.protocols:
        for topic in config.data.aux_topics:
            train_sequences = token_tensor(
                build_elicitation_sequences(
                    store, topic, protocol.num_sequences, config.model.context_length, config.seed
                ),
                device,
            )
            curve = elicit_topic(
                model, train_sequences, validation_sets[topic], active, lr, protocol,
                config.optim.micro_batch_size, config.eval.batch_size,
            )
            rows.extend({"protocol": protocol.name, "topic": topic, **row} for row in curve)
    return pd.DataFrame(rows)
