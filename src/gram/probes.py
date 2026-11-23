"""Gradient probes: what gradient would each parameter group receive from each topic?

A probe is a measurement, taken at fixed points during training, that does not affect
training in any way. It asks: if the model, as it is right now, were shown a batch of
topic t, what gradient would each parameter group get, and how does that depend on the
routing rule the batch is run under?

Two fixed held-out batches per dataset (A and B) are used. They come from the ``probe``
split, which is never trained on and is not part of the validation data.

Two routing rules are compared:

* the **unlabeled rule**: every module active, every group updated (what GRAM does with
  a batch whose label it does not know);
* the **labeled rule** for module i: core and module i active.

For every topic t and parameter group g, one row is recorded:

(a) ``relative_grad_norm``: |gradient of g| / |parameters of g| under the unlabeled rule,
    for batch A and for batch B. This is the topic x group matrix: how hard would
    unlabeled data of topic t pull on each group?

For an auxiliary topic i, the row also carries cosine similarities between the
unlabeled-rule gradient from batch A and a labeled-rule gradient from batch B, restricted
to the parameters of group g. Two different batches are used so that the cosine measures
agreement about *direction*, not the shared noise of one batch.

(b) g = module i  (``cosine_kind = "matched"``): labeled rule is core + module i.
    Does unlabeled topic-i data push module i the way labeled topic-i data does?
(c) g = core      (``cosine_kind = "core"``): the same comparison on the core parameters.
    g = module j, j != i (``cosine_kind = "mismatched"``), as a control, in two versions:

    * ``cos_unlabeled_vs_labeled``: labeled rule is core + module j, still on topic i's
      batch B. "Topic i's batch, module j's parameters": the gradient module j would get
      if topic i were (mis)labeled as topic j.
    * ``cos_unlabeled_vs_own_topic_labeled``: the labeled gradient module j gets from its
      *own* topic's batch B. A null comparison: does unlabeled topic-i data push module
      j along the direction of module j's own specialty?

Every row also has ``cos_unlabeled_a_vs_b``, the cosine between the unlabeled-rule
gradients of batches A and B. Two batches of the same topic under the *same* rule never
agree perfectly, so this is the ceiling against which the cosines above should be read.

Gradients are measured on a float32 copy of the model. Training itself runs in bfloat16;
the copy holds exactly the same weights, the gradients are just computed more precisely.
Working on a copy is also what guarantees that a probe cannot touch the model being
trained or its optimizer state.
"""

from __future__ import annotations

import contextlib
import copy
import math
import random
from typing import Sequence

import numpy as np
import torch

from gram.config import CORE
from gram.model import Transformer, next_token_loss

GroupGradients = dict[str, list[torch.Tensor]]


@contextlib.contextmanager
def preserved_random_state():
    """Restore every random number generator on exit, so a probe cannot shift later random draws."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def group_gradients(
    model: Transformer, tokens: torch.Tensor, active: Sequence[str]
) -> tuple[GroupGradients, float]:
    """Gradient of the loss on one batch, for every active group, plus the loss itself."""
    model.zero_grad(set_to_none=True)
    tokens = tokens.long()
    loss = next_token_loss(model(tokens[:, :-1], active=active), tokens[:, 1:])
    loss.backward()
    groups = model.parameter_groups()
    gradients = {name: [parameter.grad.detach().clone() for parameter in groups[name]] for name in active}
    model.zero_grad(set_to_none=True)
    return gradients, loss.item()


def norm(tensors: Sequence[torch.Tensor]) -> float:
    """Euclidean norm of a list of tensors taken together as one vector, accumulated in float64."""
    return math.sqrt(sum(float(tensor.detach().double().pow(2).sum()) for tensor in tensors))


def cosine(first: Sequence[torch.Tensor], second: Sequence[torch.Tensor]) -> float:
    """Cosine similarity between two lists of tensors, each taken together as one vector."""
    dot = sum(float((a.detach().double() * b.detach().double()).sum()) for a, b in zip(first, second))
    denominator = norm(first) * norm(second)
    return dot / denominator if denominator > 0 else float("nan")


def run_gradient_probes(
    model: Transformer,
    probe_batches: dict[str, tuple[torch.Tensor, torch.Tensor]],
    step: int,
) -> list[dict]:
    """Take one set of probe measurements. Returns one row per (topic, parameter group).

    ``probe_batches`` maps each dataset (core and every auxiliary topic) to its two fixed
    batches. Nothing about ``model`` is changed: not its parameters, not its gradients,
    not its training mode; and no random number generator is advanced.
    """
    with preserved_random_state():
        probe_model = copy.deepcopy(model).float()
        probe_model.eval()

        everything = probe_model.module_names
        aux = probe_model.aux_names
        parameter_norm = {name: norm(parameters) for name, parameters in probe_model.parameter_groups().items()}

        # The labeled rule for each module on its own topic's batch B: gradients for the
        # core group and for the module.
        own_labeled = {
            topic: group_gradients(probe_model, probe_batches[topic][1], (CORE, topic))[0] for topic in aux
        }

        rows = []
        for topic, (batch_a, batch_b) in probe_batches.items():
            unlabeled_a, loss_a = group_gradients(probe_model, batch_a, everything)
            unlabeled_b, loss_b = group_gradients(probe_model, batch_b, everything)

            for group in everything:
                grad_norm_a, grad_norm_b = norm(unlabeled_a[group]), norm(unlabeled_b[group])
                row = {
                    "step": step,
                    "topic": topic,
                    "group": group,
                    "param_norm": parameter_norm[group],
                    "grad_norm_a": grad_norm_a,
                    "grad_norm_b": grad_norm_b,
                    "relative_grad_norm_a": grad_norm_a / parameter_norm[group],
                    "relative_grad_norm_b": grad_norm_b / parameter_norm[group],
                    "loss_unlabeled_a": loss_a,
                    "loss_unlabeled_b": loss_b,
                    "cos_unlabeled_a_vs_b": cosine(unlabeled_a[group], unlabeled_b[group]),
                    "cosine_kind": "",
                    "cos_unlabeled_vs_labeled": float("nan"),
                    "cos_unlabeled_vs_own_topic_labeled": float("nan"),
                }

                if topic in aux:
                    if group == topic:
                        row["cosine_kind"] = "matched"
                        row["cos_unlabeled_vs_labeled"] = cosine(unlabeled_a[group], own_labeled[topic][topic])
                    elif group == CORE:
                        row["cosine_kind"] = "core"
                        row["cos_unlabeled_vs_labeled"] = cosine(unlabeled_a[CORE], own_labeled[topic][CORE])
                    else:
                        # Topic i's batch B run as if it were labeled with module j.
                        as_other_module, _ = group_gradients(probe_model, batch_b, (CORE, group))
                        row["cosine_kind"] = "mismatched"
                        row["cos_unlabeled_vs_labeled"] = cosine(unlabeled_a[group], as_other_module[group])
                        row["cos_unlabeled_vs_own_topic_labeled"] = cosine(unlabeled_a[group], own_labeled[group][group])
                rows.append(row)

        del probe_model
    return rows


def probe_steps(total_steps: int, num_points: int) -> list[int]:
    """The steps after which probes are taken: evenly spaced, ending with the last step."""
    return sorted({max(1, round(point * total_steps / num_points)) for point in range(1, num_points + 1)})
