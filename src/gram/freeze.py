"""Freezing a module for a single forward pass (paper Appendix H).

Why this exists
---------------
Gradient routing means "this batch may update these parameter groups and no others".
A group that must not be updated still has to run in the forward pass: on an auxiliary
batch the core MLP is active even when the core group is not being updated, and its
output feeds everything downstream.

So we need a way to run a module such that

* its output is exactly what it would normally be,
* gradients still flow *through* it to whatever produced its input
  (an auxiliary module in an earlier layer still needs its gradient), but
* nothing is added to the module's own ``.grad``.

Detaching the *parameters* does precisely that. ``param.detach()`` shares memory with
the parameter, so the numbers are identical, but it is cut out of the autograd graph,
so backpropagation has nowhere to deposit a gradient for it. The module's *input* is
left attached, which is what keeps the gradient flowing to earlier layers.

Why not just skip ``optimizer.step()`` for that group? With one micro-batch per
optimizer step that would be enough. With gradient accumulation it is not: several
micro-batches, each with its own routing decision, add into the same ``.grad`` buffers
before a single step is taken, and once summed their contributions cannot be told
apart. Freezing at the micro-batch level keeps each group's accumulated gradient made
up only of the micro-batches that were routed to it.
"""

from __future__ import annotations

import torch
from torch import nn


def run_module(module: nn.Module, *args, frozen: bool = False, **kwargs):
    """Call ``module(*args, **kwargs)``; if ``frozen``, use detached copies of its parameters.

    Buffers (for example the rotary position tables) are not parameters and are passed
    through untouched.
    """
    if not frozen:
        return module(*args, **kwargs)

    detached_parameters = {name: parameter.detach() for name, parameter in module.named_parameters()}
    return torch.func.functional_call(module, detached_parameters, args, kwargs)
