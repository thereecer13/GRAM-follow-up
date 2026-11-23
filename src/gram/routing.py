"""The routing plan: which rule each micro-batch of a run is trained under.

A *rule* says, for one micro-batch:

* which pool of data it is drawn from (pools are defined by the *given* label),
* which MLP modules are ``active`` in the forward pass, and
* which parameter groups are ``update``-d by the backward pass.

The rules GRAM uses (paper Section 3 and Appendix G):

==============  ==========  ==================  ===================  ====================
rule            pool        active              update               how many
==============  ==========  ==================  ===================  ====================
core_plain      core        core                core                 (1 - p_cr) of core
core_robust     core        core + module i     core + module i      p_cr of core
aux_isolated    topic i     core + module i     module i             (1 - p_as) of topic i
aux_spread      topic i     core + module i     core + module i      p_as of topic i
unlabeled       unlabeled   everything          everything           all unlabeled batches
==============  ==========  ==================  ===================  ====================

Dense models (baseline and filtering) have a single rule, ``dense``: the one MLP is
active and everything is updated.

The paper describes the choices as probabilities ("with probability p_cr ..."). The
official code does not flip a coin per batch. It fixes the *number* of batches under
each rule to the exact proportion and then shuffles the order, which gives the same
mixture without the sampling noise. We do the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from gram.config import CORE, UNLABELED, RoutingConfig


@dataclass(frozen=True)
class Rule:
    name: str  # dense | core_plain | core_robust | aux_isolated | aux_spread | unlabeled
    pool: str  # the given label whose pool the batch is drawn from
    active: tuple[str, ...]  # MLP modules in the forward pass
    update: tuple[str, ...]  # parameter groups that receive gradients

    @property
    def module(self) -> str | None:
        """The single auxiliary module this rule activates, if it activates exactly one."""
        aux = [name for name in self.active if name != CORE]
        return aux[0] if len(aux) == 1 else None


@dataclass
class Plan:
    """The rules of a run, how often each is used, and the order they are used in."""

    rules: list[Rule]
    counts: list[int]  # micro-batches trained under each rule
    schedule: np.ndarray  # for every micro-batch, in training order, an index into `rules`
    accumulation_steps: int
    notes: dict = field(default_factory=dict)  # the arithmetic behind the counts, for the log

    @property
    def num_steps(self) -> int:
        """Number of optimizer steps."""
        return len(self.schedule) // self.accumulation_steps

    def count_by_rule_name(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for rule, count in zip(self.rules, self.counts):
            totals[rule.name] = totals.get(rule.name, 0) + count
        return totals


def _shuffled_schedule(counts: Sequence[int], accumulation_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Lay the rules out in a random order, one entry per micro-batch.

    With gradient accumulation, consecutive groups of ``accumulation_steps`` entries
    form one optimizer step. Because the order is shuffled at the level of micro-batches,
    the micro-batches inside one step generally follow *different* rules. This is the
    paper's "heterogeneous accumulation" (Appendix H); it is what makes freezing
    necessary. Entries that do not fill a last step are dropped.
    """
    schedule = np.repeat(np.arange(len(counts)), counts)
    schedule = rng.permutation(schedule)
    usable = len(schedule) // accumulation_steps * accumulation_steps
    return schedule[:usable]


def build_dense_plan(
    pool_sizes: Mapping[str, int],
    pools: Sequence[str],
    accumulation_steps: int,
    rng: np.random.Generator,
) -> Plan:
    """Plan for a dense model: one pass over every batch of the given pools."""
    rules = [Rule("dense", pool, (CORE,), (CORE,)) for pool in pools if pool_sizes.get(pool, 0) > 0]
    counts = [pool_sizes[rule.pool] for rule in rules]
    return Plan(
        rules=rules,
        counts=counts,
        schedule=_shuffled_schedule(counts, accumulation_steps, rng),
        accumulation_steps=accumulation_steps,
        notes={"pools": list(pools), "micro_batches": int(sum(counts))},
    )


def gram_rule_counts(
    pool_sizes: Mapping[str, int],
    aux_topics: Sequence[str],
    routing: RoutingConfig,
    group_params: Mapping[str, int],
    baseline_params: int,
    module_weights: Mapping[str, float] | None = None,
) -> tuple[dict[tuple[str, str | None], int], dict]:
    """How many micro-batches GRAM trains under each rule.

    ``pool_sizes[label]`` is the number of micro-batches in one pass over that pool.
    Returns ``{(rule name, module or None): count}`` and a dictionary of the
    intermediate numbers.

    This follows the official ``do_routed_unordered`` line for line, including its use
    of Python's ``round``, so that the counts can be checked against the batch counts
    printed in the official training logs (see tests/test_routing.py).

    ``module_weights`` decides how the core-robust batches, and the auxiliary batches
    after any rescaling, are shared out between the modules. The official code uses the
    sizes of the auxiliary datasets, which is also the default here.
    """
    p_cr, p_as, p_af = routing.p_cr, routing.p_as, routing.p_af

    n_core = pool_sizes.get(CORE, 0)
    n_aux = {topic: pool_sizes.get(topic, 0) for topic in aux_topics}
    n_unlabeled = pool_sizes.get(UNLABELED, 0)
    n_labeled = n_core + sum(n_aux.values())
    n_all = n_labeled + n_unlabeled

    # Over-sampling auxiliary data (p_af > 1) takes the extra batches away from core, so
    # that the amount of labeled training stays the same.
    aux_total = sum(round(n_aux[topic] * p_af) for topic in aux_topics)
    if aux_total > n_labeled:
        raise ValueError("p_af asks for more auxiliary batches than there are labeled batches")
    core_total = n_labeled - aux_total

    # Core robustness: a fraction p_cr of the core batches run with one module active.
    core_robust = round(core_total * p_cr)
    core_plain = core_total - core_robust

    # GRAM has two modes of labeled batch: "core only" (the core MLP alone) and
    # "module" (core plus one auxiliary module: all auxiliary batches, and the
    # core-robust batches).
    core_only_mode = core_plain
    module_mode = aux_total + core_robust
    robust_share = core_robust / module_mode if module_mode > 0 else 0.0
    unlabeled = n_unlabeled

    notes = {
        "before_equal_compute": {"core_only_mode": core_only_mode, "module_mode": module_mode, "unlabeled": unlabeled},
    }

    if routing.equal_compute:
        # Appendix I. Training cost is approximated as (batches) x (active parameters).
        # GRAM's batches do not all cost the same as a baseline batch, so the number of
        # batches of each mode is scaled by a common factor lambda = F_baseline / F_gram
        # until the total cost equals the baseline's. lambda < 1 shortens training.
        core_params = group_params[CORE]
        aux_params = [group_params[topic] for topic in aux_topics]
        mean_aux_params = sum(aux_params) / len(aux_params)
        all_aux_params = sum(aux_params)

        cost_baseline = n_all * baseline_params
        cost_core_only = core_only_mode * core_params
        cost_module = module_mode * (core_params + mean_aux_params)
        cost_unlabeled = unlabeled * (core_params + all_aux_params)
        cost_gram = cost_core_only + cost_module + cost_unlabeled
        difference = cost_baseline - cost_gram

        core_only_mode += round(cost_core_only / cost_gram * difference / core_params)
        module_mode += round(cost_module / cost_gram * difference / (core_params + mean_aux_params))
        unlabeled += round(cost_unlabeled / cost_gram * difference / (core_params + all_aux_params))
        notes["equal_compute"] = {
            "lambda": cost_baseline / cost_gram,
            "core_only_mode": core_only_mode,
            "module_mode": module_mode,
            "unlabeled": unlabeled,
        }

    core_robust = round(module_mode * robust_share)
    aux_total = module_mode - core_robust

    if module_weights is None:
        module_weights = {topic: n_aux[topic] * p_af for topic in aux_topics}
    weight_sum = sum(module_weights[topic] for topic in aux_topics)
    if weight_sum > 0:
        share = {topic: module_weights[topic] / weight_sum for topic in aux_topics}
    else:  # no auxiliary data at all: share the core-robust batches out evenly
        share = {topic: 1.0 / len(aux_topics) for topic in aux_topics}

    counts: dict[tuple[str, str | None], int] = {("core_plain", None): core_only_mode}
    for topic in aux_topics:
        aux_batches = round(aux_total * share[topic])
        # Auxiliary spread applies to p_as of the topic's *original* data: if the topic
        # is over-sampled p_af times, the core is still updated on the same number of batches.
        spread = round(aux_batches * p_as / p_af)
        counts[("aux_isolated", topic)] = aux_batches - spread
        counts[("aux_spread", topic)] = spread
        counts[("core_robust", topic)] = round(core_robust * share[topic])
    counts[("unlabeled", None)] = unlabeled

    notes["module_share"] = share
    return counts, notes


def build_gram_plan(
    pool_sizes: Mapping[str, int],
    aux_topics: Sequence[str],
    routing: RoutingConfig,
    group_params: Mapping[str, int],
    baseline_params: int,
    accumulation_steps: int,
    rng: np.random.Generator,
    module_weights: Mapping[str, float] | None = None,
) -> Plan:
    """Plan for a GRAM run: the counts from ``gram_rule_counts``, in a shuffled order."""
    counts_by_rule, notes = gram_rule_counts(
        pool_sizes, aux_topics, routing, group_params, baseline_params, module_weights
    )
    everything = (CORE, *aux_topics)

    def rule_for(name: str, topic: str | None) -> Rule:
        if name == "core_plain":
            return Rule(name, CORE, (CORE,), (CORE,))
        if name == "core_robust":
            return Rule(name, CORE, (CORE, topic), (CORE, topic))
        if name == "aux_isolated":
            return Rule(name, topic, (CORE, topic), (topic,))
        if name == "aux_spread":
            return Rule(name, topic, (CORE, topic), (CORE, topic))
        if name == "unlabeled":
            return Rule(name, UNLABELED, everything, everything)
        raise ValueError(name)

    rules, counts = [], []
    for (name, topic), count in counts_by_rule.items():
        if count > 0:
            rules.append(rule_for(name, topic))
            counts.append(count)

    notes["micro_batches"] = int(sum(counts))
    return Plan(
        rules=rules,
        counts=counts,
        schedule=_shuffled_schedule(counts, accumulation_steps, rng),
        accumulation_steps=accumulation_steps,
        notes=notes,
    )
