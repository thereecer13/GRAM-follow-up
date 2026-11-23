"""The routing plan: how many micro-batches follow each rule, and in what order."""

import numpy as np
import pytest

from gram.config import AUX_TOPICS, CORE, UNLABELED, RoutingConfig
from gram.routing import build_dense_plan, build_gram_plan, gram_rule_counts

# Read off the official training logs for seed 1: micro-batches per label in one epoch,
# and the parameter count of each group of the GRAM model (core MLP width 2048).
OFFICIAL_POOL_SIZES = {
    CORE: 15143,
    "a-deadline-or-time-limit": 352,
    "alien-encounters": 345,
    "bygone-eras": 352,
    "cultural-traditions": 352,
}
OFFICIAL_GROUP_PARAMS = {CORE: 26_257_920, **{topic: 1_578_496 for topic in AUX_TOPICS}}
OFFICIAL_BASELINE_PARAMS = 26_257_920


def official_counts(**routing):
    return gram_rule_counts(
        OFFICIAL_POOL_SIZES, AUX_TOPICS, RoutingConfig(**routing), OFFICIAL_GROUP_PARAMS, OFFICIAL_BASELINE_PARAMS
    )


def per_topic(counts, rule):
    return [counts[(rule, topic)] for topic in AUX_TOPICS]


def test_counts_match_the_official_log_of_the_table_1_run():
    """results/stories/seed_1/20260616035947118035/training.log (p_cr 0.5, p_as 0.3, equal compute)."""
    counts, notes = official_counts(p_cr=0.5, p_as=0.3)
    assert counts[("core_plain", None)] == 7332
    assert per_topic(counts, "aux_isolated") == [239, 234, 239, 239]
    assert per_topic(counts, "aux_spread") == [102, 100, 102, 102]
    assert per_topic(counts, "core_robust") == [1842, 1806, 1842, 1842]
    assert sum(counts.values()) == 16021  # "num_total_steps: 16021"; the baseline ran 16544
    assert notes["before_equal_compute"] == {"core_only_mode": 7571, "module_mode": 8973, "unlabeled": 0}
    assert notes["equal_compute"]["lambda"] == pytest.approx(0.9684, abs=1e-4)


def test_counts_match_the_official_log_of_the_earlier_run():
    """results/stories/seed_1/20260614233133475656/training.log, routed_01 (p_cr 0.2, p_as 0.5)."""
    counts, _ = official_counts(p_cr=0.2, p_as=0.5)
    assert counts[("core_plain", None)] == 11922
    assert per_topic(counts, "aux_isolated") == [173, 170, 173, 173]
    assert per_topic(counts, "aux_spread") == [173, 170, 173, 173]
    assert per_topic(counts, "core_robust") == [749, 734, 749, 749]
    assert sum(counts.values()) == 16281


def test_without_equal_compute_every_batch_is_used_once():
    """One pass over the data. The official arithmetic rounds each module's share separately, so
    a total can come out a batch or two away from the pool size; we keep that arithmetic because
    it is what the tests above verify against the official logs."""
    counts, _ = official_counts(p_cr=0.5, p_as=0.3, equal_compute=False)
    core_batches = counts[("core_plain", None)] + sum(per_topic(counts, "core_robust"))
    assert core_batches == pytest.approx(OFFICIAL_POOL_SIZES[CORE], abs=2)
    for topic in AUX_TOPICS:
        aux_batches = counts[("aux_isolated", topic)] + counts[("aux_spread", topic)]
        assert aux_batches == pytest.approx(OFFICIAL_POOL_SIZES[topic], abs=1)
    assert sum(counts.values()) == pytest.approx(sum(OFFICIAL_POOL_SIZES.values()), abs=2)


def test_routing_probabilities_set_the_proportions():
    counts, _ = official_counts(p_cr=0.5, p_as=0.3, equal_compute=False)
    robust = sum(per_topic(counts, "core_robust"))
    assert robust / OFFICIAL_POOL_SIZES[CORE] == pytest.approx(0.5, abs=1e-3)
    for topic in AUX_TOPICS:
        spread, isolated = counts[("aux_spread", topic)], counts[("aux_isolated", topic)]
        assert spread / (spread + isolated) == pytest.approx(0.3, abs=5e-3)

    none, _ = official_counts(p_cr=0.0, p_as=0.0, equal_compute=False)
    assert sum(per_topic(none, "core_robust")) == 0 and sum(per_topic(none, "aux_spread")) == 0
    everything, _ = official_counts(p_cr=1.0, p_as=1.0, equal_compute=False)
    assert everything[("core_plain", None)] == 0 and sum(per_topic(everything, "aux_isolated")) == 0


def test_equal_compute_lengthens_training_when_gram_is_cheaper_per_batch():
    """With the paper's core width of 1856, core + one module costs what the baseline costs, and a
    core-only batch costs less, so matching total compute means training *longer* (lambda > 1)."""
    narrow_core = {CORE: 24_683_520, **{topic: 1_578_496 for topic in AUX_TOPICS}}
    counts, notes = gram_rule_counts(
        OFFICIAL_POOL_SIZES, AUX_TOPICS, RoutingConfig(p_cr=0.5, p_as=0.3), narrow_core, OFFICIAL_BASELINE_PARAMS
    )
    assert notes["equal_compute"]["lambda"] == pytest.approx(1.028, abs=2e-3)
    assert sum(counts.values()) > sum(OFFICIAL_POOL_SIZES.values())


def test_unlabeled_batches_are_kept_whole():
    pools = {**OFFICIAL_POOL_SIZES, UNLABELED: 4000}
    counts, _ = gram_rule_counts(
        pools, AUX_TOPICS, RoutingConfig(equal_compute=False), OFFICIAL_GROUP_PARAMS, OFFICIAL_BASELINE_PARAMS
    )
    assert counts[("unlabeled", None)] == 4000


def test_oversampling_auxiliary_data_takes_batches_from_core():
    plain, _ = official_counts(p_af=1.0, equal_compute=False)
    doubled, _ = official_counts(p_af=2.0, equal_compute=False)
    aux = lambda c: sum(per_topic(c, "aux_isolated")) + sum(per_topic(c, "aux_spread"))  # noqa: E731
    core = lambda c: c[("core_plain", None)] + sum(per_topic(c, "core_robust"))  # noqa: E731
    assert aux(doubled) == pytest.approx(2 * aux(plain), abs=4)
    assert core(doubled) == pytest.approx(core(plain) - aux(plain), abs=4)
    assert sum(doubled.values()) == pytest.approx(sum(plain.values()), abs=4)
    # The core is still updated on the same number of auxiliary batches as without over-sampling.
    assert sum(per_topic(doubled, "aux_spread")) == pytest.approx(sum(per_topic(plain, "aux_spread")), abs=4)


def gram_plan(accumulation_steps=1, seed=0, pools=OFFICIAL_POOL_SIZES, **routing):
    return build_gram_plan(
        pools, AUX_TOPICS, RoutingConfig(**routing), OFFICIAL_GROUP_PARAMS, OFFICIAL_BASELINE_PARAMS,
        accumulation_steps, np.random.default_rng(seed),
    )


def test_each_rule_activates_and_updates_the_right_groups():
    plan = gram_plan(pools={**OFFICIAL_POOL_SIZES, UNLABELED: 500})
    everything = (CORE, *AUX_TOPICS)
    seen = set()
    for rule in plan.rules:
        seen.add(rule.name)
        if rule.name == "core_plain":
            assert (rule.pool, rule.active, rule.update) == (CORE, (CORE,), (CORE,))
        elif rule.name == "core_robust":
            assert rule.pool == CORE and rule.active == rule.update == (CORE, rule.module)
        elif rule.name == "aux_isolated":
            assert rule.pool == rule.module and rule.active == (CORE, rule.module) and rule.update == (rule.module,)
        elif rule.name == "aux_spread":
            assert rule.pool == rule.module and rule.active == rule.update == (CORE, rule.module)
        elif rule.name == "unlabeled":
            assert (rule.pool, rule.active, rule.update) == (UNLABELED, everything, everything)
        assert set(rule.update) <= set(rule.active)
        assert CORE in rule.active
    assert seen == {"core_plain", "core_robust", "aux_isolated", "aux_spread", "unlabeled"}


def test_schedule_is_a_shuffle_of_the_counts():
    plan = gram_plan()
    assert plan.num_steps == 16021 == len(plan.schedule)
    assert np.bincount(plan.schedule, minlength=len(plan.rules)).tolist() == plan.counts
    assert (np.diff(plan.schedule) != 0).mean() > 0.5, "the rules are interleaved, not run one after another"
    assert (gram_plan().schedule == plan.schedule).all(), "same seed, same order"
    assert (gram_plan(seed=1).schedule != plan.schedule).any()


def test_accumulation_keeps_whole_steps_and_mixes_rules_within_a_step():
    plan = gram_plan(accumulation_steps=8)
    assert len(plan.schedule) % 8 == 0 and plan.num_steps == 16021 // 8
    windows = plan.schedule.reshape(-1, 8)
    mixed = (windows != windows[:, :1]).any(axis=1).mean()
    assert mixed > 0.9, "heterogeneous accumulation: one optimizer step mixes routing decisions"


def test_dense_plan_uses_each_batch_of_the_chosen_pools_once():
    pools = {**OFFICIAL_POOL_SIZES, UNLABELED: 700}
    baseline = build_dense_plan(pools, [CORE, *AUX_TOPICS, UNLABELED], 1, np.random.default_rng(0))
    assert baseline.num_steps == sum(pools.values())
    # Core-only filtering: data given an auxiliary label is removed; unlabeled data is kept.
    filtering = build_dense_plan(pools, [CORE, UNLABELED], 1, np.random.default_rng(0))
    assert filtering.num_steps == 15143 + 700
    assert {rule.pool for rule in filtering.rules} == {CORE, UNLABELED}
    assert all(rule.active == rule.update == (CORE,) for rule in filtering.rules)
    # The official filtering run on the same data: 15143 steps.
    assert build_dense_plan(OFFICIAL_POOL_SIZES, [CORE, UNLABELED], 1, np.random.default_rng(0)).num_steps == 15143
