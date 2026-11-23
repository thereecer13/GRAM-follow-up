"""Checks against the real Simple Stories token store.

These need data/simplestories (about 1.2 GB, made by scripts/prepare_data.py) and are
skipped when it is not there. They tie the whole chain together: real data -> pools ->
routing plan -> exactly the step counts printed in the official training logs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from gram.config import AUX_TOPICS, CORE, load_config
from gram.data import TokenStore, build_probe_batches, build_validation_sets
from gram.model import build_model
from gram.train import set_up_training

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "simplestories"

pytestmark = pytest.mark.skipif(not (STORE / "store.json").exists(), reason="the real token store has not been prepared")


@pytest.fixture(scope="module")
def store():
    return TokenStore(STORE)


def setup_for(store, set_name, run_name):
    config = load_config(ROOT / "configs" / set_name / f"{run_name}.yaml")
    config.data.store_dir = str(STORE)
    model = build_model(config.method, config.model, config.data.aux_topics)
    return config, set_up_training(config, store, model)


def test_the_store_is_the_official_data(store):
    official = json.loads((ROOT / "configs" / "official_token_counts.json").read_text())
    assert store.meta["matches_official_token_counts"] is True
    assert len(store.topics) == 48 and sorted(store.topics)[:4] == list(AUX_TOPICS)
    assert sum(store.meta["train_tokens"].values()) == official["total_train_tokens"] == 547_853_673
    assert sum(store.meta["held_out_tokens"].values()) == official["total_val_tokens"] == 60_763_919
    assert store.meta["vocab_size"] == 4096 and store.eos_token_id == 1
    assert int(np.asarray(store.tokens[:1_000_000]).max()) < 4096


def test_code_exact_plans_match_the_official_training_logs(store):
    """Step counts from results/stories/seed_1/*/training.log in the official repository."""
    _, baseline = setup_for(store, "code_exact", "baseline_seed1")
    assert baseline.total_steps == 16544  # "num_total_steps: 16544"
    pools = {pool: baseline.data.pool_num_batches(pool, 128, "single_topic") for pool in baseline.data.pool_names()}
    assert pools == {CORE: 15143, AUX_TOPICS[0]: 352, AUX_TOPICS[1]: 345, AUX_TOPICS[2]: 352, AUX_TOPICS[3]: 352}

    _, filtering = setup_for(store, "code_exact", "filtering_seed1")
    assert filtering.total_steps == 15143  # "Batch [core] count: 15143 / num_total_steps: 15143"
    assert {rule.pool for rule in filtering.plan.rules} == {CORE}

    _, gram = setup_for(store, "code_exact", "gram_seed1")
    assert gram.total_steps == 16021  # "num_total_steps: 16021"
    counts = {(rule.name, rule.module): count for rule, count in zip(gram.plan.rules, gram.plan.counts)}
    assert counts[("core_plain", None)] == 7332
    assert [counts[("aux_isolated", topic)] for topic in AUX_TOPICS] == [239, 234, 239, 239]
    assert [counts[("aux_spread", topic)] for topic in AUX_TOPICS] == [102, 100, 102, 102]
    assert [counts[("core_robust", topic)] for topic in AUX_TOPICS] == [1842, 1806, 1842, 1842]
    assert gram.plan.notes["equal_compute"]["lambda"] == pytest.approx(0.9684, abs=1e-4)


def test_paper_exact_plans_use_all_data_and_the_same_number_of_steps(store):
    _, baseline = setup_for(store, "paper_exact", "baseline_seed1")
    _, gram = setup_for(store, "paper_exact", "gram_seed1")
    assert baseline.total_steps == 16697  # no per-topic cap: every full batch of every topic
    # "The same ... number of training steps are used across all baselines and GRAM variants".
    assert abs(gram.total_steps - baseline.total_steps) <= 2
    assert "equal_compute" not in gram.plan.notes
    config, _ = setup_for(store, "paper_exact", "gram_seed1")
    model = build_model("gram", config.model, config.data.aux_topics).num_parameters()
    assert model[CORE] + model[AUX_TOPICS[0]] == 26_262_016  # core + one module is the size of the baseline


def test_every_training_sequence_is_used_at_most_once_per_epoch(store):
    _, baseline = setup_for(store, "code_exact", "baseline_seed1")
    rows = baseline.batch_rows.ravel()
    assert len(np.unique(rows)) == len(rows) == 16544 * 128
    assert rows.max() < len(baseline.data.sequences)


def test_validation_and_probe_data(store):
    validation = build_validation_sets(store, AUX_TOPICS, 256)
    assert list(validation) == [CORE, *AUX_TOPICS]
    assert len(validation[CORE]) > 200_000 and all(4000 < len(validation[topic]) < 5500 for topic in AUX_TOPICS)
    probes = build_probe_batches(store, AUX_TOPICS, 256, 128)
    assert all(a.shape == b.shape == (128, 257) for a, b in probes.values())
