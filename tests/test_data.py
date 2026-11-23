"""The data pipeline: sequences, label assignment, pools, batches and the split manifest."""

import numpy as np
import pytest
from helpers import AUX, CONTEXT, CORE_TOPICS, EOS, TOPICS

from gram.config import CORE, UNLABELED, DataConfig, LabelConfig
from gram.data import (
    assign_batches_to_schedule,
    assign_given_labels,
    build_elicitation_sequences,
    build_probe_batches,
    build_training_data,
    build_validation_sets,
    concatenate_stories,
    cut_into_sequences,
    label_names,
    lowest_keys_within_groups,
    make_pool_batches,
    rng_for,
)

NAMES = label_names(AUX)


def labels_for(store, seed=1, **label_settings):
    stories = store.stories_in("train")
    true, given = assign_given_labels(store, stories, AUX, LabelConfig(**label_settings), seed)
    return stories, true, given


def training_data(store, seed=1, cap=False, **label_settings):
    data_config = DataConfig(aux_topics=AUX, cap_topics_at_class_mean=cap)
    return build_training_data(store, data_config, LabelConfig(**label_settings), CONTEXT, seed)


# ----------------------------------------------------------------- sequences


def test_sequences_overlap_by_one_token_and_cover_the_stream():
    stream = np.arange(100, dtype=np.uint16)
    sequences = cut_into_sequences(stream, context_length=16)
    assert sequences.shape == (6, 17)  # (100 - 1) // 16
    for k, sequence in enumerate(sequences):
        assert (sequence == stream[k * 16 : k * 16 + 17]).all()
    # Inputs are sequence[:-1] and targets are sequence[1:]: every stream position from 1 to 96
    # is a target exactly once.
    targets = np.concatenate([sequence[1:] for sequence in sequences])
    assert (targets == np.arange(1, 97)).all()


def test_a_stream_too_short_for_one_sequence_gives_none():
    assert cut_into_sequences(np.arange(16, dtype=np.uint16), 16).shape == (0, 17)
    assert cut_into_sequences(np.arange(17, dtype=np.uint16), 16).shape == (1, 17)


def test_concatenate_stories_respects_order(store):
    stories = store.stories_in("train", "echo")[[5, 2, 9]]
    expected = np.concatenate([store.tokens[store.token_start[s] : store.token_start[s] + store.n_tokens[s]] for s in stories])
    assert (concatenate_stories(store, stories) == expected).all()
    assert concatenate_stories(store, stories)[-1] == EOS


# -------------------------------------------------------------------- labels


def test_stratified_selection_is_exact_and_nested():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 5, size=1000)
    keys = rng.random(1000)
    small = lowest_keys_within_groups(keys, groups, 0.1)
    large = lowest_keys_within_groups(keys, groups, 0.5)
    for group in range(5):
        size = (groups == group).sum()
        assert small[groups == group].sum() == int(np.floor(0.1 * size + 0.5))
        assert large[groups == group].sum() == int(np.floor(0.5 * size + 0.5))
    assert (large[small]).all(), "the smaller selection must be contained in the larger one"
    assert lowest_keys_within_groups(keys, groups, 1.0).all()
    assert not lowest_keys_within_groups(keys, groups, 0.0).any()


def test_clean_labels_equal_true_labels(store):
    stories, true, given = labels_for(store)
    assert (true == given).all()
    for index, topic in enumerate(TOPICS):
        expected = NAMES.index(topic) if topic in AUX else NAMES.index(CORE)
        assert (true[store.topic[stories] == index] == expected).all()


@pytest.mark.parametrize("fraction", [0.5, 0.1, 0.01])
def test_labeled_fraction_is_stratified_by_topic(store, fraction):
    stories, true, given = labels_for(store, labeled_fraction=fraction)
    unlabeled = given == NAMES.index(UNLABELED)
    for index in range(len(TOPICS)):
        of_topic = store.topic[stories] == index
        expected_labeled = int(np.floor(fraction * of_topic.sum() + 0.5))
        assert (~unlabeled[of_topic]).sum() == expected_labeled
    assert (given[~unlabeled] == true[~unlabeled]).all(), "a story that keeps its label keeps the true one"


def test_labeled_sets_are_nested_across_fractions_and_fixed_by_the_seed(store):
    unlabeled = NAMES.index(UNLABELED)
    labeled = {f: labels_for(store, labeled_fraction=f)[2] != unlabeled for f in (0.01, 0.1, 0.5)}
    assert labeled[0.1][labeled[0.01]].all() and labeled[0.5][labeled[0.1]].all()

    again = labels_for(store, labeled_fraction=0.5)[2] != unlabeled
    assert (again == labeled[0.5]).all(), "same seed, same split"
    other_seed = labels_for(store, seed=2, labeled_fraction=0.5)[2] != unlabeled
    assert (other_seed != labeled[0.5]).any(), "a different seed gives a different split"


@pytest.mark.parametrize("rate", [0.05, 0.2, 0.5])
def test_false_negatives_relabel_auxiliary_stories_as_core(store, rate):
    stories, true, given = labels_for(store, noise="false_negative", noise_rate=rate)
    changed = given != true
    assert (true[changed] > 0).all() and (given[changed] == NAMES.index(CORE)).all()
    for topic in AUX:
        of_topic = true == NAMES.index(topic)
        assert changed[of_topic].sum() == int(np.floor(rate * of_topic.sum() + 0.5))
    assert not changed[true == 0].any(), "core stories are untouched"


def test_missed_unlabeled_hits_exactly_the_false_negative_stories(store):
    _, true, as_core = labels_for(store, noise="false_negative", noise_rate=0.2)
    _, _, as_unlabeled = labels_for(store, noise="missed_unlabeled", noise_rate=0.2)
    missed_core = as_core != true
    missed_unlabeled = as_unlabeled != true
    assert (missed_core == missed_unlabeled).all(), "the 20% miss comparison must use the same stories"
    assert (as_unlabeled[missed_unlabeled] == NAMES.index(UNLABELED)).all()
    assert missed_core.sum() > 0


def test_wrong_label_sets_are_nested_across_rates(store):
    _, true, low = labels_for(store, noise="false_negative", noise_rate=0.05)
    _, _, high = labels_for(store, noise="false_negative", noise_rate=0.5)
    assert (high != true)[low != true].all()


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.2])
def test_false_positives_give_core_stories_an_auxiliary_label(store, rate):
    stories, true, given = labels_for(store, noise="false_positive", noise_rate=rate)
    changed = given != true
    assert (true[changed] == 0).all()
    assert set(given[changed]) <= {NAMES.index(topic) for topic in AUX}
    for topic in CORE_TOPICS:
        of_topic = store.topic[stories] == TOPICS.index(topic)
        assert changed[of_topic].sum() == int(np.floor(rate * of_topic.sum() + 0.5))
    if rate == 0.2:
        assert set(given[changed]) == {NAMES.index(topic) for topic in AUX}, "every module receives some"


@pytest.mark.parametrize("rate", [0.1, 0.3])
def test_wrong_module_moves_auxiliary_stories_to_a_different_module(store, rate):
    stories, true, given = labels_for(store, noise="wrong_module", noise_rate=rate)
    changed = given != true
    aux_codes = {NAMES.index(topic) for topic in AUX}
    assert (true[changed] > 0).all() and set(given[changed]) <= aux_codes
    for topic in AUX:
        of_topic = true == NAMES.index(topic)
        assert changed[of_topic].sum() == int(np.floor(rate * of_topic.sum() + 0.5))
        # The selected stories really do move (given != true), and to all three other modules.
        assert set(given[of_topic & changed]) == aux_codes - {NAMES.index(topic)}


# --------------------------------------------------- sequences, cells, manifest


def test_every_sequence_has_one_topic_and_one_given_label(store):
    data = training_data(store, labeled_fraction=0.5)
    rows_seen = np.zeros(len(data.sequences), dtype=int)
    for cell in data.cells:
        rows_seen[cell.first_sequence : cell.first_sequence + cell.num_sequences] += 1
    assert (rows_seen == 1).all(), "cells must tile the sequence table exactly"
    assert len({(cell.given_label, cell.topic) for cell in data.cells}) == len(data.cells)


def test_cell_sequences_are_the_cells_stories_in_order(store):
    data = training_data(store, noise="false_negative", noise_rate=0.2)
    manifest = data.manifest
    position_of = {int(story_id): position for position, story_id in enumerate(store.story_id)}
    for cell in data.cells:
        members = manifest[(manifest.given_label == cell.given_label) & (manifest.true_topic == cell.topic)]
        assert len(members) == cell.num_stories
        stream = concatenate_stories(store, np.array([position_of[int(s)] for s in members.story_id]))
        expected = cut_into_sequences(stream, CONTEXT)
        actual = data.sequences[cell.first_sequence : cell.first_sequence + cell.num_sequences]
        assert (actual == expected).all()


def test_manifest_lists_every_training_story_once(store):
    data = training_data(store, labeled_fraction=0.1)
    manifest = data.manifest
    train_ids = store.story_id[store.stories_in("train")]
    assert sorted(manifest.story_id) == sorted(train_ids)
    assert set(manifest.columns) >= {"story_id", "true_topic", "true_label", "given_label", "seed"}
    assert (manifest.seed == 1).all()
    assert (manifest.given_label == UNLABELED).mean() == pytest.approx(0.9, abs=0.01)


def test_manifest_points_at_the_sequences_that_contain_each_story(store):
    """first_sequence..last_sequence are exactly the sequences in which a story's tokens appear.

    Checked against a deliberately naive recomputation: walk along each cell's stream story by
    story, and for every sequence ask whether its token range overlaps the story's.
    """
    position_of = {int(story_id): position for position, story_id in enumerate(store.story_id)}
    stories_in_tail = boundary_cases = cut_short = 0

    # Several label settings, so that the streams are cut in many different places and the edge
    # cases counted at the bottom are all exercised.
    settings = [{}, {"labeled_fraction": 0.5}, {"labeled_fraction": 0.1}, {"noise": "wrong_module", "noise_rate": 0.3}]
    cells = [(data, cell) for data in (training_data(store, **kwargs) for kwargs in settings) for cell in data.cells]
    for data, cell in cells:
        manifest = data.manifest
        members = manifest[(manifest.given_label == cell.given_label) & (manifest.true_topic == cell.topic)]
        cursor = 0
        for row in members.itertuples():
            length = int(store.n_tokens[position_of[int(row.story_id)]])
            begin, finish = cursor, cursor + length - 1  # stream positions the story occupies
            cursor += length
            # Sequence k holds stream positions k*T .. k*T + T (T inputs plus the final target).
            holding = [
                k for k in range(cell.num_sequences)
                if k * CONTEXT <= finish and k * CONTEXT + CONTEXT >= begin
            ]
            if not holding:
                assert row.first_sequence == row.last_sequence == -1
                stories_in_tail += 1
                continue
            assert row.first_sequence == cell.first_sequence + holding[0]
            assert row.last_sequence == cell.first_sequence + holding[-1]
            # The story's first token really is in its first sequence, and its last token in its last.
            first_row = data.sequences[row.first_sequence]
            last_row = data.sequences[row.last_sequence]
            story = store.tokens[store.token_start[position_of[int(row.story_id)]] :][:length]
            assert first_row[begin - holding[0] * CONTEXT] == story[0]
            if finish <= holding[-1] * CONTEXT + CONTEXT:
                assert last_row[finish - holding[-1] * CONTEXT] == story[-1] == EOS
            else:
                # The story runs past the last sequence of its cell: its end is in the unused tail.
                assert holding[-1] == cell.num_sequences - 1
                cut_short += 1
            # A story that starts exactly on a boundary is the last *target* of the sequence before.
            boundary_cases += begin > 0 and begin % CONTEXT == 0

    assert stories_in_tail > 0, "the test should include stories that fall in an unused stream tail"
    assert boundary_cases > 0, "the test should include stories that start exactly on a sequence boundary"
    assert cut_short > 0, "the test should include stories whose end falls in an unused stream tail"


def test_train_fraction_keeps_a_prefix_of_every_topic(store):
    uncapped = dict(aux_topics=AUX, cap_topics_at_class_mean=False)
    full = build_training_data(store, DataConfig(**uncapped), LabelConfig(), CONTEXT, seed=1)
    part = build_training_data(store, DataConfig(train_fraction=0.25, **uncapped), LabelConfig(), CONTEXT, seed=1)
    assert set(part.manifest.story_id) < set(full.manifest.story_id)
    counts = part.manifest.true_topic.value_counts()
    assert (counts == 30).all()  # a quarter of the 120 training stories of each topic


def test_topic_cap_trims_large_topics_to_the_mean_of_their_class(store):
    capped, uncapped = training_data(store, cap=True), training_data(store, cap=False)
    assert set(capped.manifest.story_id) < set(uncapped.manifest.story_id)

    lengths = dict(zip(store.story_id, store.n_tokens))
    tokens = lambda manifest: manifest.assign(n=[lengths[i] for i in manifest.story_id]).groupby("true_topic", observed=True).n.sum()  # noqa: E731
    before, after = tokens(uncapped.manifest), tokens(capped.manifest)
    for group in (list(AUX), list(CORE_TOPICS)):
        budget = round(before[group].sum() / len(group))
        for topic in group:
            if before[topic] <= budget:
                assert after[topic] == before[topic], "a topic at or below the mean is used in full"
            else:
                assert budget - 30 < after[topic] <= budget, "a larger topic is cut back to the budget"
        assert (before[group] > budget).any(), "the test data should contain a topic above the mean"


def test_topic_cap_drops_the_last_stories_of_a_topic(store):
    capped, uncapped = training_data(store, cap=True), training_data(store, cap=False)
    kept = set(capped.manifest.story_id)
    for topic in TOPICS:
        in_order = uncapped.manifest[uncapped.manifest.true_topic == topic].story_id.tolist()
        flags = [story in kept for story in in_order]
        assert flags == sorted(flags, reverse=True), "kept stories form a prefix of the topic in canonical order"


# -------------------------------------------------------------- pools, batches


def test_pools_follow_the_given_label(store):
    data = training_data(store, noise="false_negative", noise_rate=0.2)
    core_pool_topics = {cell.topic for cell in data.pool(CORE)}
    assert core_pool_topics == set(TOPICS), "mislabeled auxiliary stories are in the core pool"
    for topic in AUX:
        assert {cell.topic for cell in data.pool(topic)} == {topic}
    assert data.pool_names() == [CORE, *AUX]

    unlabeled = training_data(store, labeled_fraction=0.5)
    assert unlabeled.pool_names() == [CORE, *AUX, UNLABELED]
    assert {cell.topic for cell in unlabeled.pool(UNLABELED)} == set(TOPICS)


def test_single_topic_batches_never_mix_topics(store):
    data = training_data(store)
    batches = make_pool_batches(data.pool(CORE), 4, "single_topic", rng_for(1, "t"))
    cell_of_row = np.empty(len(data.sequences), dtype=int)
    for index, cell in enumerate(data.cells):
        cell_of_row[cell.first_sequence : cell.first_sequence + cell.num_sequences] = index
    assert (cell_of_row[batches] == cell_of_row[batches][:, :1]).all()
    assert len(np.unique(batches)) == batches.size, "no sequence is used twice in one pass"
    assert len({data.cells[i].topic for i in np.unique(cell_of_row[batches])}) == len(CORE_TOPICS)


def test_mixed_batches_mix_topics_and_use_the_whole_pool(store):
    data = training_data(store)
    batches = make_pool_batches(data.pool(CORE), 4, "mixed", rng_for(1, "t"))
    cell_of_row = np.empty(len(data.sequences), dtype=int)
    for index, cell in enumerate(data.cells):
        cell_of_row[cell.first_sequence : cell.first_sequence + cell.num_sequences] = index
    mixed = (cell_of_row[batches] != cell_of_row[batches][:, :1]).any(axis=1)
    assert mixed.mean() > 0.5
    assert len(np.unique(batches)) == batches.size
    assert batches.size == data.pool_num_sequences(CORE) // 4 * 4, "only an incomplete last batch is left out"


def test_batches_only_contain_sequences_of_their_pool(store):
    data = training_data(store, labeled_fraction=0.5)
    schedule = [CORE] * 10 + [AUX[0]] * 5 + [UNLABELED] * 10
    batches = assign_batches_to_schedule(schedule, data, 4, "mixed", seed=1)
    given_of_row = np.empty(len(data.sequences), dtype=object)
    for cell in data.cells:
        given_of_row[cell.first_sequence : cell.first_sequence + cell.num_sequences] = cell.given_label
    for pool, batch in zip(schedule, batches):
        assert (given_of_row[batch] == pool).all()


def test_pool_batch_counts_agree_with_the_batches_actually_made(store):
    data = training_data(store, labeled_fraction=0.5)
    for pool in data.pool_names():
        for assembly in ("single_topic", "mixed"):
            made = make_pool_batches(data.pool(pool), 4, assembly, rng_for(1, "t"))
            assert data.pool_num_batches(pool, 4, assembly) == len(made)


def test_schedule_longer_than_a_pool_starts_a_fresh_pass(store):
    data = training_data(store)
    one_pass = data.pool_num_sequences(AUX[0]) // 4
    batches = assign_batches_to_schedule([AUX[0]] * (one_pass + 3), data, 4, "mixed", seed=1)
    first_pass, second_pass = batches[:one_pass], batches[one_pass:]
    assert len(np.unique(first_pass)) == first_pass.size
    assert np.isin(second_pass, first_pass).any(), "the second pass reuses the pool"
    again = assign_batches_to_schedule([AUX[0]] * (one_pass + 3), data, 4, "mixed", seed=1)
    assert (again == batches).all(), "the assignment is a pure function of the seed"


# ------------------------------------------------ validation, probes, elicitation


def story_token_sets(store, split):
    """The set of stories of a split, each identified by its exact tokens."""
    return {
        store.tokens[store.token_start[s] : store.token_start[s] + store.n_tokens[s]].tobytes()
        for s in store.stories_in(split)
    }


def test_train_validation_and_probe_stories_are_disjoint(store):
    splits = {name: set(store.stories_in(name)) for name in ("train", "val", "probe")}
    assert not splits["train"] & splits["val"] and not splits["train"] & splits["probe"] and not splits["val"] & splits["probe"]
    assert sum(len(members) for members in splits.values()) == len(store)
    for topic in TOPICS:
        assert len(store.stories_in("probe", topic)) > 0, "every topic contributes probe stories"


def test_validation_sets_cover_core_and_each_auxiliary_topic(store):
    sets = build_validation_sets(store, AUX, CONTEXT)
    assert list(sets) == [CORE, *AUX]
    per_topic = [len(cut_into_sequences(concatenate_stories(store, store.stories_in("val", t)), CONTEXT)) for t in CORE_TOPICS]
    assert len(sets[CORE]) == sum(per_topic)
    for topic in AUX:
        expected = cut_into_sequences(concatenate_stories(store, store.stories_in("val", topic)), CONTEXT)
        assert (sets[topic] == expected).all()


def test_probe_batches_are_fixed_and_come_from_the_probe_split(store):
    batches = build_probe_batches(store, AUX, CONTEXT, batch_size=4)
    assert list(batches) == [CORE, *AUX]
    probe_tokens = np.concatenate([concatenate_stories(store, store.stories_in("probe", t)) for t in TOPICS])
    probe_windows = {w.tobytes() for w in np.lib.stride_tricks.sliding_window_view(probe_tokens, CONTEXT + 1)}
    for name, (first, second) in batches.items():
        assert first.shape == second.shape == (4, CONTEXT + 1)
        assert not (first[:, None] == second[None]).all(-1).any(), "A and B share no sequence"
        for sequence in np.concatenate([first, second]):
            assert sequence.tobytes() in probe_windows
    again = build_probe_batches(store, AUX, CONTEXT, batch_size=4)
    assert all((batches[name][0] == again[name][0]).all() for name in batches)


def test_elicitation_sequences_are_training_sequences_of_the_topic(store):
    topic = AUX[1]
    sequences = build_elicitation_sequences(store, topic, 8, CONTEXT, seed=1)
    all_of_topic = cut_into_sequences(concatenate_stories(store, store.stories_in("train", topic)), CONTEXT)
    known = {sequence.tobytes() for sequence in all_of_topic}
    assert sequences.shape == (8, CONTEXT + 1)
    assert all(sequence.tobytes() in known for sequence in sequences)
    assert len({sequence.tobytes() for sequence in sequences}) == 8
    assert (build_elicitation_sequences(store, topic, 8, CONTEXT, seed=1) == sequences).all()
    assert (build_elicitation_sequences(store, topic, 8, CONTEXT, seed=2) != sequences).any()
