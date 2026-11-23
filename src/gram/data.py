"""From stories to batches.

The pipeline has three layers, each a plain function of the one before it:

1. **The token store** is written once by ``scripts/prepare_data.py``. It holds every
   story, tokenised, together with its topic and its split (``train``, ``val`` or
   ``probe``). It never changes and is shared by every run.

2. **Labels** are decided per run. Every training story has a *true* label (``core``, or
   the name of its auxiliary topic) and a *given* label, which is what the training
   method is allowed to see. With perfect, complete labels the two are the same. The
   labeled-fraction experiments replace some given labels by ``unlabeled``; the
   wrong-label experiments replace some by a different label.

3. **Sequences and batches.** Stories that share a given label and a true topic are
   concatenated into one long token stream, the stream is cut into fixed-length training
   sequences, and sequences are grouped into batches. All stories with the same given
   label form a *pool*; a batch is always drawn from a single pool, because the routing
   rule is chosen per batch from the given label.

Everything random is derived from the run seed, so a run can be rebuilt exactly, and two
methods run with the same seed see the identical label split and the identical batches.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gram.config import CORE, UNLABELED, DataConfig, LabelConfig

TOKEN_DTYPE = np.uint16
SPLITS = ("train", "val", "probe")


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    """An independent random stream for one named purpose, derived from the run seed.

    Giving every use of randomness its own stream means that changing how one thing is
    sampled (or adding a new random choice) cannot silently change another.
    """
    return np.random.default_rng([seed, zlib.crc32(purpose.encode())])


# --------------------------------------------------------------------------- #
# 1. The token store
# --------------------------------------------------------------------------- #


class TokenStore:
    """Every story, tokenised, with its topic and split. Read-only.

    Stories are kept in a fixed *canonical order*: the training split in the shuffled
    order produced by the train/validation split, then the validation split. The arrays
    below all have one entry per story, in that order, and "a story" elsewhere in this
    module means a position in that order.
    """

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        if not (directory / "store.json").exists():
            raise FileNotFoundError(
                f"No token store in {directory}. Create it with: python scripts/prepare_data.py"
            )
        self.directory = directory
        self.meta: dict = json.loads((directory / "store.json").read_text())
        self.topics: tuple[str, ...] = tuple(self.meta["topics"])
        self.eos_token_id: int = int(self.meta["eos_token_id"])

        self.tokens = np.memmap(directory / "tokens.bin", dtype=TOKEN_DTYPE, mode="r")

        stories = pd.read_parquet(directory / "stories.parquet")
        self.story_id = stories["story_id"].to_numpy()  # row index in the Hugging Face dataset
        self.topic = pd.Categorical(stories["topic"], categories=self.topics).codes.astype(np.int16)
        self.split = pd.Categorical(stories["split"], categories=SPLITS).codes.astype(np.int8)
        self.token_start = stories["token_start"].to_numpy()  # offset of the story in tokens.bin
        self.n_tokens = stories["n_tokens"].to_numpy()  # length, including the end-of-story token

        assert (self.topic >= 0).all() and (self.split >= 0).all(), "unknown topic or split in the store"

    def __len__(self) -> int:
        return len(self.story_id)

    def stories_in(self, split: str, topic: str | None = None) -> np.ndarray:
        """Stories of a split (and optionally one topic), in canonical order."""
        mask = self.split == SPLITS.index(split)
        if topic is not None:
            mask &= self.topic == self.topics.index(topic)
        return np.flatnonzero(mask)


def write_token_store(
    directory: str | Path,
    tokens: np.ndarray,
    story_id: np.ndarray,
    topic: Sequence[str],
    split: Sequence[str],
    n_tokens: np.ndarray,
    meta: dict,
) -> None:
    """Write a token store. ``tokens`` is every story's tokens, concatenated in order."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    n_tokens = np.asarray(n_tokens, dtype=np.int32)
    assert int(n_tokens.sum()) == len(tokens), "token count does not match the story lengths"
    token_start = np.cumsum(n_tokens, dtype=np.int64) - n_tokens

    np.asarray(tokens, dtype=TOKEN_DTYPE).tofile(directory / "tokens.bin")
    pd.DataFrame(
        {
            "story_id": np.asarray(story_id, dtype=np.int32),
            "topic": pd.Categorical(topic),
            "split": pd.Categorical(split, categories=SPLITS),
            "token_start": token_start,
            "n_tokens": n_tokens,
        }
    ).to_parquet(directory / "stories.parquet", compression="zstd", index=False)

    meta = dict(meta)
    meta["topics"] = sorted(set(topic))
    meta["num_stories"] = int(len(n_tokens))
    meta["num_tokens"] = int(len(tokens))
    (directory / "store.json").write_text(json.dumps(meta, indent=2))


def choose_probe_stories(
    topic: np.ndarray, split: np.ndarray, n_tokens: np.ndarray, tokens_per_topic: int
) -> np.ndarray:
    """Pick the validation stories that are set aside as gradient-probe data.

    For every topic, the first validation stories (in canonical order) are taken until
    they add up to at least ``tokens_per_topic`` tokens. These stories are removed from
    the validation split, so the probes never share data with the validation losses,
    and of course they are never trained on. Returns a boolean mask over all stories.
    """
    is_val = split == "val"
    probe = np.zeros(len(topic), dtype=bool)
    for name in np.unique(topic[is_val]):
        candidates = np.flatnonzero(is_val & (topic == name))
        tokens_before = np.cumsum(n_tokens[candidates]) - n_tokens[candidates]
        probe[candidates[tokens_before < tokens_per_topic]] = True
    return probe


def prepare_token_store(
    directory: str | Path,
    dataset_id: str,
    revision: str,
    tokenizer_id: str,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    probe_sequences_per_topic: int = 256,
    context_length: int = 256,
    batch_size: int = 20_000,
) -> dict:
    """Download Simple Stories, split it, tokenise it, and write the token store.

    This follows the official data preparation step for step, so that the result is the
    same data the paper's models saw:

    * only the ``train`` split of the Hugging Face dataset is used;
    * 10% of it is held out with ``train_test_split(test_size=0.1, seed=42)``. The split
      is made *before* dividing by topic, so every topic gets its own held-out stories;
    * topics are lower-cased with spaces turned into dashes;
    * each story is tokenised without special tokens and then terminated with the
      end-of-story token.

    On top of that, a small part of each topic's held-out stories is set aside as
    gradient-probe data (see ``choose_probe_stories``). Returns the store's metadata.
    """
    from datasets import load_dataset  # imported here: only data preparation needs these
    from transformers import AutoTokenizer

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_id, split="train", revision=revision)
    dataset = dataset.select_columns(["topic", "story"])
    dataset = dataset.add_column("story_id", np.arange(len(dataset), dtype=np.int32))
    parts = dataset.train_test_split(test_size=val_fraction, seed=split_seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    eos_token_id = tokenizer.eos_token_id

    story_id, topic, split, n_tokens = [], [], [], []
    with open(directory / "tokens.bin", "wb") as token_file:
        for split_name, part in (("train", parts["train"]), ("val", parts["test"])):
            for batch in part.iter(batch_size=batch_size):
                encoded = tokenizer(batch["story"], add_special_tokens=False)["input_ids"]
                for ids in encoded:
                    ids.append(eos_token_id)
                np.fromiter((token for ids in encoded for token in ids), dtype=TOKEN_DTYPE).tofile(token_file)
                story_id.extend(batch["story_id"])
                topic.extend(str(name).lower().replace(" ", "-") for name in batch["topic"])
                split.extend([split_name] * len(encoded))
                n_tokens.extend(len(ids) for ids in encoded)

    topic = np.array(topic)
    split = np.array(split, dtype=object)
    n_tokens = np.array(n_tokens, dtype=np.int32)

    # One extra token because a sequence of T inputs needs T + 1 tokens.
    probe_tokens = probe_sequences_per_topic * context_length + 1
    split[choose_probe_stories(topic, split, n_tokens, probe_tokens)] = "probe"

    tokens = np.memmap(directory / "tokens.bin", dtype=TOKEN_DTYPE, mode="r")
    assert len(tokens) == int(n_tokens.sum())
    assert int(tokens.max()) < len(tokenizer)

    def tokens_by_topic(mask: np.ndarray) -> dict[str, int]:
        return {name: int(n_tokens[mask & (topic == name)].sum()) for name in sorted(set(topic))}

    is_train = split == "train"
    meta = {
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "tokenizer_id": tokenizer_id,
        "vocab_size": len(tokenizer),
        "eos_token_id": eos_token_id,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "probe_sequences_per_topic": probe_sequences_per_topic,
        "context_length_for_probe_sizing": context_length,
        # "held out" = val + probe: the official 10% split, before the probe stories are set aside.
        "train_tokens": tokens_by_topic(is_train),
        "held_out_tokens": tokens_by_topic(~is_train),
        "val_tokens": tokens_by_topic(split == "val"),
        "probe_tokens": tokens_by_topic(split == "probe"),
    }
    del tokens

    # tokens.bin is already on disk in its final form; write the two small files next to it.
    token_start = np.cumsum(n_tokens, dtype=np.int64) - n_tokens
    pd.DataFrame(
        {
            "story_id": np.asarray(story_id, dtype=np.int32),
            "topic": pd.Categorical(topic),
            "split": pd.Categorical(split, categories=SPLITS),
            "token_start": token_start,
            "n_tokens": n_tokens,
        }
    ).to_parquet(directory / "stories.parquet", compression="zstd", index=False)
    meta["topics"] = sorted(set(topic))
    meta["num_stories"] = int(len(n_tokens))
    meta["num_tokens"] = int(n_tokens.sum())
    (directory / "store.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- #
# Streams and sequences
# --------------------------------------------------------------------------- #


def concatenate_stories(store: TokenStore, stories: np.ndarray) -> np.ndarray:
    """The tokens of the given stories, one after another, as a single stream."""
    starts = store.token_start[stories]
    lengths = store.n_tokens[stories].astype(np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=TOKEN_DTYPE)
    position_in_stream = np.cumsum(lengths) - lengths
    # For every token of the stream, its offset in the store.
    source = np.repeat(starts - position_in_stream, lengths) + np.arange(total)
    return np.asarray(store.tokens[source])


def cut_into_sequences(stream: np.ndarray, context_length: int) -> np.ndarray:
    """Cut a token stream into sequences of ``context_length + 1`` tokens.

    Sequence k is ``stream[k*T : k*T + T + 1]``. The model reads the first T tokens of a
    sequence and is scored on the last T, so consecutive sequences share one token and
    every token of the stream is a prediction target exactly once. A sequence usually
    holds the end of one story and the beginning of the next; the attention mask keeps
    the two apart. Whatever is left over at the end of the stream (fewer than T tokens)
    is not used.
    """
    num_sequences = max(0, (len(stream) - 1) // context_length)
    if num_sequences == 0:
        return np.empty((0, context_length + 1), dtype=TOKEN_DTYPE)
    windows = np.lib.stride_tricks.sliding_window_view(stream, context_length + 1)[::context_length]
    return np.ascontiguousarray(windows[:num_sequences])


def first_stories_per_topic(store: TokenStore, stories: np.ndarray, fraction: float) -> np.ndarray:
    """Keep the first ``fraction`` of each topic's stories (in canonical order)."""
    if fraction >= 1.0:
        return stories
    keep = np.zeros(len(stories), dtype=bool)
    for code in np.unique(store.topic[stories]):
        of_topic = np.flatnonzero(store.topic[stories] == code)
        keep[of_topic[: int(np.ceil(fraction * len(of_topic)))]] = True
    return stories[keep]


def cap_topics_at_class_mean(store: TokenStore, stories: np.ndarray, aux_topics: Sequence[str]) -> np.ndarray:
    """Drop the stories a topic has beyond the mean topic size of its class.

    The official code does this implicitly: it gives each of the 44 core topics a budget
    of (all core tokens) / 44, and each auxiliary topic a budget of (all auxiliary
    tokens) / 4, and reads each topic's stream only up to its budget. Topics smaller than
    the mean are used in full. Here the same budget is applied to whole stories: a topic
    keeps its stories, in canonical order, for as long as they fit within the budget.
    On the real data this gives exactly the official number of batches for every topic.
    """
    topic = store.topic[stories]
    lengths = store.n_tokens[stories].astype(np.int64)
    tokens_per_topic = np.bincount(topic, weights=lengths, minlength=len(store.topics)).astype(np.int64)
    is_aux_topic = np.array([name in aux_topics for name in store.topics])

    keep = np.ones(len(stories), dtype=bool)
    for members in (is_aux_topic, ~is_aux_topic):
        present = members & (tokens_per_topic > 0)
        if not present.any():
            continue
        budget = round(int(tokens_per_topic[present].sum()) / int(present.sum()))
        for code in np.flatnonzero(present):
            of_topic = np.flatnonzero(topic == code)
            keep[of_topic[np.cumsum(lengths[of_topic]) > budget]] = False
    return stories[keep]


# --------------------------------------------------------------------------- #
# 2. Labels
# --------------------------------------------------------------------------- #


def label_names(aux_topics: Sequence[str]) -> tuple[str, ...]:
    """All labels a story can be given, in a fixed order: core, each auxiliary topic, unlabeled."""
    return (CORE, *aux_topics, UNLABELED)


def lowest_keys_within_groups(keys: np.ndarray, groups: np.ndarray, fraction: float) -> np.ndarray:
    """Within each group, select the ``round(fraction * group size)`` members with the smallest keys.

    Selecting by rank of a fixed random key (rather than drawing afresh) has two useful
    consequences: the selection is exactly stratified by group, and it is *nested*: the
    stories selected at a small fraction are a subset of those selected at a larger one.
    """
    order = np.lexsort((keys, groups))
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate([[0], boundaries])
    sizes = np.diff(np.concatenate([starts, [len(groups)]]))
    rank_in_group = np.arange(len(groups)) - np.repeat(starts, sizes)
    quota = np.repeat(np.floor(fraction * sizes + 0.5).astype(np.int64), sizes)

    selected = np.zeros(len(groups), dtype=bool)
    selected[order] = rank_in_group < quota
    return selected


def assign_given_labels(
    store: TokenStore,
    stories: np.ndarray,
    aux_topics: Sequence[str],
    labels: LabelConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide the true and the given label of every training story.

    Returns two integer arrays (codes into ``label_names(aux_topics)``), one entry per
    story in ``stories``.

    The true label is ``core`` unless the story's topic is one of the auxiliary topics.
    The given label starts out equal to the true label and is then changed by, at most,
    one of the following (they are never combined):

    * ``labeled_fraction`` f < 1: within every topic, a fraction 1 - f of the stories
      lose their label and become ``unlabeled``.
    * ``false_negative`` at rate r: within every auxiliary topic, a fraction r of the
      stories are given the label ``core``.
    * ``missed_unlabeled`` at rate r: *the same stories* as ``false_negative`` at rate r,
      but they are given no label (``unlabeled``) instead of the label ``core``.
    * ``false_positive`` at rate r: within every core topic, a fraction r of the stories
      are given an auxiliary label chosen uniformly at random.
    * ``wrong_module`` at rate r: within every auxiliary topic, a fraction r of the
      stories are given one of the *other* auxiliary labels, chosen uniformly at random.

    Which stories are affected depends only on the seed, never on the method, so every
    method trained with the same seed and label settings sees the identical split.
    """
    names = label_names(aux_topics)
    unlabeled = names.index(UNLABELED)
    num_aux = len(aux_topics)

    # True label: 0 for core, 1..num_aux for the auxiliary topics.
    label_of_topic = np.zeros(len(store.topics), dtype=np.int8)
    for index, name in enumerate(aux_topics):
        label_of_topic[store.topics.index(name)] = index + 1
    topic = store.topic[stories]
    true_label = label_of_topic[topic]
    given_label = true_label.copy()

    # One random key per training story of the *whole store*, so that a story's key does
    # not depend on which other stories this particular run happens to use.
    all_train = store.stories_in("train")
    position = np.searchsorted(all_train, stories)
    assert (all_train[position] == stories).all(), "labels are only assigned to training stories"

    def keys(purpose: str) -> np.ndarray:
        return rng_for(seed, purpose).random(len(all_train))[position]

    if labels.labeled_fraction < 1.0:
        is_labeled = lowest_keys_within_groups(keys("labeled"), topic, labels.labeled_fraction)
        given_label[~is_labeled] = unlabeled

    is_aux = true_label > 0
    if labels.noise in ("false_negative", "missed_unlabeled", "wrong_module"):
        # All three select among auxiliary stories with the same keys, so at equal rates
        # they affect the same stories.
        chosen = np.zeros(len(stories), dtype=bool)
        chosen[is_aux] = lowest_keys_within_groups(keys("mislabeled")[is_aux], topic[is_aux], labels.noise_rate)
        if labels.noise == "false_negative":
            given_label[chosen] = names.index(CORE)
        elif labels.noise == "missed_unlabeled":
            given_label[chosen] = unlabeled
        else:
            # Move to one of the other auxiliary labels: shift by 1 .. num_aux - 1, wrapping round.
            assert num_aux >= 2, "wrong_module needs at least two auxiliary topics"
            shift = rng_for(seed, "wrong_module_target").integers(1, num_aux, size=len(all_train))[position]
            given_label[chosen] = (true_label[chosen] - 1 + shift[chosen]) % num_aux + 1
    elif labels.noise == "false_positive":
        chosen = np.zeros(len(stories), dtype=bool)
        chosen[~is_aux] = lowest_keys_within_groups(keys("mislabeled")[~is_aux], topic[~is_aux], labels.noise_rate)
        target = rng_for(seed, "false_positive_target").integers(1, num_aux + 1, size=len(all_train))[position]
        given_label[chosen] = target[chosen]

    return true_label, given_label


# --------------------------------------------------------------------------- #
# 3. Training sequences, pools and batches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """The training stories that share a given label and a true topic.

    Each cell is one token stream, cut into ``num_sequences`` sequences which occupy
    rows ``first_sequence .. first_sequence + num_sequences - 1`` of the sequence table.
    Because a stream never mixes topics or given labels, every training sequence has
    exactly one true topic and exactly one given label.
    """

    given_label: str
    topic: str
    first_sequence: int
    num_sequences: int
    num_stories: int


@dataclass
class TrainingData:
    """Everything one run trains on."""

    sequences: np.ndarray  # (num_sequences, context_length + 1) token ids
    cells: list[Cell]
    aux_topics: tuple[str, ...]
    # One row per training story: who it is, what it is, what label it was given, and
    # which rows of ``sequences`` contain it. This is the split manifest saved with a run.
    manifest: pd.DataFrame

    def pool(self, given_label: str) -> list[Cell]:
        """The cells that make up one pool: all stories with this given label."""
        return [cell for cell in self.cells if cell.given_label == given_label]

    def pool_names(self) -> list[str]:
        """Pools that contain at least one sequence, in label order."""
        present = {cell.given_label for cell in self.cells if cell.num_sequences > 0}
        return [name for name in label_names(self.aux_topics) if name in present]

    def pool_num_sequences(self, given_label: str) -> int:
        return sum(cell.num_sequences for cell in self.pool(given_label))

    def pool_num_batches(self, given_label: str, batch_size: int, assembly: str) -> int:
        """Number of full batches in one pass over a pool (see ``make_pool_batches``)."""
        if assembly == "single_topic":
            return sum(cell.num_sequences // batch_size for cell in self.pool(given_label))
        return self.pool_num_sequences(given_label) // batch_size


def build_training_data(
    store: TokenStore,
    data: DataConfig,
    labels: LabelConfig,
    context_length: int,
    seed: int,
) -> TrainingData:
    """Turn the training stories into sequences, for one run's label settings."""
    aux_topics = tuple(data.aux_topics)
    unknown = sorted(set(aux_topics) - set(store.topics))
    if unknown:
        raise ValueError(f"auxiliary topics not found in the data: {unknown}")
    names = label_names(aux_topics)

    stories = first_stories_per_topic(store, store.stories_in("train"), data.train_fraction)
    if data.cap_topics_at_class_mean:
        stories = cap_topics_at_class_mean(store, stories, aux_topics)
    true_label, given_label = assign_given_labels(store, stories, aux_topics, labels, seed)
    topic = store.topic[stories]
    lengths = store.n_tokens[stories].astype(np.int64)

    # Group the stories into cells. A stable sort keeps the canonical order within a cell.
    cell_key = given_label.astype(np.int64) * len(store.topics) + topic
    order = np.argsort(cell_key, kind="stable")
    sorted_keys = cell_key[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    cell_starts = np.concatenate([[0], boundaries])
    cell_ends = np.concatenate([boundaries, [len(order)]])

    cells: list[Cell] = []
    pieces: list[np.ndarray] = []
    first_sequence = np.full(len(stories), -1, dtype=np.int64)
    last_sequence = np.full(len(stories), -1, dtype=np.int64)
    next_row = 0

    for start, end in zip(cell_starts, cell_ends):
        members = order[start:end]  # indices into `stories`, in canonical order
        stream = concatenate_stories(store, stories[members])
        sequences = cut_into_sequences(stream, context_length)
        key = int(sorted_keys[start])
        cells.append(
            Cell(
                given_label=names[key // len(store.topics)],
                topic=store.topics[key % len(store.topics)],
                first_sequence=next_row,
                num_sequences=len(sequences),
                num_stories=len(members),
            )
        )
        pieces.append(sequences)

        # Which sequences of this cell contain each story? A story occupies stream
        # positions [begin, finish]; sequence k covers positions [k*T, k*T + T].
        begin = np.cumsum(lengths[members]) - lengths[members]
        finish = begin + lengths[members] - 1
        first = np.maximum(0, (begin - 1) // context_length)
        last = np.minimum(len(sequences) - 1, finish // context_length)
        inside = first <= last  # false for stories that fall in the unused tail of the stream
        first_sequence[members[inside]] = next_row + first[inside]
        last_sequence[members[inside]] = next_row + last[inside]
        next_row += len(sequences)

    manifest = pd.DataFrame(
        {
            "story_id": store.story_id[stories],
            "true_topic": pd.Categorical.from_codes(topic, categories=list(store.topics)),
            "true_label": pd.Categorical.from_codes(true_label, categories=list(names)),
            "given_label": pd.Categorical.from_codes(given_label, categories=list(names)),
            "seed": np.int32(seed),
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
        }
    )
    return TrainingData(
        sequences=np.concatenate(pieces) if pieces else np.empty((0, context_length + 1), TOKEN_DTYPE),
        cells=cells,
        aux_topics=aux_topics,
        manifest=manifest,
    )


def make_pool_batches(
    cells: Sequence[Cell], batch_size: int, assembly: str, rng: np.random.Generator
) -> np.ndarray:
    """Group one pool's sequences into batches, for one pass over the pool.

    Returns an array of shape (num_batches, batch_size) holding row numbers of the
    sequence table, in the order the batches should be used. Sequences that do not fill
    a whole batch are left out of this pass.

    ``assembly`` is how sequences are grouped:

    * ``"single_topic"``: every batch holds sequences of a single cell, i.e. of one true
      topic. This is what the official code does. Its drawback is that it uses the true
      topic to build batches, which a trainer that only has the given labels could not do.
    * ``"mixed"``: the pool's sequences are shuffled together, whatever their true topic.
    """
    if assembly == "mixed":
        rows = np.concatenate(
            [np.arange(cell.first_sequence, cell.first_sequence + cell.num_sequences) for cell in cells]
            or [np.empty(0, dtype=np.int64)]
        )
        rows = rng.permutation(rows)
        return rows[: len(rows) // batch_size * batch_size].reshape(-1, batch_size)

    if assembly == "single_topic":
        per_cell = []
        for cell in cells:
            rows = rng.permutation(np.arange(cell.first_sequence, cell.first_sequence + cell.num_sequences))
            per_cell.append(rows[: len(rows) // batch_size * batch_size].reshape(-1, batch_size))
        batches = np.concatenate(per_cell or [np.empty((0, batch_size), dtype=np.int64)])
        return rng.permutation(batches)  # shuffles the batches, not their contents

    raise ValueError(f"unknown batch assembly {assembly!r}")


def assign_batches_to_schedule(
    schedule_pools: Sequence[str],
    data: TrainingData,
    batch_size: int,
    assembly: str,
    seed: int,
) -> np.ndarray:
    """Give every entry of a training schedule its batch.

    ``schedule_pools[i]`` names the pool that micro-batch i is drawn from. Each pool's
    batches are handed out in order; if the schedule asks a pool for more batches than
    one pass provides (longer training, or over-sampling), the pool is reshuffled and
    used again. Returns an array of shape (len(schedule_pools), batch_size) of row
    numbers into ``data.sequences``.
    """
    schedule_pools = np.asarray(schedule_pools)
    assigned = np.empty((len(schedule_pools), batch_size), dtype=np.int64)
    for pool in np.unique(schedule_pools):
        positions = np.flatnonzero(schedule_pools == pool)
        rng = rng_for(seed, f"batches:{pool}")
        passes: list[np.ndarray] = []
        available = 0
        while available < len(positions):
            one_pass = make_pool_batches(data.pool(pool), batch_size, assembly, rng)
            if len(one_pass) == 0:
                raise ValueError(f"pool {pool!r} is scheduled but does not hold a single full batch")
            passes.append(one_pass)
            available += len(one_pass)
        assigned[positions] = np.concatenate(passes)[: len(positions)]
    return assigned


# --------------------------------------------------------------------------- #
# Validation data, probe batches, elicitation data
# --------------------------------------------------------------------------- #


def dataset_names(aux_topics: Sequence[str]) -> tuple[str, ...]:
    """The datasets every model is scored on: core (the 44 other topics pooled) and each auxiliary topic."""
    return (CORE, *aux_topics)


def _topic_sequences(store: TokenStore, split: str, topic: str, context_length: int, fraction: float = 1.0):
    stories = first_stories_per_topic(store, store.stories_in(split, topic), fraction)
    return cut_into_sequences(concatenate_stories(store, stories), context_length)


def build_validation_sets(
    store: TokenStore, aux_topics: Sequence[str], context_length: int, fraction: float = 1.0
) -> dict[str, np.ndarray]:
    """Held-out sequences for every dataset: ``{dataset name: (n, context_length + 1) tokens}``.

    Each topic's validation stories are cut into sequences on their own; the core
    dataset is the union over all non-auxiliary topics.
    """
    core_topics = [name for name in store.topics if name not in aux_topics]
    sets = {CORE: np.concatenate([_topic_sequences(store, "val", name, context_length, fraction) for name in core_topics])}
    for name in aux_topics:
        sets[name] = _topic_sequences(store, "val", name, context_length, fraction)
    return sets


def build_probe_batches(
    store: TokenStore, aux_topics: Sequence[str], context_length: int, batch_size: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Two fixed held-out batches (A and B) per dataset, for the gradient probes.

    They come from the ``probe`` split, which is disjoint from both the training and
    the validation data, and they are the same for every run and every seed. For an
    auxiliary topic, A and B are the first and the second ``batch_size`` sequences of
    its probe stream. For core, sequences are taken evenly from all core topics and
    dealt alternately to A and B, so both batches cover every core topic.
    """
    batches: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    core_topics = [name for name in store.topics if name not in aux_topics]
    per_topic = int(np.ceil(2 * batch_size / len(core_topics)))
    by_topic = [_topic_sequences(store, "probe", name, context_length)[:per_topic] for name in core_topics]
    if min(len(sequences) for sequences in by_topic) < per_topic:
        raise ValueError("the probe split is too small for this batch size; rebuild the token store")
    # Shape (per_topic, num_core_topics, T + 1): reading it row by row cycles through the topics.
    dealt = np.stack(by_topic, axis=1).reshape(-1, context_length + 1)
    batches[CORE] = (dealt[0::2][:batch_size], dealt[1::2][:batch_size])

    for name in aux_topics:
        sequences = _topic_sequences(store, "probe", name, context_length)
        if len(sequences) < 2 * batch_size:
            raise ValueError("the probe split is too small for this batch size; rebuild the token store")
        batches[name] = (sequences[:batch_size], sequences[batch_size : 2 * batch_size])
    return batches


def build_elicitation_sequences(
    store: TokenStore, topic: str, num_sequences: int, context_length: int, seed: int
) -> np.ndarray:
    """Training sequences of one topic for the elicitation adversary to finetune on.

    They are a random sample of the topic's training sequences, taken from the topic's
    full training stream. The sample depends on the seed and the topic only, not on the
    label settings, so the adversary gets the same data for every model with that seed.
    """
    sequences = _topic_sequences(store, "train", topic, context_length)
    if len(sequences) < num_sequences:
        raise ValueError(f"topic {topic!r} has only {len(sequences)} training sequences, {num_sequences} requested")
    chosen = rng_for(seed, f"elicitation:{topic}").permutation(len(sequences))[:num_sequences]
    return sequences[np.sort(chosen)]
