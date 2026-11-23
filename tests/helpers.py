"""Small synthetic stand-ins for the real data and model, so the tests run in seconds on a CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gram.config import (
    CheckpointConfig,
    DataConfig,
    ElicitConfig,
    ElicitProtocol,
    EvalConfig,
    ModelConfig,
    OptimConfig,
    ProbeConfig,
    RunConfig,
)
from gram.data import TokenStore, choose_probe_stories, write_token_store

# Eight topics; as in the real data, the first four alphabetically are the auxiliary ones.
TOPICS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
AUX = TOPICS[:4]
CORE_TOPICS = TOPICS[4:]

VOCAB = 64
EOS = 1
CONTEXT = 16

TINY_MODEL = ModelConfig(
    vocab_size=VOCAB,
    context_length=CONTEXT,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    d_model=32,
    eos_token_id=EOS,
    dense_mlp_dim=64,
    core_mlp_dim=64,
    aux_mlp_dim=16,
)


# Each topic owns TOKENS_PER_TOPIC token ids, and its stories walk around them in a cycle.
TOKENS_PER_TOPIC = 7
FOLLOW_CYCLE = 0.9  # probability that the next token is the cycle's successor


def topic_tokens(topic: str) -> np.ndarray:
    """The token ids that belong to one topic (ids 0 and 1 are reserved for unknown / end-of-story)."""
    start = 2 + TOPICS.index(topic) * TOKENS_PER_TOPIC
    return np.arange(start, start + TOKENS_PER_TOPIC)


def make_story(topic: str, length: int, rng: np.random.Generator) -> np.ndarray:
    """A story the tiny models can learn, and can only learn from that topic's own data.

    The story stays within the topic's own tokens. With probability FOLLOW_CYCLE the next
    token is the current one advanced by a topic-specific stride; otherwise it is a random
    token of the topic. Knowing the stride takes a model from log(7) = 1.95 nats per token
    down to roughly 0.5, and the stride of one topic says nothing about another's.
    """
    own = topic_tokens(topic)
    stride = 1 + TOPICS.index(topic) % (TOKENS_PER_TOPIC - 1)
    position = int(rng.integers(TOKENS_PER_TOPIC))
    story = []
    for _ in range(length - 1):
        story.append(own[position])
        if rng.random() < FOLLOW_CYCLE:
            position = (position + stride) % TOKENS_PER_TOPIC
        else:
            position = int(rng.integers(TOKENS_PER_TOPIC))
    return np.array(story + [EOS])


def make_store(
    directory: str | Path,
    train_stories_per_topic: int = 120,
    val_stories_per_topic: int = 60,
    probe_sequences_per_topic: int = 8,
    seed: int = 0,
) -> TokenStore:
    """Write a small synthetic token store shaped like the real one and open it.

    Stories of different topics are interleaved, as they are in the real store, and
    topics are deliberately given slightly different amounts of data.
    """
    rng = np.random.default_rng(seed)
    topic, split, pieces = [], [], []
    for split_name, per_topic in (("train", train_stories_per_topic), ("val", val_stories_per_topic)):
        names = rng.permutation(np.repeat(TOPICS, per_topic))
        for name in names:
            pieces.append(make_story(str(name), int(rng.integers(6, 30)), rng))
            topic.append(str(name))
            split.append(split_name)

    topic = np.array(topic)
    split = np.array(split, dtype=object)
    n_tokens = np.array([len(piece) for piece in pieces])
    split[choose_probe_stories(topic, split, n_tokens, probe_sequences_per_topic * CONTEXT + 1)] = "probe"

    write_token_store(
        directory,
        tokens=np.concatenate(pieces),
        # Story ids are deliberately not 0, 1, 2, ... so that tests cannot confuse an id with a position.
        story_id=rng.permutation(len(pieces)) + 1000,
        topic=topic,
        split=split,
        n_tokens=n_tokens,
        meta={"eos_token_id": EOS, "vocab_size": VOCAB},
    )
    return TokenStore(directory)


def tiny_run_config(tmp_path: Path, method: str = "gram", **overrides) -> RunConfig:
    """A complete run configuration small enough to train in a second or two."""
    config = RunConfig(
        name=f"{method}_test",
        method=method,
        seed=1,
        output_dir=str(tmp_path / "runs"),
        device="cpu",
        data=DataConfig(store_dir=str(tmp_path / "store"), aux_topics=AUX, batch_assembly="single_topic"),
        model=TINY_MODEL,
        optim=OptimConfig(lr=3e-3, batch_size=4, micro_batch_size=4, precision="fp32"),
        eval=EvalConfig(batch_size=16, curve_points=0, curve_core_batches=4),
        elicit=ElicitConfig(
            protocols=[
                ElicitProtocol(name="code", num_sequences=8, max_steps=6, patience=2),
                ElicitProtocol(name="paper", num_sequences=4, max_steps=3, patience=0),
            ]
        ),
        checkpoints=CheckpointConfig(num_spaced=4, resume_every_steps=7),
        probes=ProbeConfig(enabled=True, num_points=3),
    )
    for dotted, value in overrides.items():
        target = config
        *parents, leaf = dotted.split("__")
        for parent in parents:
            target = getattr(target, parent)
        assert hasattr(target, leaf), f"unknown config field {dotted}"
        setattr(target, leaf, value)
    config.validate()
    return config
