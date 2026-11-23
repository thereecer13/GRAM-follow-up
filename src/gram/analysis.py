"""Helpers for looking at finished runs.

    from gram.analysis import load_model, story_exposures

    model, config = load_model("runs/code_exact/gram_seed1")            # the trained model
    model, config = load_model("runs/code_exact/gram_seed1", step=1654)  # an earlier checkpoint
    seen = story_exposures("runs/labeled_fraction/gram_labeled0.5_seed1")
    unlabeled_and_trained = seen[seen.rule == "unlabeled"].story_id.unique()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gram.checkpoint import final_checkpoint_path, model_from_checkpoint, step_checkpoint_path
from gram.config import RunConfig
from gram.model import Transformer


def load_model(
    run_dir: str | Path, step: int | None = None, device: str | torch.device = "cpu", dtype: torch.dtype | None = torch.float32
) -> tuple[Transformer, RunConfig]:
    """Load a run's model: the final one, or the kept checkpoint at ``step``.

    Checkpoints store the parameters in the precision they were trained in (bfloat16 for
    the real runs). By default they are converted to float32, which is exact and more
    convenient to analyse; pass ``dtype=None`` to keep them as saved.
    """
    path = final_checkpoint_path(run_dir) if step is None else step_checkpoint_path(run_dir, step)
    if not path.exists():
        kept = json.loads((Path(run_dir) / "plan.json").read_text())["checkpoint_steps"]
        raise FileNotFoundError(f"no checkpoint at step {step} in {run_dir}; kept steps are {kept} and the final one")
    model, config, _ = model_from_checkpoint(path, device)
    if dtype is not None:
        model = model.to(dtype)
    return model.eval(), config


def story_exposures(run_dir: str | Path) -> pd.DataFrame:
    """Every time a story was trained on: one row per (story, training sequence, micro-batch).

    Columns: ``story_id``, ``true_topic``, ``true_label``, ``given_label``,
    ``sequence_row``, ``step`` (the optimizer step, counted from 1), ``micro_batch``,
    ``rule`` (the routing rule of that micro-batch), ``active`` and ``update`` (the modules
    in the forward pass and the groups updated).

    A story usually spans two or three training sequences, so it normally appears in two
    or three rows. A story that is in the manifest but not in this table was never
    trained on: its sequences fell in an unused stream tail, in an incomplete batch, or
    in the part of an epoch cut off by the equal-compute rescaling.
    """
    run_dir = Path(run_dir)
    manifest = pd.read_parquet(run_dir / "manifest.parquet")
    log = np.load(run_dir / "batch_log.npz")
    accumulation = json.loads((run_dir / "plan.json").read_text())["accumulation_steps"]

    # One row per (story, sequence row that contains it).
    in_sequences = manifest[manifest.first_sequence >= 0]
    spans = (in_sequences.last_sequence - in_sequences.first_sequence + 1).to_numpy()
    story_rows = in_sequences.loc[in_sequences.index.repeat(spans)].copy()
    offset = np.arange(len(story_rows)) - np.repeat(np.cumsum(spans) - spans, spans)
    story_rows["sequence_row"] = story_rows.first_sequence.to_numpy() + offset

    # One row per (sequence row, micro-batch that trained on it).
    rows = log["sequence_rows"]
    micro_batch = np.repeat(np.arange(len(rows)), rows.shape[1])
    rule_index = log["rule_index"][micro_batch]
    trained = pd.DataFrame(
        {
            "sequence_row": rows.ravel(),
            "step": micro_batch // accumulation + 1,
            "micro_batch": micro_batch,
            "rule": log["rule_names"][rule_index],
            "active": log["rule_active"][rule_index],
            "update": log["rule_update"][rule_index],
        }
    )

    exposures = story_rows.merge(trained, on="sequence_row", how="inner")
    columns = ["story_id", "true_topic", "true_label", "given_label", "sequence_row", "step", "micro_batch", "rule", "active", "update"]
    return exposures[columns].sort_values(["step", "micro_batch", "sequence_row"], ignore_index=True)
