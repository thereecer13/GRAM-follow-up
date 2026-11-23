"""Running one experiment from start to finish.

    python -m gram.run configs/some_run.yaml           # train, evaluate, elicit
    python -m gram.run configs/some_run.yaml --check   # re-evaluate the saved model and compare

A run goes through three stages. Each writes its results to the run directory and is
skipped if they are already there, so the same command can simply be repeated after an
interruption:

1. ``train``     the model (gram/train.py), leaving ``checkpoints/final.pt``
2. ``evaluate``  the validation loss of every profile on every dataset -> ``eval/losses.csv``
3. ``elicit``    adversarial finetuning on each forget topic -> ``eval/elicitation*.csv``

Evaluation never uses the model that training left in memory. It loads ``final.pt`` from
disk into a freshly built model, so every reported number is, by construction, a number
that the saved checkpoint reproduces. To confirm that the save/load round trip lost
nothing, training records a small "fingerprint" (a few losses of the in-memory model)
which the evaluation recomputes from the checkpoint and requires to match.

``--check`` repeats the evaluation of an already finished run and compares it with the
stored losses. Do this before reporting numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from gram.checkpoint import final_checkpoint_path, model_from_checkpoint
from gram.config import RunConfig, load_config
from gram.data import TokenStore, build_validation_sets
from gram.elicit import run_elicitation, summarise_elicitation
from gram.eval import evaluate_profiles, fingerprint
from gram.train import require_same_config, resolve_device, token_tensor, train

# Losses are means over tens of thousands of predictions, computed in float32.
# Re-evaluating the same checkpoint on the same machine should agree far better than this.
CHECK_TOLERANCE = 1e-5


def load_validation_sets(config: RunConfig, store: TokenStore, device: torch.device) -> dict[str, torch.Tensor]:
    sets = build_validation_sets(store, config.data.aux_topics, config.model.context_length, config.data.val_fraction)
    return {name: token_tensor(tokens, device) for name, tokens in sets.items()}


def evaluate_run(config: RunConfig, check: bool = False) -> pd.DataFrame:
    """Evaluate the saved final checkpoint. With ``check``, compare against the stored losses instead of writing."""
    run_dir = config.run_dir
    device = resolve_device(config.device)
    store = TokenStore(config.data.store_dir)
    model, _, _ = model_from_checkpoint(final_checkpoint_path(run_dir), device)
    validation_sets = load_validation_sets(config, store, device)

    # The checkpoint must reproduce what the model in memory computed at the end of training.
    recorded = json.loads((run_dir / "train_summary.json").read_text()).get("fingerprint")
    if recorded is not None:
        recomputed = fingerprint(model, validation_sets, config.eval.batch_size)
        worst = max(abs(recomputed[key] - value) for key, value in recorded.items())
        if worst > CHECK_TOLERANCE:
            raise RuntimeError(
                f"{run_dir}: the saved checkpoint does not reproduce the trained model "
                f"(largest loss difference {worst:.2e}). Do not report numbers from this run."
            )

    losses = evaluate_profiles(model, validation_sets, config.eval.batch_size)
    path = run_dir / "eval" / "losses.csv"
    if check:
        stored = pd.read_csv(path)
        merged = stored.merge(losses, on=["profile", "dataset"], suffixes=("_stored", "_now"))
        assert len(merged) == len(stored) == len(losses), "the set of evaluated profiles changed"
        worst = float((merged.loss_stored - merged.loss_now).abs().max())
        result = {
            "ok": worst <= CHECK_TOLERANCE,
            "largest_difference": worst,
            "tolerance": CHECK_TOLERANCE,
            "num_losses": len(merged),
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (run_dir / "eval" / "check.json").write_text(json.dumps(result, indent=2))
        print(f"{'ok      ' if result['ok'] else 'MISMATCH'} {config.name}: {len(merged)} losses re-evaluated "
              f"from the checkpoint, largest difference {worst:.2e}")
        if not result["ok"]:
            raise RuntimeError(f"{run_dir}: re-evaluation does not match the stored losses")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        losses.to_csv(path, index=False)
    return losses


def elicit_run(config: RunConfig) -> pd.DataFrame:
    """Run the elicitation adversary against the saved final checkpoint."""
    run_dir = config.run_dir
    device = resolve_device(config.device)
    store = TokenStore(config.data.store_dir)
    model, _, _ = model_from_checkpoint(final_checkpoint_path(run_dir), device)
    validation_sets = load_validation_sets(config, store, device)

    curves = run_elicitation(model, config, store, validation_sets, device)
    summary = summarise_elicitation(curves)
    curves.to_csv(run_dir / "eval" / "elicitation_curves.csv", index=False)
    summary.to_csv(run_dir / "eval" / "elicitation.csv", index=False)
    return summary


def run(config: RunConfig, stop_after_steps: int | None = None) -> str:
    """Train, evaluate and elicit. Returns "finished" or "interrupted"."""
    run_dir = config.run_dir
    require_same_config(run_dir, config)
    if (run_dir / "DONE").exists():
        return "finished"

    started = time.time()
    if train(config, stop_after_steps=stop_after_steps) != "finished":
        return "interrupted"
    seconds = {"train": time.time() - started}

    tick = time.time()
    if not (run_dir / "eval" / "losses.csv").exists():
        evaluate_run(config)
    seconds["evaluate"] = time.time() - tick

    tick = time.time()
    if config.elicit.enabled and not (run_dir / "eval" / "elicitation.csv").exists():
        elicit_run(config)
    seconds["elicit"] = time.time() - tick

    (run_dir / "DONE").write_text(json.dumps({"seconds": seconds}, indent=2))
    return "finished"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="the run's YAML configuration file")
    parser.add_argument("--check", action="store_true", help="re-evaluate a finished run and compare with its stored losses")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.check:
        evaluate_run(config, check=True)
        return
    status = run(config)
    print(f"{config.name}: {status}")
    # A distinct exit code lets a wrapper script tell "interrupted, run me again" from "done".
    sys.exit(0 if status == "finished" else 75)


if __name__ == "__main__":
    main()
