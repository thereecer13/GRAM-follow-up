"""Saving and loading checkpoints.

A checkpoint holds everything needed to continue training or to analyse the model later:
the parameters of every group, the state of every group's optimizer, the step, and the
run's configuration. ``parameter_groups`` lists which state-dict entries belong to which
group, so a checkpoint can be taken apart by group without rebuilding the model.

Files in ``<run>/checkpoints/``:

* ``step_0000000.pt`` ...  the evenly spaced checkpoints kept for analysis (the first one
                           is the initialisation)
* ``final.pt``             the trained model
* ``latest.pt``            a rolling checkpoint for resuming after an interruption;
                           removed once training has finished
"""

from __future__ import annotations

from pathlib import Path

import torch

from gram.config import RunConfig, config_from_dict, config_to_dict
from gram.model import Transformer, build_model


def checkpoint_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints"


def step_checkpoint_path(run_dir: str | Path, step: int) -> Path:
    return checkpoint_dir(run_dir) / f"step_{step:07d}.pt"


def final_checkpoint_path(run_dir: str | Path) -> Path:
    return checkpoint_dir(run_dir) / "final.pt"


def latest_checkpoint_path(run_dir: str | Path) -> Path:
    return checkpoint_dir(run_dir) / "latest.pt"


def save_checkpoint(
    path: Path,
    step: int,
    total_steps: int,
    model: Transformer,
    optimizers: dict[str, torch.optim.Optimizer],
    config: RunConfig,
) -> None:
    """Write a checkpoint. The file appears under its final name only once it is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "total_steps": total_steps,
        "model": model.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "parameter_groups": model.parameter_group_names(),
        "module_names": list(model.module_names),
        "config": config_to_dict(config),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def newest_resumable_checkpoint(run_dir: str | Path) -> tuple[Path, int] | None:
    """The checkpoint with the highest step that training can continue from, and that step."""
    candidates = [(int(path.stem.split("_")[1]), path) for path in checkpoint_dir(run_dir).glob("step_*.pt")]
    latest = latest_checkpoint_path(run_dir)
    if latest.exists():
        candidates.append((int(load_checkpoint(latest)["step"]), latest))
    if not candidates:
        return None
    step, path = max(candidates, key=lambda candidate: candidate[0])
    return path, step


def model_from_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[Transformer, RunConfig, dict]:
    """Rebuild the model stored in a checkpoint, with its parameters in the dtype they were saved in.

    Returns the model, the run's configuration, and the raw checkpoint dictionary.
    """
    checkpoint = load_checkpoint(path, device)
    config = config_from_dict(checkpoint["config"])
    model = build_model(config.method, config.model, config.data.aux_topics)
    dtype = next(iter(checkpoint["model"].values())).dtype
    model.to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    return model, config, checkpoint
