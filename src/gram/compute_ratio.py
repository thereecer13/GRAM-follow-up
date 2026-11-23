"""The compute-ratio metric (paper Section 2 and Appendix M).

A loss on its own is hard to read: is 1.71 nats close to 1.68 or far from it? The
compute ratio answers in units of training effort. It asks: *how far through its own
training was the baseline when it reached this loss?*

For one dataset D:

1. Record the baseline's validation loss on D at many points during its training.
2. Fit a power law through those points,  L(s) = A * (s + s0) ** (-alpha),  with
   s the training step.
3. Invert it. A loss l maps to the "step-equivalent" s(l) = (A / l) ** (1 / alpha) - s0:
   the step at which the baseline had that loss.
4. The compute ratio of a model M on D is

       CR(M, D) = s(loss of M on D) / s(final loss of the baseline on D).

A ratio of 1 means "as good as the fully trained baseline"; 0.77 means "as good as the
baseline was 77% of the way through training". For data a model should know (core,
retain), higher is better; for data it should not know (forget), lower is better.

Details that matter, all from Appendix M:

* The fit is least squares on the *logarithm* of the loss, so that the flat, late part
  of the curve (where the ratios are read off) counts as much as the steep early part.
* There is no additive floor in the power law. That keeps it invertible for any loss,
  including losses better than the baseline's, which then give a ratio above 1.
* With several baseline seeds, all their points are pooled into *one* curve per dataset.
* The denominator is the baseline's final loss mapped through the fitted curve, not its
  actual number of steps. Any bias of the fit then enters numerator and denominator
  alike and cancels. With several seeds it is the mean of their step-equivalents.

This module needs only numpy and scipy, so results can be recomputed without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def fit_power_law(steps: np.ndarray, losses: np.ndarray) -> tuple[float, float, float]:
    """Fit  loss = A * (step + s0) ** (-alpha)  in log space. Returns (A, alpha, s0).

    The starting point and bounds are those of the official analysis code, so that the
    same learning curve gives the same fit here as there.
    """
    steps = np.asarray(steps, dtype=float)
    losses = np.asarray(losses, dtype=float)
    spacing = steps[1] - steps[0] if len(steps) > 1 else 1.0
    tiny = 1e-6

    # A > 0; a sane range for the exponent; and s0 large enough that step + s0 > 0
    # everywhere (the logarithm must exist) but not unboundedly large.
    lower = [1e-10, 1e-3, -float(steps.min()) + tiny]
    upper = [1e8, 5.0, float(steps.max()) + 10 * spacing]
    start = [max(float(losses[0]), 1e-3), 0.3, min(100.0, upper[2] * 0.9)]
    log_losses = np.log(losses)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        scale, alpha, offset = parameters
        return np.log(scale) - alpha * np.log(steps + offset) - log_losses

    result = least_squares(residuals, start, bounds=(lower, upper), max_nfev=40_000)
    scale, alpha, offset = result.x
    return float(scale), float(alpha), float(offset)


def power_law(steps: np.ndarray, scale: float, alpha: float, offset: float) -> np.ndarray:
    return scale * (np.asarray(steps, dtype=float) + offset) ** (-alpha)


def step_equivalent(loss: float, scale: float, alpha: float, offset: float) -> float:
    """The baseline training step at which the fitted curve reaches ``loss``."""
    step = (scale / max(loss, 1e-12)) ** (1.0 / alpha) - offset
    return max(float(step), 1e-3)


@dataclass
class Reference:
    """What a loss is compared against: the baseline's curve and final loss, per dataset."""

    curves: dict[str, tuple[float, float, float]]  # dataset -> (A, alpha, s0)
    denominators: dict[str, float]  # dataset -> mean step-equivalent of the baseline's final loss
    baseline_final_losses: dict[str, list[float]] = field(default_factory=dict)
    num_baseline_seeds: int = 0
    # How well the power law describes the measured curve: the root-mean-square difference
    # between fitted and measured log-loss. 0.003 means the fit is typically off by 0.3%.
    fit_rms_log_error: dict[str, float] = field(default_factory=dict)

    def compute_ratio(self, loss: float, dataset: str) -> float:
        if loss is None or not np.isfinite(loss):
            return float("nan")
        return step_equivalent(loss, *self.curves[dataset]) / self.denominators[dataset]

    def describe(self) -> pd.DataFrame:
        rows = []
        for dataset, (scale, alpha, offset) in self.curves.items():
            rows.append(
                {
                    "dataset": dataset,
                    "A": scale,
                    "alpha": alpha,
                    "s0": offset,
                    "fit_rms_log_error": self.fit_rms_log_error.get(dataset, float("nan")),
                    "denominator_steps": self.denominators[dataset],
                    "baseline_final_loss_mean": float(np.mean(self.baseline_final_losses[dataset])),
                    "num_baseline_seeds": self.num_baseline_seeds,
                }
            )
        return pd.DataFrame(rows)


def build_reference(curve_points: pd.DataFrame, final_losses: pd.DataFrame) -> Reference:
    """Fit the reference from one or more baseline runs.

    ``curve_points`` has one row per validation measurement taken during baseline
    training, with columns ``seed``, ``dataset``, ``step``, ``loss``.
    ``final_losses`` has the fully trained baselines' losses: ``seed``, ``dataset``, ``loss``.

    All seeds are pooled into a single curve per dataset, and the denominator is the mean
    over seeds of each baseline's own final loss mapped through that curve.
    """
    curves, denominators, finals, fit_errors = {}, {}, {}, {}
    for dataset, points in curve_points.groupby("dataset", sort=False):
        points = points.sort_values(["seed", "step"])
        steps, measured = points["step"].to_numpy(), points["loss"].to_numpy()
        curves[dataset] = fit_power_law(steps, measured)
        log_error = np.log(power_law(steps, *curves[dataset])) - np.log(measured)
        fit_errors[dataset] = float(np.sqrt(np.mean(log_error**2)))
        losses = final_losses.loc[final_losses["dataset"] == dataset].sort_values("seed")["loss"].tolist()
        if not losses:
            raise ValueError(f"no final baseline loss for dataset {dataset!r}")
        finals[dataset] = losses
        denominators[dataset] = float(np.mean([step_equivalent(loss, *curves[dataset]) for loss in losses]))
    return Reference(
        curves=curves,
        denominators=denominators,
        baseline_final_losses=finals,
        num_baseline_seeds=int(final_losses["seed"].nunique()),
        fit_rms_log_error=fit_errors,
    )


def mean_and_ci90(values) -> tuple[float, float]:
    """Mean over seeds and the half-width of its 90% t-interval (what the paper's tables print)."""
    from scipy import stats

    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan")
    half_width = stats.t.ppf(0.95, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(half_width)
