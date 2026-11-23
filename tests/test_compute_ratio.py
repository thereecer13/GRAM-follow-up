"""The compute-ratio metric."""

import numpy as np
import pandas as pd
import pytest

from gram.compute_ratio import build_reference, fit_power_law, mean_and_ci90, power_law, step_equivalent

TRUE = (6.2, 0.13, -78.0)  # roughly what the official Simple Stories baseline curves fit to


def curve(seed, steps=np.arange(83, 16544, 83), noise=0.004, parameters=TRUE):
    rng = np.random.default_rng(seed)
    return steps, power_law(steps, *parameters) * np.exp(noise * rng.standard_normal(len(steps)))


def baseline_frames(seeds=(1, 2, 3), datasets=("core", "alien")):
    curve_rows, final_rows = [], []
    for seed in seeds:
        for index, dataset in enumerate(datasets):
            parameters = (TRUE[0] + 0.1 * index, TRUE[1], TRUE[2])
            steps, losses = curve(seed * 10 + index, parameters=parameters)
            curve_rows += [{"seed": seed, "dataset": dataset, "step": s, "loss": l} for s, l in zip(steps, losses)]
            # Fully trained baselines end a little below their own curve (the final learning-rate decay).
            final = power_law(16544, *parameters) * 0.985 * (1 + 0.002 * (seed - 2))
            final_rows.append({"seed": seed, "dataset": dataset, "loss": final})
    return pd.DataFrame(curve_rows), pd.DataFrame(final_rows)


def test_fit_recovers_a_known_power_law():
    steps, losses = curve(0, noise=0.0)
    scale, alpha, offset = fit_power_law(steps, losses)
    assert (scale, alpha, offset) == pytest.approx(TRUE, rel=1e-3)
    noisy = fit_power_law(*curve(0, noise=0.004))
    assert np.abs(power_law(steps, *noisy) / power_law(steps, *TRUE) - 1).max() < 0.004


def test_step_equivalent_inverts_the_curve():
    for step in (500.0, 5000.0, 16544.0, 40000.0):
        assert step_equivalent(float(power_law(step, *TRUE)), *TRUE) == pytest.approx(step, rel=1e-9)
    # No floor in the curve: a loss better than anything the baseline reached still maps to a finite step.
    assert step_equivalent(float(power_law(16544, *TRUE)) * 0.97, *TRUE) > 16544


def test_the_baseline_scores_one_by_construction():
    curves, finals = baseline_frames()
    reference = build_reference(curves, finals)
    for dataset, group in finals.groupby("dataset"):
        ratios = [reference.compute_ratio(loss, dataset) for loss in group.loss]
        assert np.mean(ratios) == pytest.approx(1.0, abs=1e-12)
        assert not np.allclose(ratios, 1.0), "individual baseline seeds scatter around 1"


def test_denominator_is_the_mapped_final_loss_not_the_number_of_steps():
    curves, finals = baseline_frames()
    reference = build_reference(curves, finals)
    # The baselines finish below their fitted curve, so their final loss maps to a step beyond
    # the end of training. Dividing by the actual 16544 steps would score the baseline above 1.
    assert all(denominator > 16544 * 1.05 for denominator in reference.denominators.values())


def test_lower_loss_means_higher_ratio_and_small_loss_gaps_are_large_compute_gaps():
    curves, finals = baseline_frames()
    reference = build_reference(curves, finals)
    baseline_loss = finals[finals.dataset == "core"].loss.mean()
    better, worse = reference.compute_ratio(baseline_loss - 0.01, "core"), reference.compute_ratio(baseline_loss + 0.01, "core")
    assert better > 1 > worse
    # With an exponent near 0.13, a hundredth of a nat is worth several percent of training.
    assert 0.03 < 1 - worse < 0.07
    assert np.isnan(reference.compute_ratio(float("nan"), "core"))


def test_seeds_are_pooled_into_one_curve_per_dataset():
    curves, finals = baseline_frames()
    pooled = build_reference(curves, finals)
    single = build_reference(curves[curves.seed == 1], finals[finals.seed == 1])
    assert pooled.num_baseline_seeds == 3 and single.num_baseline_seeds == 1
    assert set(pooled.curves) == {"core", "alien"}
    assert pooled.curves["core"] != single.curves["core"]
    summary = pooled.describe()
    assert list(summary.dataset) == ["core", "alien"] and (summary.alpha.between(0.12, 0.14)).all()


def test_mean_and_confidence_interval_match_the_papers_convention():
    # Per-seed GRAM forget ratios from the official results (core-only profile): 0.7643 (0.0166).
    mean, half_width = mean_and_ci90([0.7558, 0.7622, 0.7749])
    assert mean == pytest.approx(0.7643, abs=1e-4)
    assert half_width == pytest.approx(0.0166, abs=3e-4)
    assert mean_and_ci90([0.8])[0] == 0.8 and np.isnan(mean_and_ci90([0.8])[1])
    assert np.isnan(mean_and_ci90([])[0])
