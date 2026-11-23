import numpy as np

from scripts.run_hgge_informed_associations import (
    cross_fitted_standardize,
    minimum_detectable_effect,
    spatial_block_bootstrap_coefficients,
)


def test_cross_fitted_standardize_uses_other_folds_and_common_shift_scale() -> None:
    folds = np.asarray([0, 0, 1, 1, 2, 2])
    shifts = np.asarray([0.0, 5.0])
    raw = np.asarray(
        [
            [1.0, 0.5],
            [2.0, 1.5],
            [3.0, 2.5],
            [4.0, 3.5],
            [5.0, 4.5],
            [6.0, 5.5],
        ]
    )
    exposure, shifted, scaling = cross_fitted_standardize(raw, shifts, folds)
    assert np.isfinite(exposure).all()
    assert np.isfinite(shifted[5.0]).all()
    assert len(scaling) == 3
    # Every five-minute counterfactual is exactly 0.5 training SD below the
    # baseline only when that fold's training SD is 2.0; the function must use
    # the same fold-specific mean and scale for both states.
    for fold, row in enumerate(scaling):
        selected = folds == fold
        expected = -0.5 / row["training_sd"]
        np.testing.assert_allclose(
            shifted[5.0][selected] - exposure[selected], expected
        )


def test_spatial_block_bootstrap_preserves_linear_signal() -> None:
    rng = np.random.default_rng(71)
    x = rng.normal(size=200)
    design = np.column_stack((np.ones(len(x)), x))
    blocks = np.repeat(np.arange(20), 10)
    outcome = 1.5 + 2.0 * x + rng.normal(scale=0.1, size=len(x))
    draws = spatial_block_bootstrap_coefficients(
        design, outcome, blocks, replicates=250, seed=71
    )
    assert draws.shape == (250, 2)
    assert abs(float(np.mean(draws[:, 1])) - 2.0) < 0.05


def test_mde_increases_with_stricter_alpha_and_higher_power() -> None:
    baseline = minimum_detectable_effect(0.2, 25, 0.05, 0.80)
    strict = minimum_detectable_effect(0.2, 25, 0.05 / 96, 0.80)
    high_power = minimum_detectable_effect(0.2, 25, 0.05, 0.90)
    assert strict > baseline
    assert high_power > baseline
