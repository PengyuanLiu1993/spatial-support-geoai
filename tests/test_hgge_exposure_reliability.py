from __future__ import annotations

import numpy as np

from scripts.summarise_hgge_exposure_reliability import (
    finite_pairwise_correlations,
    one_way_random_effects_icc,
)


def test_perfect_seed_agreement_has_unit_icc() -> None:
    areas = np.asarray([1.0, 2.0, 4.0, 8.0])
    values = np.repeat(areas[:, None], 5, axis=1)
    result = one_way_random_effects_icc(values)
    assert result["icc_single_seed"] == 1.0
    assert result["icc_seed_mean"] == 1.0


def test_seed_noise_reduces_icc() -> None:
    values = np.asarray(
        [
            [1.0, 1.5, 0.5],
            [2.0, 1.0, 3.0],
            [3.0, 4.0, 2.0],
            [4.0, 2.5, 5.5],
        ]
    )
    result = one_way_random_effects_icc(values)
    assert result["icc_single_seed"] < result["icc_seed_mean"] < 1.0


def test_pairwise_seed_correlations_use_seed_columns() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )
    correlations = finite_pairwise_correlations(values)
    np.testing.assert_allclose(correlations, np.ones(3))
