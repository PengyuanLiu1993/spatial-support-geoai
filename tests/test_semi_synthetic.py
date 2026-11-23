import numpy as np

from gravity_gnn.semi_synthetic import (
    GRAVITY_COLUMNS,
    TRUE_SCALES_MINUTES,
    gravity_features,
    multiscale_gravity_features,
    relative_spatial_basis,
    raw_gravity_signal,
)


def test_gravity_features_are_monotone_in_travel_time() -> None:
    close = np.full((1, len(GRAVITY_COLUMNS)), 5.0)
    far = np.full((1, len(GRAVITY_COLUMNS)), 20.0)
    close_features = gravity_features(close)
    far_features = gravity_features(far)
    assert np.all(close_features > far_features)
    assert np.allclose(
        gravity_features(
            np.asarray([[TRUE_SCALES_MINUTES[name] for name in GRAVITY_COLUMNS]])
        ),
        np.exp(-1.0),
    )


def test_multiscale_features_have_expected_width() -> None:
    travel = np.ones((4, len(GRAVITY_COLUMNS)))
    transformed = multiscale_gravity_features(travel, scales=(5.0, 10.0, 20.0))
    assert transformed.shape == (4, 3 * len(GRAVITY_COLUMNS))


def test_raw_signal_is_deterministic_and_finite() -> None:
    travel = np.arange(1, 1 + 3 * len(GRAVITY_COLUMNS), dtype=float).reshape(3, -1)
    coordinates = np.asarray([[0.0, 0.0], [1000.0, 2000.0], [5000.0, 8000.0]])
    first = raw_gravity_signal(travel, coordinates)
    second = raw_gravity_signal(travel, coordinates)
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second)


def test_relative_spatial_basis_is_translation_invariant() -> None:
    coordinates = np.asarray([[0.0, 0.0], [10.0, 5.0], [20.0, 15.0]])
    shifted = coordinates + np.asarray([500_000.0, 200_000.0])
    first = relative_spatial_basis(coordinates)
    second = relative_spatial_basis(shifted)
    assert first.shape == (3, 45)
    assert np.allclose(first, second)
