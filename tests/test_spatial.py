from __future__ import annotations

import numpy as np

from gravity_gnn.spatial import gravity_neighborhood_features


def test_gravity_neighborhood_is_target_free_and_distance_sensitive():
    coordinates = np.asarray([[0.0, 0.0], [100.0, 0.0], [1000.0, 0.0]])
    features = np.asarray([[0.0], [1.0], [10.0]])
    mass = np.ones(3)
    result = gravity_neighborhood_features(
        coordinates, features, mass, scales_m=(200.0,), neighbors=2
    )
    assert result.shape == (3, 3)
    neighbor_mean = result[:, 1]
    assert 1.0 < neighbor_mean[0] < 2.0
    assert 0.5 < neighbor_mean[2] < 1.0
    assert np.isfinite(result).all()


def test_gravity_neighborhood_rejects_invalid_scale():
    with np.testing.assert_raises(ValueError):
        gravity_neighborhood_features(
            np.zeros((2, 2)), np.zeros((2, 1)), np.ones(2), scales_m=(0.0,)
        )
