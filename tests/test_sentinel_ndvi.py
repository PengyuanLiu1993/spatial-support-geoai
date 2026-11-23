from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from scripts.extract_sentinel_ndvi_context import select_items, zonal_features


class FakeItem:
    def __init__(self, item_id: str, tile: str, month: int, cloud: float):
        self.id = item_id
        self.datetime = datetime(2025, month, 15, tzinfo=timezone.utc)
        self.properties = {"s2:mgrs_tile": tile, "eo:cloud_cover": cloud}


def test_select_items_keeps_lowest_cloud_scene_per_tile_month():
    items = [
        FakeItem("cloudy", "30UYC", 4, 70.0),
        FakeItem("clear", "30UYC", 4, 5.0),
        FakeItem("other_month", "30UYC", 5, 20.0),
        FakeItem("other_tile", "31UCT", 4, 10.0),
    ]
    selected = select_items(items, per_tile_month=1)
    assert {item.id for item in selected} == {"clear", "other_month", "other_tile"}


def test_zonal_features_use_only_finite_pixels():
    labels = np.asarray([[1, 1], [2, 2]], dtype=np.int32)
    composite = np.asarray([[0.2, 0.6], [0.4, np.nan]], dtype=np.float32)
    temporal_std = np.asarray([[0.1, 0.3], [0.2, np.nan]], dtype=np.float32)
    observations = np.asarray([[4, 6], [5, 0]], dtype=np.float32)
    result = zonal_features(labels, composite, temporal_std, observations, 2)
    np.testing.assert_allclose(result["s2_ndvi_mean"], [0.4, 0.4])
    np.testing.assert_allclose(result["s2_ndvi_valid_fraction"], [1.0, 0.5])
    np.testing.assert_allclose(result["s2_ndvi_temporal_std_mean"], [0.2, 0.2])
    np.testing.assert_allclose(result["s2_ndvi_observations_mean"], [5.0, 2.5])
