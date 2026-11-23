from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from scripts.run_gnn_baselines import queen_edges


def test_queen_edges_are_symmetric_and_exclude_self_edges():
    frame = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1)]},
        geometry="geometry",
        crs=27700,
    )
    edges = queen_edges(frame)
    assert {tuple(edge) for edge in edges.T} == {(0, 1), (1, 0)}
