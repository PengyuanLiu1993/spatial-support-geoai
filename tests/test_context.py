import geopandas as gpd
import numpy as np
from shapely.geometry import box

from gravity_gnn.context import (
    area_weighted_grid_values,
    normalise_postcode,
    radius_counts,
    radius_weighted_sums,
)


def test_radius_context_helpers() -> None:
    query = np.array([[0.0, 0.0], [10.0, 0.0]])
    source = np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]])
    counts = radius_counts(query, source, (5.0, 11.0))
    weighted = radius_weighted_sums(query, source, np.array([1.0, 2.0, 4.0]), (5.0, 11.0))
    assert counts.tolist() == [[2.0, 2.0], [0.0, 3.0]]
    assert weighted.tolist() == [[3.0, 3.0], [0.0, 7.0]]


def test_area_weighted_grid_values() -> None:
    areas = gpd.GeoDataFrame(
        {"code": ["A"], "geometry": [box(0.0, 0.0, 2.0, 1.0)]}, crs=27700
    )
    grid = gpd.GeoDataFrame(
        {"value": [2.0, 6.0], "geometry": [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)]},
        crs=27700,
    )
    result = area_weighted_grid_values(areas, grid, "value", "code")
    assert result["A"] == 4.0


def test_normalise_postcode() -> None:
    import pandas as pd

    values = pd.Series([" m1 1aa ", "SW1A 1AA", None])
    assert normalise_postcode(values).tolist() == ["M11AA", "SW1A1AA", ""]
