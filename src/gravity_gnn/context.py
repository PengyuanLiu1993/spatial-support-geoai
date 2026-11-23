from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box


def load_pcm_grid(path: str | Path, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Load and spatially trim a DEFRA 1 km Pollution Climate Mapping grid."""

    frame = pd.read_csv(path, skiprows=5, low_memory=False)
    if len(frame.columns) != 4:
        raise ValueError(f"unexpected PCM columns in {path}: {frame.columns.tolist()}")
    value_name = str(frame.columns[-1])
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    frame = frame.dropna(subset=[value_name, "x", "y"]).copy()
    min_x, min_y, max_x, max_y = bounds
    frame = frame[
        frame["x"].between(min_x - 500.0, max_x + 500.0)
        & frame["y"].between(min_y - 500.0, max_y + 500.0)
    ].copy()
    frame["geometry"] = [
        box(x - 500.0, y - 500.0, x + 500.0, y + 500.0)
        for x, y in zip(frame["x"], frame["y"], strict=True)
    ]
    return gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:27700")


def area_weighted_grid_values(
    areas: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    value_column: str,
    area_id: str,
) -> pd.Series:
    """Area-weight a regular grid over polygons without target information."""

    if areas.crs != grid.crs:
        grid = grid.to_crs(areas.crs)
    values: dict[object, float] = {}
    for row in areas[[area_id, "geometry"]].itertuples(index=False):
        geometry = row.geometry
        candidates = grid.iloc[list(grid.sindex.query(geometry, predicate="intersects"))]
        intersections = candidates.geometry.intersection(geometry)
        weights = intersections.area.to_numpy(dtype=float)
        valid = weights > 0
        if not np.any(valid):
            values[getattr(row, area_id)] = float("nan")
            continue
        observations = candidates[value_column].to_numpy(dtype=float)
        values[getattr(row, area_id)] = float(np.average(observations[valid], weights=weights[valid]))
    return pd.Series(values, name=value_column, dtype=float)


def radius_counts(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    radii: Iterable[float],
) -> np.ndarray:
    """Count source points around every query for several Euclidean radii."""

    radii = tuple(float(radius) for radius in radii)
    if not radii:
        return np.empty((len(query_xy), 0), dtype=np.float32)
    if len(source_xy) == 0:
        return np.zeros((len(query_xy), len(radii)), dtype=np.float32)
    tree = cKDTree(np.asarray(source_xy, dtype=float))
    return np.column_stack(
        [tree.query_ball_point(query_xy, radius, return_length=True) for radius in radii]
    ).astype(np.float32)


def radius_weighted_sums(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    source_values: np.ndarray,
    radii: Iterable[float],
) -> np.ndarray:
    """Sum source weights around every query for several Euclidean radii."""

    radii = tuple(float(radius) for radius in radii)
    if not radii:
        return np.empty((len(query_xy), 0), dtype=np.float32)
    if len(source_xy) == 0:
        return np.zeros((len(query_xy), len(radii)), dtype=np.float32)
    tree = cKDTree(np.asarray(source_xy, dtype=float))
    weights = np.asarray(source_values, dtype=float)
    output = np.zeros((len(query_xy), len(radii)), dtype=np.float32)
    for column, radius in enumerate(radii):
        neighborhoods = tree.query_ball_point(query_xy, radius)
        output[:, column] = np.fromiter(
            (weights[index].sum() for index in neighborhoods),
            dtype=np.float32,
            count=len(query_xy),
        )
    return output


def normalise_postcode(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)


def load_codepoint_coordinates(
    codepoint_dir: str | Path,
    needed_postcodes: set[str],
) -> pd.DataFrame:
    """Load only requested Code-Point Open postcodes from the split CSV archive."""

    if not needed_postcodes:
        return pd.DataFrame(columns=["postcode", "easting", "northing"])
    parts: list[pd.DataFrame] = []
    for path in sorted(Path(codepoint_dir).glob("*.csv")):
        part = pd.read_csv(
            path,
            header=None,
            usecols=[0, 2, 3],
            names=["postcode", "easting", "northing"],
            dtype={"postcode": str, "easting": float, "northing": float},
        )
        part["postcode"] = normalise_postcode(part["postcode"])
        selected = part[part["postcode"].isin(needed_postcodes)]
        if len(selected):
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=["postcode", "easting", "northing"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("postcode")


def load_active_nhs_coordinates(
    organisation_csv: str | Path,
    postcode_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """Geocode active NHS ODS organisation rows using Code-Point Open."""

    organisations = pd.read_csv(organisation_csv, header=None, dtype=str)
    organisations = organisations[organisations.iloc[:, 12].eq("ACTIVE")].copy()
    organisations["postcode"] = normalise_postcode(organisations.iloc[:, 9])
    organisations = organisations[organisations["postcode"].ne("")]
    organisations = organisations.drop_duplicates("postcode")
    return organisations[["postcode"]].merge(
        postcode_coordinates, on="postcode", how="inner", validate="one_to_one"
    )
