from __future__ import annotations

import hashlib
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
from shapely import from_wkb, make_valid
from sklearn.cluster import KMeans


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_wkb(value: str):
    geometry = from_wkb(bytes.fromhex(value))
    return make_valid(geometry) if not geometry.is_valid else geometry


class AdministrativeBoundaryHandler(osmium.SimpleHandler):
    """Read one exact administrative boundary from a regional OSM PBF."""

    def __init__(self, name: str, admin_level: str) -> None:
        super().__init__()
        self.name = name
        self.admin_level = str(admin_level)
        self.factory = osmium.geom.WKBFactory()
        self.geometry = None
        self.osm_id: str | None = None

    def area(self, area: osmium.osm.Area) -> None:
        if area.tags.get("name") != self.name:
            return
        if area.tags.get("boundary") != "administrative":
            return
        if area.tags.get("admin_level") != self.admin_level:
            return
        self.geometry = _decode_wkb(self.factory.create_multipolygon(area))
        kind = "way" if area.from_way() else "relation"
        self.osm_id = f"{kind}/{area.orig_id()}"


def load_administrative_boundary(
    pbf: str | Path,
    area_name: str,
    admin_level: str,
    target_crs: str = "EPSG:27700",
) -> gpd.GeoDataFrame:
    handler = AdministrativeBoundaryHandler(area_name, admin_level)
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")
    if handler.geometry is None:
        raise ValueError(
            f"administrative boundary not found: name={area_name!r}, "
            f"admin_level={admin_level!r}"
        )
    boundary = gpd.GeoDataFrame(
        [
            {
                "name": area_name,
                "admin_level": str(admin_level),
                "osm_id": handler.osm_id,
                "geometry": handler.geometry,
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    return boundary.to_crs(target_crs)


def select_lsoas(
    ahah: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if ahah.crs is None or boundary.crs is None:
        raise ValueError("AHAH and boundary geometries must have a CRS")
    if ahah.crs != boundary.crs:
        ahah = ahah.to_crs(boundary.crs)
    candidates = ahah[ahah["lsoa21cd"].notna()].copy()
    representative = candidates.geometry.representative_point()
    selected = candidates[representative.within(boundary.geometry.iloc[0])].copy()
    if selected.empty:
        raise ValueError("case-study boundary selected no AHAH LSOAs")
    if selected["lsoa21cd"].duplicated().any():
        raise ValueError("case-study AHAH selection contains duplicate LSOA codes")
    return selected.sort_values("lsoa21cd").reset_index(drop=True)


def make_spatial_fold_plan(
    lsoas: gpd.GeoDataFrame,
    folds: int = 5,
    seed: int = 20260826,
) -> dict:
    """Create compact spatial test folds and a deterministic adjacent validation rule."""

    if folds < 3:
        raise ValueError("at least three folds are required")
    if len(lsoas) < folds:
        raise ValueError("number of LSOAs must be at least the number of folds")
    points = lsoas.geometry.representative_point()
    coordinates = np.column_stack((points.x, points.y))
    model = KMeans(n_clusters=folds, random_state=seed, n_init=20)
    raw_fold = model.fit_predict(coordinates)
    centres = np.asarray(model.cluster_centers_)
    overall = coordinates.mean(axis=0)
    angles = np.arctan2(centres[:, 1] - overall[1], centres[:, 0] - overall[0])
    ordered_raw = np.argsort(angles)
    fold_lookup = {int(raw): int(index) for index, raw in enumerate(ordered_raw)}
    assigned = np.asarray([fold_lookup[int(value)] for value in raw_fold], dtype=int)

    lsoa_codes = lsoas["lsoa21cd"].astype(str).to_numpy()
    fold_rows = []
    for test_fold in range(folds):
        validation_fold = (test_fold + 1) % folds
        fold_rows.append(
            {
                "fold": test_fold,
                "test_lsoas": int(np.sum(assigned == test_fold)),
                "validation_lsoas": int(np.sum(assigned == validation_fold)),
                "train_lsoas": int(np.sum((assigned != test_fold) & (assigned != validation_fold))),
                "test_centroid_x": float(coordinates[assigned == test_fold, 0].mean()),
                "test_centroid_y": float(coordinates[assigned == test_fold, 1].mean()),
            }
        )

    return {
        "method": "kmeans_on_lsoa_representative_points",
        "crs": str(lsoas.crs),
        "folds": folds,
        "seed": seed,
        "fold_id_order": "centroid_angle_ascending",
        "validation_rule": "next_clockwise_fold",
        "lsoa_to_fold": {
            code: int(fold) for code, fold in zip(lsoa_codes, assigned, strict=True)
        },
        "fold_statistics": fold_rows,
    }


def boundary_area_km2(boundary: gpd.GeoDataFrame) -> float:
    return float(boundary.geometry.area.sum() / math.pow(1000.0, 2))
