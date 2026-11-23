from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import STRtree, from_wkb, get_parts, transform

from gravity_gnn.case_studies import sha256
from gravity_gnn.context import radius_counts


WATERWAY_LINES = {"river", "canal", "stream", "drain", "ditch", "dock"}


def is_water_polygon(tags: dict[str, str]) -> bool:
    return (
        tags.get("natural") == "water"
        or tags.get("waterway") == "riverbank"
        or tags.get("landuse") in {"reservoir", "basin"}
        or tags.get("water")
        in {"reservoir", "basin", "lake", "pond", "river", "canal"}
    )


class BlueGeometryHandler(osmium.SimpleHandler):
    def __init__(self, minimum_water_area_m2: float) -> None:
        super().__init__()
        self.minimum_water_area_m2 = float(minimum_water_area_m2)
        self.factory = osmium.geom.WKBFactory()
        self.transformer = Transformer.from_crs(4326, 27700, always_xy=True)
        self.polygons = []
        self.lines = []

    @staticmethod
    def tags(osm_object) -> dict[str, str]:
        return {tag.k: tag.v for tag in osm_object.tags}

    def project(self, geometry):
        return transform(geometry, self.transformer.transform, interleaved=False)

    def area(self, area: osmium.osm.Area) -> None:
        if not is_water_polygon(self.tags(area)):
            return
        try:
            geometry = self.project(
                from_wkb(bytes.fromhex(self.factory.create_multipolygon(area)))
            )
        except Exception:
            return
        if not geometry.is_empty and geometry.area >= self.minimum_water_area_m2:
            self.polygons.append(geometry)

    def way(self, way: osmium.osm.Way) -> None:
        tags = self.tags(way)
        if tags.get("waterway") not in WATERWAY_LINES and tags.get("natural") != "coastline":
            return
        try:
            geometry = self.project(
                from_wkb(bytes.fromhex(self.factory.create_linestring(way)))
            )
        except Exception:
            return
        if not geometry.is_empty and geometry.length > 0:
            self.lines.append(geometry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract exact target-free OSM blue-space geometry context"
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--minimum-water-area-m2", type=float, default=7500.0)
    parser.add_argument("--sample-spacing-m", type=float, default=250.0)
    parser.add_argument("--radii", default="1000,3000,5000")
    return parser


def sample_lines(geometries, spacing: float) -> np.ndarray:
    records = []
    for geometry in geometries:
        for part in get_parts(geometry):
            if part.geom_type not in {"LineString", "LinearRing"} or part.length <= 0:
                continue
            for distance in np.arange(0.0, part.length + spacing, spacing):
                point = part.interpolate(min(float(distance), part.length))
                records.append((point.x, point.y))
    return np.asarray(records, dtype=float).reshape(-1, 2)


def exact_nearest(points, geometries) -> np.ndarray:
    if not geometries:
        return np.full(len(points), 100_000.0)
    tree = STRtree(geometries)
    return np.asarray(
        [point.distance(geometries[int(tree.nearest(point))]) for point in points],
        dtype=float,
    )


def main() -> None:
    args = build_parser().parse_args()
    radii = tuple(float(value) for value in args.radii.split(",") if value.strip())
    handler = BlueGeometryHandler(args.minimum_water_area_m2)
    handler.apply_file(str(args.pbf), locations=True, idx="flex_mem")

    case_dir = args.case_root / args.study
    lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
    lsoa_points = list(lsoas.geometry.representative_point())
    query_xy = np.asarray([(point.x, point.y) for point in lsoa_points], dtype=float)
    polygons = gpd.GeoDataFrame(
        {"geometry": handler.polygons}, geometry="geometry", crs=27700
    )
    lines = gpd.GeoDataFrame({"geometry": handler.lines}, geometry="geometry", crs=27700)

    output = lsoas[["lsoa21cd", "geometry"]].copy()
    all_geometries = handler.polygons + handler.lines
    output["log_osm_blue_exact_nearest_m"] = np.log1p(
        exact_nearest(lsoa_points, all_geometries)
    )
    coverage = np.zeros(len(lsoas), dtype=float)
    line_length = np.zeros(len(lsoas), dtype=float)
    polygon_index = polygons.sindex if len(polygons) else None
    line_index = lines.sindex if len(lines) else None
    for index, geometry in enumerate(lsoas.geometry):
        if polygon_index is not None:
            candidates = polygons.iloc[
                list(polygon_index.query(geometry, predicate="intersects"))
            ]
            if len(candidates):
                coverage[index] = min(
                    float(candidates.geometry.intersection(geometry).area.sum())
                    / max(float(geometry.area), 1.0),
                    1.0,
                )
        if line_index is not None:
            candidates = lines.iloc[
                list(line_index.query(geometry, predicate="intersects"))
            ]
            if len(candidates):
                line_length[index] = float(
                    candidates.geometry.intersection(geometry).length.sum()
                )
    output["osm_blue_water_coverage_fraction"] = coverage
    output["log_osm_blue_line_length_per_km2"] = np.log1p(
        line_length / np.maximum(lsoas.geometry.area.to_numpy(dtype=float) / 1e6, 1e-4)
    )

    boundary_lines = [polygon.boundary for polygon in handler.polygons]
    sample_xy = sample_lines(handler.lines + boundary_lines, args.sample_spacing_m)
    counts = radius_counts(query_xy, sample_xy, radii)
    feature_names = [
        "log_osm_blue_exact_nearest_m",
        "osm_blue_water_coverage_fraction",
        "log_osm_blue_line_length_per_km2",
    ]
    for column, radius in enumerate(radii):
        name = f"log_osm_blue_sample_count_{int(radius)}m"
        output[name] = np.log1p(counts[:, column])
        feature_names.append(name)

    for scale in radii:
        if len(sample_xy):
            distances, _ = cKDTree(sample_xy).query(query_xy, k=min(256, len(sample_xy)))
            if distances.ndim == 1:
                distances = distances[:, None]
            potential = np.log1p(np.exp(-distances / scale).sum(axis=1))
        else:
            potential = np.zeros(len(query_xy))
        name = f"log_osm_blue_gravity_potential_{int(scale)}m"
        output[name] = potential
        feature_names.append(name)

    output_path = case_dir / "osm_blue_context.parquet"
    output.to_parquet(output_path, index=False)
    summary = {
        "study": args.study,
        "source_pbf": str(args.pbf.resolve()),
        "minimum_water_polygon_area_m2": args.minimum_water_area_m2,
        "water_polygons": len(handler.polygons),
        "waterway_lines": len(handler.lines),
        "sample_spacing_m": args.sample_spacing_m,
        "sample_points": len(sample_xy),
        "features": feature_names,
        "target_derived_features": False,
        "output": str(output_path.resolve()),
        "sha256": {
            "source_pbf": sha256(args.pbf),
            "output": sha256(output_path),
        },
    }
    (case_dir / "osm_blue_context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
