from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import from_wkb
from shapely.geometry import Point

from gravity_gnn.case_studies import sha256
from gravity_gnn.context import radius_counts
from gravity_gnn.geo_osm import parse_numeric_tag, poi_category


POI_GROUPS = {
    "primary_care": {"clinic_doctor"},
    "dentist": {"dentist"},
    "pharmacy": {"pharmacy"},
    "hospital": {"hospital"},
    "leisure": {"sport_fitness"},
    "active_green": {"greenspace", "sport_fitness"},
    "green_blue": {"greenspace", "green_blue_space"},
    "fast_food": {"fast_food"},
    "gambling": {"gambling"},
    "pub_bar": {"pub_bar"},
    "tobacco": {"tobacco"},
    "food_retail": {"supermarket", "convenience"},
    "social_food": {"restaurant_cafe"},
    "education": {"school", "university"},
}


class PolygonGridLocator:
    """Fast exact point-in-LSOA lookup using a coarse in-memory grid."""

    def __init__(self, polygons: gpd.GeoSeries, cell_size: float = 2000.0) -> None:
        self.polygons = list(polygons)
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, geometry in enumerate(self.polygons):
            min_x, min_y, max_x, max_y = geometry.bounds
            for cell_x in range(
                math.floor(min_x / self.cell_size), math.floor(max_x / self.cell_size) + 1
            ):
                for cell_y in range(
                    math.floor(min_y / self.cell_size), math.floor(max_y / self.cell_size) + 1
                ):
                    self.cells[(cell_x, cell_y)].append(index)

    def locate(self, x: float, y: float) -> int | None:
        key = (math.floor(x / self.cell_size), math.floor(y / self.cell_size))
        point = Point(x, y)
        for index in self.cells.get(key, ()):
            if self.polygons[index].covers(point):
                return index
        return None


class ContextHandler(osmium.SimpleHandler):
    def __init__(self, locator: PolygonGridLocator, lsoa_count: int) -> None:
        super().__init__()
        self.locator = locator
        self.transformer = Transformer.from_crs(4326, 27700, always_xy=True)
        self.factory = osmium.geom.WKBFactory()
        self.building_count = np.zeros(lsoa_count, dtype=np.int64)
        self.known_levels_count = np.zeros(lsoa_count, dtype=np.int64)
        self.levels_sum = np.zeros(lsoa_count, dtype=float)
        self.known_height_count = np.zeros(lsoa_count, dtype=np.int64)
        self.height_sum = np.zeros(lsoa_count, dtype=float)
        self.poi_xy: dict[str, list[tuple[float, float]]] = defaultdict(list)

    @staticmethod
    def tags(osm_object) -> dict[str, str]:
        return {tag.k: tag.v for tag in osm_object.tags}

    def projected_location(self, lon: float, lat: float) -> tuple[float, float, int | None]:
        x, y = self.transformer.transform(lon, lat)
        return x, y, self.locator.locate(x, y)

    def area(self, area: osmium.osm.Area) -> None:
        tags = self.tags(area)
        category = poi_category(tags)
        is_building = bool(tags.get("building"))
        if not is_building and category is None:
            return
        try:
            geometry = from_wkb(bytes.fromhex(self.factory.create_multipolygon(area)))
        except Exception:
            return
        if geometry.is_empty:
            return
        anchor = geometry.representative_point()
        x, y, index = self.projected_location(anchor.x, anchor.y)
        if index is None:
            return
        if is_building:
            self.building_count[index] += 1
            levels = parse_numeric_tag(tags.get("building:levels"))
            if levels is not None:
                self.known_levels_count[index] += 1
                self.levels_sum[index] += levels
            height = parse_numeric_tag(tags.get("height"))
            if height is not None:
                self.known_height_count[index] += 1
                self.height_sum[index] += height
        if category is not None:
            self.poi_xy[category].append((x, y))

    def node(self, node: osmium.osm.Node) -> None:
        category = poi_category(self.tags(node))
        if category is None or not node.location.valid():
            return
        x, y, index = self.projected_location(node.location.lon, node.location.lat)
        if index is not None:
            self.poi_xy[category].append((x, y))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream target-free building and POI context from a regional OSM PBF"
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--radii", default="1000,3000,5000")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    radii = tuple(float(value) for value in args.radii.split(",") if value.strip())
    case_dir = args.case_root / args.study
    lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
    points = lsoas.geometry.representative_point()
    query_xy = np.column_stack((points.x, points.y))
    handler = ContextHandler(PolygonGridLocator(lsoas.geometry), len(lsoas))
    handler.apply_file(str(args.pbf), locations=True, idx="flex_mem")

    output = lsoas[["lsoa21cd", "geometry"]].copy()
    area_km2 = np.maximum(lsoas.geometry.area.to_numpy(dtype=float) / 1_000_000.0, 1e-4)
    output["log_osm_building_count"] = np.log1p(handler.building_count)
    output["log_osm_buildings_per_km2"] = np.log1p(handler.building_count / area_km2)
    output["osm_levels_known_fraction"] = handler.known_levels_count / np.maximum(
        handler.building_count, 1
    )
    output["osm_levels_known_mean"] = handler.levels_sum / np.maximum(
        handler.known_levels_count, 1
    )
    output["osm_height_known_fraction"] = handler.known_height_count / np.maximum(
        handler.building_count, 1
    )
    output["osm_height_known_mean"] = handler.height_sum / np.maximum(
        handler.known_height_count, 1
    )
    feature_names = [
        "log_osm_building_count",
        "log_osm_buildings_per_km2",
        "osm_levels_known_fraction",
        "osm_levels_known_mean",
        "osm_height_known_fraction",
        "osm_height_known_mean",
    ]

    all_points = {
        category: np.asarray(values, dtype=float).reshape(-1, 2)
        for category, values in handler.poi_xy.items()
    }
    for group_name, categories in POI_GROUPS.items():
        selected = [all_points[name] for name in categories if name in all_points]
        source_xy = np.concatenate(selected, axis=0) if selected else np.empty((0, 2))
        if len(source_xy):
            nearest, _ = cKDTree(source_xy).query(query_xy, k=1)
        else:
            nearest = np.full(len(lsoas), 100_000.0)
        counts = radius_counts(query_xy, source_xy, radii)
        nearest_name = f"log_osm_{group_name}_nearest_m"
        output[nearest_name] = np.log1p(nearest)
        feature_names.append(nearest_name)
        for column, radius in enumerate(radii):
            name = f"log_osm_{group_name}_count_{int(radius)}m"
            output[name] = np.log1p(counts[:, column])
            feature_names.append(name)

    output_path = case_dir / "osm_context.parquet"
    output.to_parquet(output_path, index=False)
    summary = {
        "study": args.study,
        "source_pbf": str(args.pbf.resolve()),
        "lsoas": int(len(lsoas)),
        "buildings": int(handler.building_count.sum()),
        "poi_categories": {
            name: int(len(values)) for name, values in sorted(handler.poi_xy.items())
        },
        "features": feature_names,
        "output": str(output_path.resolve()),
        "sha256": {"source_pbf": sha256(args.pbf), "output": sha256(output_path)},
    }
    (case_dir / "osm_context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
