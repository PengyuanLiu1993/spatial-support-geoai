from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import Point

from gravity_gnn.case_studies import sha256
from gravity_gnn.geo_osm import is_walkable_highway, poi_category


class FacilityNetworkHandler(osmium.SimpleHandler):
    """Stream walkable road segments and target-independent POIs from a PBF."""

    def __init__(self, lonlat_bounds: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.lonlat_bounds = lonlat_bounds
        self.factory = osmium.geom.WKBFactory()
        self.road_coordinates: dict[int, tuple[float, float]] = {}
        self.road_edges: list[tuple[int, int]] = []
        self.facilities: list[dict[str, object]] = []

    @staticmethod
    def tags(osm_object) -> dict[str, str]:
        return {tag.k: tag.v for tag in osm_object.tags}

    def in_bounds(self, lon: float, lat: float) -> bool:
        min_lon, min_lat, max_lon, max_lat = self.lonlat_bounds
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

    def node(self, node: osmium.osm.Node) -> None:
        if not node.location.valid():
            return
        category = poi_category(self.tags(node))
        if category is None or not self.in_bounds(node.location.lon, node.location.lat):
            return
        self.facilities.append(
            {
                "facility_id": f"n{node.id}",
                "category": category,
                "lon": float(node.location.lon),
                "lat": float(node.location.lat),
                "source_geometry": "node",
            }
        )

    def area(self, area: osmium.osm.Area) -> None:
        category = poi_category(self.tags(area))
        if category is None:
            return
        try:
            geometry = from_wkb(bytes.fromhex(self.factory.create_multipolygon(area)))
        except Exception:
            return
        if geometry.is_empty:
            return
        anchor = geometry.representative_point()
        if not self.in_bounds(anchor.x, anchor.y):
            return
        kind = "w" if area.from_way() else "r"
        self.facilities.append(
            {
                "facility_id": f"{kind}{area.orig_id()}",
                "category": category,
                "lon": float(anchor.x),
                "lat": float(anchor.y),
                "source_geometry": "area",
            }
        )

    def way(self, way: osmium.osm.Way) -> None:
        if not is_walkable_highway(self.tags(way)):
            return
        valid_nodes = [node for node in way.nodes if node.location.valid()]
        for source, destination in zip(valid_nodes[:-1], valid_nodes[1:], strict=False):
            if source.ref == destination.ref:
                continue
            source_lonlat = (float(source.location.lon), float(source.location.lat))
            destination_lonlat = (
                float(destination.location.lon),
                float(destination.location.lat),
            )
            if not (
                self.in_bounds(*source_lonlat)
                or self.in_bounds(*destination_lonlat)
            ):
                continue
            self.road_coordinates.setdefault(int(source.ref), source_lonlat)
            self.road_coordinates.setdefault(int(destination.ref), destination_lonlat)
            self.road_edges.append((int(source.ref), int(destination.ref)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the walk network and OSM facilities for a case study"
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--buffer-metres", type=float, default=5000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.buffer_metres < 0:
        raise ValueError("buffer-metres must be non-negative")
    case_dir = args.case_root / args.study
    lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
    buffered_bounds = gpd.GeoSeries(
        [lsoas.geometry.union_all().envelope.buffer(args.buffer_metres)], crs=27700
    ).to_crs(4326).total_bounds
    handler = FacilityNetworkHandler(tuple(float(value) for value in buffered_bounds))
    handler.apply_file(str(args.pbf), locations=True, idx="flex_mem")
    if not handler.road_edges or not handler.facilities:
        raise RuntimeError("OSM extraction produced an empty network or facility set")

    road_ids = np.asarray(sorted(handler.road_coordinates), dtype=np.int64)
    road_lookup = {int(node_id): index for index, node_id in enumerate(road_ids)}
    lon = np.fromiter(
        (handler.road_coordinates[int(node_id)][0] for node_id in road_ids),
        dtype=float,
        count=len(road_ids),
    )
    lat = np.fromiter(
        (handler.road_coordinates[int(node_id)][1] for node_id in road_ids),
        dtype=float,
        count=len(road_ids),
    )
    transformer = Transformer.from_crs(4326, 27700, always_xy=True)
    road_x, road_y = transformer.transform(lon, lat)
    edge_u = np.fromiter(
        (road_lookup[source] for source, _ in handler.road_edges),
        dtype=np.int64,
        count=len(handler.road_edges),
    )
    edge_v = np.fromiter(
        (road_lookup[destination] for _, destination in handler.road_edges),
        dtype=np.int64,
        count=len(handler.road_edges),
    )
    edge_length = np.hypot(
        np.asarray(road_x)[edge_u] - np.asarray(road_x)[edge_v],
        np.asarray(road_y)[edge_u] - np.asarray(road_y)[edge_v],
    ).clip(min=0.1)
    network_path = case_dir / "walk_network.npz"
    np.savez_compressed(
        network_path,
        road_node_id=road_ids,
        road_x=np.asarray(road_x, dtype=np.float32),
        road_y=np.asarray(road_y, dtype=np.float32),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_length_m=edge_length.astype(np.float32),
    )

    facility_lon = np.asarray([row["lon"] for row in handler.facilities], dtype=float)
    facility_lat = np.asarray([row["lat"] for row in handler.facilities], dtype=float)
    facility_x, facility_y = transformer.transform(facility_lon, facility_lat)
    facilities = gpd.GeoDataFrame(
        {
            "facility_id": [str(row["facility_id"]) for row in handler.facilities],
            "category": [str(row["category"]) for row in handler.facilities],
            "source_geometry": [
                str(row["source_geometry"]) for row in handler.facilities
            ],
        },
        geometry=gpd.points_from_xy(facility_x, facility_y),
        crs=27700,
    )
    facilities["round_x"] = np.round(facilities.geometry.x / 5.0).astype(np.int64)
    facilities["round_y"] = np.round(facilities.geometry.y / 5.0).astype(np.int64)
    facilities = (
        facilities.sort_values(["category", "facility_id"])
        .drop_duplicates(["category", "round_x", "round_y"])
        .drop(columns=["round_x", "round_y"])
        .reset_index(drop=True)
    )
    facilities_path = case_dir / "osm_facilities.parquet"
    facilities.to_parquet(facilities_path, index=False)

    edge_counts: dict[str, int] = defaultdict(int)
    for category in facilities["category"]:
        edge_counts[str(category)] += 1
    summary = {
        "study": args.study,
        "source_pbf": str(args.pbf.resolve()),
        "buffer_metres": args.buffer_metres,
        "road_nodes": int(len(road_ids)),
        "road_segments": int(len(edge_u)),
        "osm_facilities": int(len(facilities)),
        "facility_categories": dict(sorted(edge_counts.items())),
        "outputs": {
            "network": str(network_path.resolve()),
            "facilities": str(facilities_path.resolve()),
        },
        "sha256": {
            "source_pbf": sha256(args.pbf),
            "network": sha256(network_path),
            "facilities": sha256(facilities_path),
        },
    }
    (case_dir / "facility_network_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
