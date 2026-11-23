from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import osmium
from shapely import from_wkb, make_valid
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from gravity_gnn.geo_osm import is_walkable_highway, parse_numeric_tag, poi_category


def decode_wkb(value: str):
    geometry = from_wkb(bytes.fromhex(value))
    return make_valid(geometry) if not geometry.is_valid else geometry


class BoundaryHandler(osmium.SimpleHandler):
    def __init__(self, name: str, admin_level: str) -> None:
        super().__init__()
        self.name = name
        self.admin_level = admin_level
        self.factory = osmium.geom.WKBFactory()
        self.geometry = None
        self.osm_id = None

    def area(self, area: osmium.osm.Area) -> None:
        if area.tags.get("name") != self.name:
            return
        if area.tags.get("boundary") != "administrative":
            return
        if area.tags.get("admin_level") != self.admin_level:
            return
        self.geometry = decode_wkb(self.factory.create_multipolygon(area))
        kind = "way" if area.from_way() else "relation"
        self.osm_id = f"{kind}/{area.orig_id()}"


class FeatureHandler(osmium.SimpleHandler):
    def __init__(self, boundary, network_boundary) -> None:
        super().__init__()
        self.factory = osmium.geom.WKBFactory()
        self.boundary = prep(boundary)
        self.network_boundary = prep(network_boundary)
        self.buildings: list[dict] = []
        self.pois: list[dict] = []
        self.roads: list[dict] = []

    @staticmethod
    def _tags(osm_object) -> dict[str, str]:
        return {tag.k: tag.v for tag in osm_object.tags}

    def area(self, area: osmium.osm.Area) -> None:
        tags = self._tags(area)
        category = poi_category(tags)
        if not tags.get("building") and category is None:
            return
        try:
            geometry = decode_wkb(self.factory.create_multipolygon(area))
        except Exception:
            return
        if geometry.is_empty:
            return
        anchor = geometry.representative_point()
        if not self.boundary.covers(anchor):
            return
        kind = "w" if area.from_way() else "r"
        osm_id = f"{kind}{area.orig_id()}"
        if tags.get("building"):
            self.buildings.append(
                {
                    "osm_id": osm_id,
                    "building": tags.get("building", ""),
                    "building_levels": parse_numeric_tag(tags.get("building:levels")),
                    "height_m": parse_numeric_tag(tags.get("height")),
                    "roof_levels": parse_numeric_tag(tags.get("roof:levels")),
                    "name": tags.get("name", ""),
                    "geometry": geometry,
                }
            )
        if category is not None:
            self.pois.append(
                {
                    "osm_id": osm_id,
                    "category": category,
                    "name": tags.get("name", ""),
                    "source_geometry": "area",
                    "geometry": anchor,
                    "area_geometry": geometry,
                }
            )

    def node(self, node: osmium.osm.Node) -> None:
        tags = self._tags(node)
        category = poi_category(tags)
        if category is None or not node.location.valid():
            return
        point = Point(node.location.lon, node.location.lat)
        if not self.boundary.covers(point):
            return
        self.pois.append(
            {
                "osm_id": f"n{node.id}",
                "category": category,
                "name": tags.get("name", ""),
                "source_geometry": "node",
                "geometry": point,
                "area_geometry": None,
            }
        )

    def way(self, way: osmium.osm.Way) -> None:
        tags = self._tags(way)
        if not is_walkable_highway(tags):
            return
        valid_nodes = [node for node in way.nodes if node.location.valid()]
        for index, (source, destination) in enumerate(
            zip(valid_nodes[:-1], valid_nodes[1:], strict=False)
        ):
            if source.ref == destination.ref:
                continue
            line = LineString(
                [
                    (source.location.lon, source.location.lat),
                    (destination.location.lon, destination.location.lat),
                ]
            )
            if not self.network_boundary.intersects(line):
                continue
            self.roads.append(
                {
                    "segment_id": f"w{way.id}:{index}",
                    "way_id": way.id,
                    "u": source.ref,
                    "v": destination.ref,
                    "highway": tags.get("highway", ""),
                    "name": tags.get("name", ""),
                    "foot": tags.get("foot", ""),
                    "access": tags.get("access", ""),
                    "geometry": line,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a fixed Camden OSM pilot")
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-name", default="London Borough of Camden")
    parser.add_argument("--admin-level", default="8")
    parser.add_argument("--network-buffer-metres", type=float, default=1000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    boundary_handler = BoundaryHandler(args.area_name, args.admin_level)
    boundary_handler.apply_file(args.pbf, locations=True, idx="flex_mem")
    if boundary_handler.geometry is None:
        raise RuntimeError(f"administrative boundary not found: {args.area_name}")
    boundary = boundary_handler.geometry
    boundary_gdf = gpd.GeoDataFrame(
        [{"name": args.area_name, "osm_id": boundary_handler.osm_id, "geometry": boundary}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    network_boundary = (
        boundary_gdf.to_crs(27700)
        .buffer(args.network_buffer_metres)
        .to_crs(4326)
        .iloc[0]
    )

    handler = FeatureHandler(boundary, network_boundary)
    handler.apply_file(args.pbf, locations=True, idx="flex_mem")

    buildings = gpd.GeoDataFrame(handler.buildings, geometry="geometry", crs="EPSG:4326")
    pois = gpd.GeoDataFrame(handler.pois, geometry="geometry", crs="EPSG:4326")
    roads = gpd.GeoDataFrame(handler.roads, geometry="geometry", crs="EPSG:4326")
    if buildings.empty or roads.empty:
        raise RuntimeError("OSM extraction unexpectedly produced no buildings or roads")

    boundary_gdf.to_parquet(args.output / "boundary.parquet", index=False)
    buildings.to_parquet(args.output / "buildings_raw.parquet", index=False)
    pois.drop(columns="area_geometry").to_parquet(args.output / "pois_raw.parquet", index=False)
    roads.to_parquet(args.output / "walk_segments_raw.parquet", index=False)

    summary = {
        "source_pbf": str(args.pbf.resolve()),
        "area_name": args.area_name,
        "boundary_osm_id": boundary_handler.osm_id,
        "network_buffer_metres": args.network_buffer_metres,
        "buildings": len(buildings),
        "pois": len(pois),
        "walk_segments": len(roads),
        "poi_categories": pois["category"].value_counts().sort_index().to_dict(),
    }
    (args.output / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
