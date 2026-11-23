from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from sklearn.neighbors import NearestNeighbors

from gravity_gnn.data import load_graph_npz
from gravity_gnn.routing import k_nearest_network_sources


WALKING_METRES_PER_MINUTE = 5000.0 / 60.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def elongation(geometry) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon) or rectangle.is_empty:
        return 1.0
    coordinates = np.asarray(rectangle.exterior.coords)[:4]
    lengths = np.linalg.norm(coordinates - np.roll(coordinates, -1, axis=0), axis=1)
    positive = lengths[lengths > 1e-6]
    return float(positive.max() / positive.min()) if len(positive) else 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the real Camden graph bundle")
    parser.add_argument("--ahah", type=Path, required=True)
    parser.add_argument("--osm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pbf", type=Path, required=True)
    parser.add_argument("--max-buildings-per-lsoa", type=int, default=120)
    parser.add_argument("--building-neighbors", type=int, default=5)
    parser.add_argument("--poi-neighbors", type=int, default=4)
    parser.add_argument("--walk-limit-minutes", type=float, default=30.0)
    parser.add_argument("--spatial-block-metres", type=float, default=1000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_dir = args.output.parent

    boundary = gpd.read_parquet(args.osm_dir / "boundary.parquet").to_crs(27700)
    boundary_geometry = boundary.geometry.iloc[0]
    ahah = gpd.read_parquet(args.ahah)
    ahah = ahah[ahah["lsoa21cd"].notna()].copy()
    if ahah.crs.to_epsg() != 27700:
        ahah = ahah.to_crs(27700)

    buildings = gpd.read_parquet(args.osm_dir / "buildings_raw.parquet").to_crs(27700)
    buildings["anchor"] = buildings.geometry.representative_point()
    anchors = gpd.GeoDataFrame(
        buildings.drop(columns="geometry"), geometry="anchor", crs=27700
    )
    joined = gpd.sjoin(
        anchors,
        ahah[["lsoa21cd", "ahah", "ahah_pct", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns="index_right")
    buildings = buildings.join(joined[["lsoa21cd", "ahah", "ahah_pct"]])
    buildings = buildings.dropna(subset=["lsoa21cd", "ahah"]).copy()
    buildings["sample_order"] = buildings["osm_id"].map(stable_order)
    buildings = (
        buildings.sort_values(["lsoa21cd", "sample_order"])
        .groupby("lsoa21cd", group_keys=False)
        .head(args.max_buildings_per_lsoa)
        .reset_index(drop=True)
    )

    buildings["area_m2"] = buildings.geometry.area.clip(lower=1.0)
    buildings["perimeter_m"] = buildings.geometry.length.clip(lower=1.0)
    buildings["compactness"] = (
        4.0 * math.pi * buildings["area_m2"] / buildings["perimeter_m"].pow(2)
    ).clip(0.0, 1.0)
    buildings["convexity"] = (
        buildings["area_m2"] / buildings.geometry.convex_hull.area.clip(lower=1.0)
    ).clip(0.0, 1.0)
    buildings["elongation"] = buildings.geometry.map(elongation).clip(1.0, 20.0)

    known_levels = buildings["building_levels"].where(buildings["building_levels"] > 0)
    levels_from_height = buildings["height_m"].where(buildings["height_m"] > 0) / 3.0
    type_median = known_levels.groupby(buildings["building"]).transform("median")
    global_median = float(known_levels.median()) if known_levels.notna().any() else 2.0
    buildings["levels_missing"] = known_levels.isna().astype(float)
    buildings["levels_imputed"] = (
        known_levels.fillna(levels_from_height)
        .fillna(type_median)
        .fillna(global_median)
        .clip(1.0, 80.0)
    )
    buildings["height_missing"] = buildings["height_m"].isna().astype(float)
    buildings["height_imputed_m"] = (
        buildings["height_m"].fillna(3.0 * buildings["levels_imputed"]).clip(3.0, 300.0)
    )
    buildings["volume_m3"] = buildings["area_m2"] * buildings["height_imputed_m"]

    building_xy = np.column_stack((buildings["anchor"].x, buildings["anchor"].y))
    building_tree = cKDTree(building_xy)
    buildings["building_count_250m"] = building_tree.query_ball_point(
        building_xy, r=250.0, return_length=True
    ) - 1
    neighbor_model = NearestNeighbors(
        n_neighbors=min(args.building_neighbors + 1, len(buildings)), algorithm="kd_tree"
    ).fit(building_xy)
    neighbor_distance, neighbor_index = neighbor_model.kneighbors(building_xy)
    neighbor_distance = neighbor_distance[:, 1:]
    neighbor_index = neighbor_index[:, 1:]
    buildings["neighbor_volume_mean"] = buildings["volume_m3"].to_numpy()[
        neighbor_index
    ].mean(axis=1)

    roads = gpd.read_parquet(args.osm_dir / "walk_segments_raw.parquet").to_crs(27700)
    node_coordinates: dict[int, tuple[float, float]] = {}
    edge_lengths: dict[tuple[int, int], float] = {}
    for row in roads.itertuples(index=False):
        coordinates = list(row.geometry.coords)
        node_coordinates.setdefault(int(row.u), coordinates[0])
        node_coordinates.setdefault(int(row.v), coordinates[-1])
        key = tuple(sorted((int(row.u), int(row.v))))
        length = max(float(row.geometry.length), 0.1)
        edge_lengths[key] = min(edge_lengths.get(key, float("inf")), length)
    road_node_ids = np.array(sorted(node_coordinates), dtype=np.int64)
    road_node_lookup = {node_id: index for index, node_id in enumerate(road_node_ids)}
    road_xy = np.array([node_coordinates[int(node_id)] for node_id in road_node_ids])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (source, destination), length in edge_lengths.items():
        u = road_node_lookup[source]
        v = road_node_lookup[destination]
        rows.extend((u, v))
        columns.extend((v, u))
        values.extend((length, length))
    road_graph = csr_matrix(
        (values, (rows, columns)), shape=(len(road_node_ids), len(road_node_ids))
    )
    component_count, component_label = connected_components(road_graph, directed=False)
    road_tree = cKDTree(road_xy)
    building_snap_distance, building_road_index = road_tree.query(building_xy, k=1)
    buildings["road_snap_m"] = building_snap_distance
    road_degree = np.diff(road_graph.indptr)
    buildings["road_degree"] = road_degree[building_road_index]

    pois = gpd.read_parquet(args.osm_dir / "pois_raw.parquet").to_crs(27700)
    pois = pois[pois.geometry.within(boundary_geometry)].copy().reset_index(drop=True)
    poi_xy = np.column_stack((pois.geometry.x, pois.geometry.y))
    poi_snap_distance, poi_road_index = road_tree.query(poi_xy, k=1)
    pois["road_snap_m"] = poi_snap_distance
    pois["road_index"] = poi_road_index
    pois = pois[pois["road_snap_m"] <= 300.0].reset_index(drop=True)

    walk_limit_m = args.walk_limit_minutes * WALKING_METRES_PER_MINUTE
    category_edges: dict[str, list[tuple[int, int, float]]] = {}
    used_pois: set[int] = set()
    for category, category_pois in pois.groupby("category", sort=True):
        if len(category_pois) == 0:
            continue
        nearest_distance, nearest_poi = k_nearest_network_sources(
            road_graph,
            source_nodes=category_pois["road_index"].to_numpy(dtype=np.int64),
            source_ids=category_pois.index.to_numpy(dtype=np.int64),
            source_initial_cost=category_pois["road_snap_m"].to_numpy(dtype=float),
            k=args.poi_neighbors,
            limit=walk_limit_m,
        )
        records: list[tuple[int, int, float]] = []
        for building_index, road_index in enumerate(building_road_index):
            for network_distance, poi_index in zip(
                nearest_distance[int(road_index)],
                nearest_poi[int(road_index)],
                strict=True,
            ):
                if poi_index < 0 or not np.isfinite(network_distance):
                    continue
                total_distance = float(network_distance) + float(
                    building_snap_distance[building_index]
                )
                if total_distance > walk_limit_m:
                    continue
                records.append(
                    (
                        building_index,
                        int(poi_index),
                        total_distance / WALKING_METRES_PER_MINUTE,
                    )
                )
                used_pois.add(int(poi_index))
        if records:
            category_edges[category] = records

    used_poi_indices = sorted(used_pois)
    selected_pois = pois.loc[used_poi_indices].copy().reset_index().rename(
        columns={"index": "raw_poi_index"}
    )
    poi_node_lookup = {
        int(raw_index): index for index, raw_index in enumerate(selected_pois["raw_poi_index"])
    }
    relation_names = sorted(category_edges)
    relation_lookup = {name: index for index, name in enumerate(relation_names)}

    edge_source: list[int] = []
    edge_destination: list[int] = []
    edge_impedance: list[float] = []
    edge_type: list[int] = []
    number_of_buildings = len(buildings)
    edge_counts: dict[str, int] = {}
    for category in relation_names:
        count = 0
        for building_index, raw_poi_index, minutes in category_edges[category]:
            edge_source.append(number_of_buildings + poi_node_lookup[raw_poi_index])
            edge_destination.append(building_index)
            edge_impedance.append(minutes)
            edge_type.append(relation_lookup[category])
            count += 1
        edge_counts[category] = count

    proximity_relation = len(relation_names)
    relation_names.append("building_proximity")
    for building_index in range(number_of_buildings):
        for neighbor, distance_m in zip(
            neighbor_index[building_index], neighbor_distance[building_index], strict=False
        ):
            edge_source.append(int(neighbor))
            edge_destination.append(building_index)
            edge_impedance.append(float(distance_m) / WALKING_METRES_PER_MINUTE)
            edge_type.append(proximity_relation)
    edge_counts["building_proximity"] = int(neighbor_index.size)

    building_feature_names = [
        "log_footprint_area",
        "log_perimeter",
        "compactness",
        "elongation",
        "convexity",
        "levels_imputed",
        "levels_missing",
        "height_imputed_m",
        "height_missing",
        "log_volume",
        "building_count_250m",
        "log_neighbor_volume_mean",
        "road_degree",
        "road_snap_m",
    ]
    category_feature_names = [f"poi_{category}" for category in sorted(pois["category"].unique())]
    feature_names = building_feature_names + category_feature_names + [
        "node_is_building",
        "node_is_poi",
    ]
    total_nodes = number_of_buildings + len(selected_pois)
    x = np.zeros((total_nodes, len(feature_names)), dtype=np.float32)
    x[:number_of_buildings, : len(building_feature_names)] = np.column_stack(
        (
            np.log1p(buildings["area_m2"]),
            np.log1p(buildings["perimeter_m"]),
            buildings["compactness"],
            buildings["elongation"],
            buildings["convexity"],
            buildings["levels_imputed"],
            buildings["levels_missing"],
            buildings["height_imputed_m"],
            buildings["height_missing"],
            np.log1p(buildings["volume_m3"]),
            buildings["building_count_250m"],
            np.log1p(buildings["neighbor_volume_mean"]),
            buildings["road_degree"],
            buildings["road_snap_m"],
        )
    ).astype(np.float32)
    category_feature_lookup = {
        name.removeprefix("poi_"): len(building_feature_names) + index
        for index, name in enumerate(category_feature_names)
    }
    for poi_offset, category in enumerate(selected_pois["category"]):
        x[number_of_buildings + poi_offset, category_feature_lookup[category]] = 1.0
    x[:number_of_buildings, -2] = 1.0
    x[number_of_buildings:, -1] = 1.0

    building_mass = buildings["volume_m3"].to_numpy(dtype=np.float32)
    building_mass = building_mass / np.median(building_mass)
    poi_mass = np.ones(len(selected_pois), dtype=np.float32)
    node_mass = np.concatenate((building_mass, poi_mass))
    y = np.full(total_nodes, np.nan, dtype=np.float32)
    y[:number_of_buildings] = buildings["ahah"].to_numpy(dtype=np.float32)
    building_mask = np.zeros(total_nodes, dtype=bool)
    building_mask[:number_of_buildings] = True

    lsoa_codes = sorted(buildings["lsoa21cd"].unique())
    lsoa_lookup = {code: index for index, code in enumerate(lsoa_codes)}
    label_group = np.full(total_nodes, -1, dtype=np.int64)
    label_group[:number_of_buildings] = buildings["lsoa21cd"].map(lsoa_lookup).to_numpy()
    lsoa_geometry = ahah.set_index("lsoa21cd").loc[lsoa_codes].geometry.representative_point()
    block_x = np.floor(lsoa_geometry.x / args.spatial_block_metres).astype(int)
    block_y = np.floor(lsoa_geometry.y / args.spatial_block_metres).astype(int)
    block_keys = list(zip(block_x, block_y, strict=False))
    unique_blocks = {key: index for index, key in enumerate(sorted(set(block_keys)))}
    lsoa_to_block = {
        code: unique_blocks[key] for code, key in zip(lsoa_codes, block_keys, strict=False)
    }
    group = np.full(total_nodes, -1, dtype=np.int64)
    group[:number_of_buildings] = buildings["lsoa21cd"].map(lsoa_to_block).to_numpy()
    counts_by_lsoa = buildings["lsoa21cd"].value_counts()
    weights = buildings["lsoa21cd"].map(lambda code: 1.0 / counts_by_lsoa[code]).to_numpy()
    weights = weights / weights.mean()
    sample_weight = np.zeros(total_nodes, dtype=np.float32)
    sample_weight[:number_of_buildings] = weights.astype(np.float32)

    buildings["node_index"] = np.arange(number_of_buildings)
    buildings["label_group"] = label_group[:number_of_buildings]
    buildings["spatial_block"] = group[:number_of_buildings]
    selected_pois["node_index"] = number_of_buildings + np.arange(len(selected_pois))
    selected_lsoas = ahah[ahah["lsoa21cd"].isin(lsoa_codes)].copy()
    buildings.drop(columns="anchor").to_parquet(
        output_dir / "camden_buildings_model.parquet", index=False
    )
    selected_pois.to_parquet(output_dir / "camden_pois_model.parquet", index=False)
    selected_lsoas.to_parquet(output_dir / "camden_lsoas.parquet", index=False)

    np.savez_compressed(
        args.output,
        x=x,
        node_mass=node_mass.astype(np.float32),
        edge_index=np.asarray([edge_source, edge_destination], dtype=np.int64),
        edge_impedance=np.asarray(edge_impedance, dtype=np.float32),
        edge_type=np.asarray(edge_type, dtype=np.int64),
        y=y,
        building_mask=building_mask,
        group=group,
        label_group=label_group,
        sample_weight=sample_weight,
        feature_names=np.asarray(feature_names),
        relation_names=np.asarray(relation_names),
    )
    validated = load_graph_npz(args.output)

    summary = {
        "graph_path": str(args.output.resolve()),
        "nodes": validated.num_nodes,
        "buildings": number_of_buildings,
        "pois": len(selected_pois),
        "edges": validated.num_edges,
        "lsoas": len(lsoa_codes),
        "spatial_blocks": len(unique_blocks),
        "road_nodes": len(road_node_ids),
        "road_undirected_edges": len(edge_lengths),
        "road_components": int(component_count),
        "largest_road_component_nodes": int(np.bincount(component_label).max()),
        "walk_limit_minutes": args.walk_limit_minutes,
        "poi_neighbors": args.poi_neighbors,
        "max_buildings_per_lsoa": args.max_buildings_per_lsoa,
        "edge_counts": edge_counts,
        "features": feature_names,
        "relations": relation_names,
        "building_level_known_fraction": float(buildings["building_levels"].notna().mean()),
        "building_height_known_fraction": float(buildings["height_m"].notna().mean()),
        "building_road_snap_m": {
            "median": float(np.median(building_snap_distance)),
            "p95": float(np.quantile(building_snap_distance, 0.95)),
            "max": float(np.max(building_snap_distance)),
        },
        "ahah": {
            "mean": float(buildings["ahah"].mean()),
            "std": float(buildings["ahah"].std()),
            "min": float(buildings["ahah"].min()),
            "max": float(buildings["ahah"].max()),
        },
        "sha256": {
            "ahah_geoparquet": sha256(args.ahah),
            "source_pbf": sha256(args.source_pbf),
            "graph_npz": sha256(args.output),
        },
    }
    summary_path = output_dir / "camden_graph_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
