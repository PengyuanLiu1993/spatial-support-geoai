from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from gravity_gnn.case_studies import sha256
from gravity_gnn.context import (
    load_active_nhs_coordinates,
    load_codepoint_coordinates,
    normalise_postcode,
)
from gravity_gnn.data import load_graph_npz
from gravity_gnn.routing import k_nearest_network_sources


WALKING_METRES_PER_MINUTE = 80.0
ACTIVE_GREEN_FUNCTIONS = {
    "Play Space",
    "Playing Field",
    "Other Sports Facility",
    "Tennis Court",
    "Bowling Green",
    "Golf Course",
}
RELATIONS = (
    "nhs_gp",
    "nhs_dentist",
    "osm_hospital",
    "osm_pharmacy",
    "osm_leisure",
    "os_green",
    "os_active_green",
    "osm_fast_food",
    "osm_gambling",
    "osm_pub_bar",
    "osm_tobacco",
    "osm_food_retail",
)
OSM_RELATION_CATEGORIES = {
    "osm_hospital": {"hospital"},
    "osm_pharmacy": {"pharmacy"},
    "osm_leisure": {"sport_fitness"},
    "osm_fast_food": {"fast_food"},
    "osm_gambling": {"gambling"},
    "osm_pub_bar": {"pub_bar"},
    "osm_tobacco": {"tobacco"},
    "osm_food_retail": {"supermarket", "convenience"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a routed facility-to-LSOA multi-relation graph"
    )
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--nhs-gp",
        type=Path,
        default=Path("data/raw/nhs_ods_2026_08_26/gp_practices.csv"),
    )
    parser.add_argument(
        "--nhs-dentist",
        type=Path,
        default=Path("data/raw/nhs_ods_2026_08_26/dental_practices.csv"),
    )
    parser.add_argument(
        "--codepoint-dir",
        type=Path,
        default=Path("data/raw/code_point_open_2026_08/extracted/Data/CSV"),
    )
    parser.add_argument(
        "--greenspace",
        type=Path,
        default=Path(
            "data/raw/os_open_greenspace_2026_04/extracted/Data/opgrsp_gb.gpkg"
        ),
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--walk-limit-minutes", type=float, default=40.0)
    parser.add_argument("--maximum-snap-metres", type=float, default=500.0)
    return parser


def queen_edges(frame: gpd.GeoDataFrame) -> np.ndarray:
    source: list[int] = []
    destination: list[int] = []
    index = frame.sindex
    for node, geometry in enumerate(frame.geometry):
        for neighbor in index.query(geometry, predicate="touches"):
            neighbor = int(neighbor)
            if neighbor != node:
                source.append(neighbor)
                destination.append(node)
    return np.asarray([source, destination], dtype=np.int64)


def load_tabular_case(case_dir: Path) -> tuple[gpd.GeoDataFrame, list[str]]:
    sources = [
        ("official_context.parquet", "official_context_summary.json"),
        ("osm_context.parquet", "osm_context_summary.json"),
        ("os_green_context.parquet", "os_green_context_summary.json"),
        ("sentinel_ndvi_context.parquet", "sentinel_ndvi_context_summary.json"),
    ]
    frame = gpd.read_parquet(case_dir / sources[0][0]).to_crs(27700)
    feature_names: list[str] = []
    for position, (parquet_name, summary_name) in enumerate(sources):
        if position:
            extra = gpd.read_parquet(case_dir / parquet_name).drop(columns="geometry")
            frame = frame.merge(extra, on="lsoa21cd", validate="one_to_one")
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        feature_names.extend(summary["features"])
    return frame, feature_names


def load_nhs_facilities(
    relation: str,
    path: Path,
    postcode_coordinates: pd.DataFrame,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    located = load_active_nhs_coordinates(path, postcode_coordinates)
    min_x, min_y, max_x, max_y = bounds
    located = located[
        located["easting"].between(min_x - 5000.0, max_x + 5000.0)
        & located["northing"].between(min_y - 5000.0, max_y + 5000.0)
    ].copy()
    return gpd.GeoDataFrame(
        {
            "facility_id": relation + ":" + located["postcode"].astype(str),
            "relation": relation,
            "mass_raw": 1.0,
        },
        geometry=gpd.points_from_xy(located["easting"], located["northing"]),
        crs=27700,
    )


def load_facilities(args: argparse.Namespace, lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    case_dir = args.case_root / args.study
    bounds = tuple(float(value) for value in lsoas.total_bounds)
    organisation_paths = {"nhs_gp": args.nhs_gp, "nhs_dentist": args.nhs_dentist}
    needed_postcodes: set[str] = set()
    for path in organisation_paths.values():
        frame = pd.read_csv(path, header=None, dtype=str)
        frame = frame[frame.iloc[:, 12].eq("ACTIVE")].copy()
        postcodes = normalise_postcode(frame.iloc[:, 9])
        needed_postcodes.update(postcodes[postcodes.ne("")])
    postcode_coordinates = load_codepoint_coordinates(args.codepoint_dir, needed_postcodes)
    parts = [
        load_nhs_facilities(relation, path, postcode_coordinates, bounds)
        for relation, path in organisation_paths.items()
    ]

    osm = gpd.read_parquet(case_dir / "osm_facilities.parquet").to_crs(27700)
    for relation, categories in OSM_RELATION_CATEGORIES.items():
        selected = osm[osm["category"].isin(categories)].copy()
        selected["facility_id"] = relation + ":" + selected["facility_id"].astype(str)
        selected["relation"] = relation
        selected["mass_raw"] = 1.0
        parts.append(selected[["facility_id", "relation", "mass_raw", "geometry"]])

    min_x, min_y, max_x, max_y = bounds
    bbox = (min_x - 5000.0, min_y - 5000.0, max_x + 5000.0, max_y + 5000.0)
    green = gpd.read_file(args.greenspace, layer="greenspace_site", bbox=bbox).to_crs(27700)
    green["mass_raw"] = green.geometry.area.clip(lower=1.0)
    for relation, selected in (
        ("os_green", green),
        ("os_active_green", green[green["function"].isin(ACTIVE_GREEN_FUNCTIONS)]),
    ):
        selected = selected.copy()
        selected["facility_id"] = [f"{relation}:{index}" for index in selected.index]
        selected["relation"] = relation
        selected["geometry"] = selected.geometry.representative_point()
        parts.append(selected[["facility_id", "relation", "mass_raw", "geometry"]])

    facilities = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=27700)
    facilities = facilities[facilities["relation"].isin(RELATIONS)].copy()
    facilities = facilities.drop_duplicates(["facility_id"]).reset_index(drop=True)
    relation_median = facilities.groupby("relation")["mass_raw"].transform("median")
    facilities["mass"] = np.clip(
        facilities["mass_raw"].to_numpy(dtype=float)
        / np.maximum(relation_median.to_numpy(dtype=float), 1.0),
        0.1,
        10.0,
    )
    return facilities


def graph_features(
    lsoas: gpd.GeoDataFrame,
    relations: list[str],
    edge_destination: np.ndarray,
    edge_type: np.ndarray,
    edge_minutes: np.ndarray,
    edge_source_mass: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    scales = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    n = len(lsoas)
    for relation_index, relation in enumerate(relations):
        mask = edge_type == relation_index
        destination = edge_destination[mask]
        minutes = edge_minutes[mask]
        mass = edge_source_mass[mask]
        count = np.bincount(destination, minlength=n).astype(float)
        minimum = np.full(n, 40.0, dtype=float)
        np.minimum.at(minimum, destination, minutes)
        time_sum = np.bincount(destination, weights=minutes, minlength=n)
        mean_time = time_sum / np.maximum(count, 1.0)
        mass_sum = np.bincount(destination, weights=mass, minlength=n)
        blocks.extend(
            [
                np.log1p(count)[:, None],
                (minimum / 40.0)[:, None],
                (mean_time / 40.0)[:, None],
                np.log1p(mass_sum)[:, None],
            ]
        )
        names.extend(
            [
                f"facility_{relation}_log_count_40min",
                f"facility_{relation}_nearest_time_40scaled",
                f"facility_{relation}_mean_time_40scaled",
                f"facility_{relation}_log_mass_40min",
            ]
        )
        for scale in scales:
            weight = np.power(mass, 0.15) * np.exp(-np.power(minutes / scale, 1.5))
            potential = np.bincount(destination, weights=weight, minlength=n)
            blocks.append(np.log1p(potential)[:, None])
            names.append(f"facility_{relation}_gravity_s{int(scale)}")
    return np.column_stack(blocks).astype(np.float32), names


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 1 or args.walk_limit_minutes <= 20:
        raise ValueError("top-k must be positive and walk limit must exceed 20 minutes")
    case_dir = args.case_root / args.study
    lsoas, feature_names = load_tabular_case(case_dir)
    facilities = load_facilities(args, lsoas)

    network = np.load(case_dir / "walk_network.npz")
    road_xy = np.column_stack((network["road_x"], network["road_y"])).astype(float)
    edge_u = network["edge_u"].astype(np.int64)
    edge_v = network["edge_v"].astype(np.int64)
    length = network["edge_length_m"].astype(float)
    rows = np.concatenate((edge_u, edge_v))
    columns = np.concatenate((edge_v, edge_u))
    values = np.concatenate((length, length))
    road_graph = csr_matrix(
        (values, (rows, columns)), shape=(len(road_xy), len(road_xy))
    )
    component_count, component_label = connected_components(road_graph, directed=False)
    road_tree = cKDTree(road_xy)

    lsoa_points = lsoas.geometry.representative_point()
    lsoa_xy = np.column_stack((lsoa_points.x, lsoa_points.y))
    lsoa_snap, lsoa_road = road_tree.query(lsoa_xy, k=1)
    facility_xy = np.column_stack((facilities.geometry.x, facilities.geometry.y))
    facility_snap, facility_road = road_tree.query(facility_xy, k=1)
    facilities["snap_m"] = facility_snap
    facilities["road_index"] = facility_road
    facilities = facilities[facilities["snap_m"] <= args.maximum_snap_metres].copy()

    n_lsoa = len(lsoas)
    facilities = facilities.reset_index(drop=True)
    facilities["node_index"] = n_lsoa + np.arange(len(facilities), dtype=np.int64)
    edge_source: list[np.ndarray] = []
    edge_destination: list[np.ndarray] = []
    edge_minutes: list[np.ndarray] = []
    edge_type: list[np.ndarray] = []
    relation_names: list[str] = []
    audit: dict[str, dict[str, object]] = {}
    walk_limit_m = args.walk_limit_minutes * WALKING_METRES_PER_MINUTE

    for relation in RELATIONS:
        print(f"routing {args.study}/{relation}", flush=True)
        selected = facilities[facilities["relation"].eq(relation)].copy()
        if selected.empty:
            raise ValueError(f"no snapped facilities for required relation {relation}")
        distance, ids = k_nearest_network_sources(
            road_graph,
            source_nodes=selected["road_index"].to_numpy(dtype=np.int64),
            source_ids=np.arange(len(selected), dtype=np.int64),
            source_initial_cost=selected["snap_m"].to_numpy(dtype=float),
            k=args.top_k,
            limit=walk_limit_m,
        )
        distance = distance[lsoa_road] + lsoa_snap[:, None]
        ids = ids[lsoa_road]
        valid = (ids >= 0) & np.isfinite(distance) & (distance <= walk_limit_m)
        destination, rank = np.nonzero(valid)
        local_id = ids[destination, rank]
        minutes = distance[destination, rank] / WALKING_METRES_PER_MINUTE
        relation_index = len(relation_names)
        relation_names.append(relation)
        edge_source.append(
            selected.iloc[local_id]["node_index"].to_numpy(dtype=np.int64)
        )
        edge_destination.append(destination.astype(np.int64))
        edge_minutes.append(minutes.astype(np.float32))
        edge_type.append(np.full(len(destination), relation_index, dtype=np.int64))
        audit[relation] = {
            "facilities": int(len(selected)),
            "edges": int(len(destination)),
            "isolated_lsoas": int(n_lsoa - len(np.unique(destination))),
            "minutes_percentiles": {
                name: float(value)
                for name, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.percentile(minutes, (5, 25, 50, 75, 95)),
                    strict=True,
                )
            },
            "structural_edge_counts": {
                "le_15": int(np.sum(minutes <= 15.0)),
                "15_to_20": int(np.sum((minutes > 15.0) & (minutes <= 20.0))),
                "gt_20": int(np.sum(minutes > 20.0)),
            },
        }
        print(
            f"completed {args.study}/{relation}: {len(selected)} facilities, "
            f"{len(destination)} edges",
            flush=True,
        )

    facility_edge_source = np.concatenate(edge_source)
    facility_edge_destination = np.concatenate(edge_destination)
    facility_edge_minutes = np.concatenate(edge_minutes)
    facility_edge_type = np.concatenate(edge_type)
    source_mass = facilities.set_index("node_index").loc[facility_edge_source, "mass"].to_numpy(
        dtype=float
    )
    gravity_x, gravity_names = graph_features(
        lsoas,
        relation_names,
        facility_edge_destination,
        facility_edge_type,
        facility_edge_minutes,
        source_mass,
    )

    queen = queen_edges(lsoas)
    queen_distance = np.linalg.norm(lsoa_xy[queen[0]] - lsoa_xy[queen[1]], axis=1)
    spatial_relation = len(relation_names)
    relation_names.append("lsoa_queen")
    all_source = np.concatenate((facility_edge_source, queen[0]))
    all_destination = np.concatenate((facility_edge_destination, queen[1]))
    all_minutes = np.concatenate(
        (facility_edge_minutes, queen_distance / WALKING_METRES_PER_MINUTE)
    ).astype(np.float32)
    all_type = np.concatenate(
        (
            facility_edge_type,
            np.full(queen.shape[1], spatial_relation, dtype=np.int64),
        )
    )

    total_nodes = n_lsoa + len(facilities)
    x = np.zeros((total_nodes, len(feature_names)), dtype=np.float32)
    x[:n_lsoa] = lsoas[feature_names].to_numpy(dtype=np.float32)
    node_mass = np.ones(total_nodes, dtype=np.float32)
    node_mass[n_lsoa:] = facilities["mass"].to_numpy(dtype=np.float32)
    y = np.full(total_nodes, np.nan, dtype=np.float32)
    y[:n_lsoa] = lsoas["ahah"].to_numpy(dtype=np.float32)
    building_mask = np.zeros(total_nodes, dtype=bool)
    building_mask[:n_lsoa] = True
    group = np.full(total_nodes, -1, dtype=np.int64)
    group[:n_lsoa] = np.arange(n_lsoa, dtype=np.int64)

    graph_path = case_dir / "facility_lsoa_graph.npz"
    np.savez_compressed(
        graph_path,
        x=x,
        node_mass=node_mass,
        edge_index=np.asarray([all_source, all_destination], dtype=np.int64),
        edge_impedance=all_minutes,
        edge_type=all_type,
        y=y,
        building_mask=building_mask,
        group=group,
        feature_names=np.asarray(feature_names),
        relation_names=np.asarray(relation_names),
        lsoa_codes=lsoas["lsoa21cd"].to_numpy(dtype=str),
    )
    validated = load_graph_npz(graph_path)

    gravity_values = pd.DataFrame(
        gravity_x, columns=gravity_names, index=lsoas.index
    )
    gravity_frame = gpd.GeoDataFrame(
        pd.concat((lsoas[["lsoa21cd", "geometry"]], gravity_values), axis=1),
        geometry="geometry",
        crs=lsoas.crs,
    )
    gravity_path = case_dir / "facility_gravity_features.parquet"
    gravity_frame.to_parquet(gravity_path, index=False)
    summary = {
        "study": args.study,
        "graph": str(graph_path.resolve()),
        "gravity_features": str(gravity_path.resolve()),
        "lsoas": n_lsoa,
        "facility_nodes": int(len(facilities)),
        "nodes": validated.num_nodes,
        "facility_edges": int(len(facility_edge_source)),
        "queen_edges": int(queen.shape[1]),
        "relations": relation_names,
        "top_k": args.top_k,
        "walk_limit_minutes": args.walk_limit_minutes,
        "walking_metres_per_minute": WALKING_METRES_PER_MINUTE,
        "impedance": "local OSM walk-network shortest-path minutes including endpoint snap distances",
        "road_nodes": int(len(road_xy)),
        "road_components": int(component_count),
        "largest_road_component_nodes": int(np.bincount(component_label).max()),
        "lsoa_snap_m_percentiles": {
            name: float(value)
            for name, value in zip(
                ("p50", "p90", "p95", "p99", "max"),
                np.percentile(lsoa_snap, (50, 90, 95, 99, 100)),
                strict=True,
            )
        },
        "feature_count": len(feature_names),
        "gravity_feature_count": len(gravity_names),
        "relation_audit": audit,
        "sha256": {
            "graph": sha256(graph_path),
            "gravity_features": sha256(gravity_path),
            "walk_network": sha256(case_dir / "walk_network.npz"),
            "osm_facilities": sha256(case_dir / "osm_facilities.parquet"),
            "nhs_gp": sha256(args.nhs_gp),
            "nhs_dentist": sha256(args.nhs_dentist),
            "greenspace": sha256(args.greenspace),
        },
    }
    (case_dir / "facility_lsoa_graph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
