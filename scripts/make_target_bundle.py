from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np


DEFAULT_TARGETS = [
    "ahah",
    "domain_h",
    "domain_g",
    "domain_e",
    "domain_r",
    "GP",
    "dentist",
    "hospital",
    "pharmacy",
    "greenspace_active",
    "bluespace",
    "fast_food",
    "gambling",
    "pub_bar",
    "tobacco",
    "leisure",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align LSOA targets to graph nodes without rebuilding graph edges"
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--lsoas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = np.load(args.graph, allow_pickle=False)
    node_count = int(graph["x"].shape[0])
    building_mask = graph["building_mask"].astype(bool)
    buildings = gpd.read_parquet(args.buildings)[["node_index", "lsoa21cd"]]
    lsoas = gpd.read_parquet(args.lsoas)
    targets = [name.strip() for name in args.targets.split(",") if name.strip()]
    missing = sorted(set(targets) - set(lsoas.columns))
    if missing:
        raise ValueError(f"LSOA table is missing targets: {missing}")
    if buildings["node_index"].duplicated().any():
        raise ValueError("building node indices must be unique")
    expected_nodes = np.flatnonzero(building_mask)
    actual_nodes = np.sort(buildings["node_index"].to_numpy(dtype=np.int64))
    if not np.array_equal(actual_nodes, expected_nodes):
        raise ValueError("building table does not align with graph building nodes")

    joined = buildings.merge(
        lsoas[["lsoa21cd", *targets]], on="lsoa21cd", how="left", validate="many_to_one"
    )
    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, float]] = {}
    for name in targets:
        values = joined[name].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"target {name} contains missing or non-finite values")
        node_values = np.full(node_count, np.nan, dtype=np.float32)
        node_values[joined["node_index"].to_numpy(dtype=np.int64)] = values
        arrays[name] = node_values
        unique_values = lsoas[name].to_numpy(dtype=float)
        stats[name] = {
            "mean": float(np.mean(unique_values)),
            "std": float(np.std(unique_values, ddof=1)),
            "min": float(np.min(unique_values)),
            "max": float(np.max(unique_values)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "graph": str(args.graph.resolve()),
        "buildings": str(args.buildings.resolve()),
        "lsoas": str(args.lsoas.resolve()),
        "nodes": node_count,
        "labelled_buildings": int(building_mask.sum()),
        "lsoas_count": int(lsoas["lsoa21cd"].nunique()),
        "targets": targets,
        "target_stats_at_lsoa": stats,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
