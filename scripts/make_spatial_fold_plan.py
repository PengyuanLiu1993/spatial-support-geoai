from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np

# Avoid joblib's unreliable physical-core probe on non-English Windows hosts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic contiguous spatial clusters for cross-validation"
    )
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--lsoas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.folds < 3:
        raise ValueError("at least three folds are required")

    buildings = gpd.read_parquet(args.buildings)
    lsoas = gpd.read_parquet(args.lsoas)
    if lsoas.crs is None:
        raise ValueError("LSOA geometry must have a coordinate reference system")
    if lsoas.crs.to_epsg() != 27700:
        lsoas = lsoas.to_crs(27700)

    lsoa_group = buildings[["lsoa21cd", "spatial_block"]].drop_duplicates()
    if lsoa_group["lsoa21cd"].duplicated().any():
        raise ValueError("an LSOA appears in more than one spatial block")
    points = lsoas[["lsoa21cd", "geometry"]].copy()
    points["point"] = points.geometry.representative_point()
    points["x"] = points["point"].x
    points["y"] = points["point"].y
    points = points.merge(lsoa_group, on="lsoa21cd", validate="one_to_one")

    group_frame = (
        points.groupby("spatial_block", as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"), lsoas=("lsoa21cd", "size"))
        .sort_values("spatial_block")
        .reset_index(drop=True)
    )
    group_ids = group_frame["spatial_block"].to_numpy(dtype=np.int64)
    expected = np.arange(group_ids.max() + 1)
    if not np.array_equal(group_ids, expected):
        raise ValueError("spatial block ids must be contiguous and start at zero")
    if len(group_frame) < args.folds:
        raise ValueError("the requested number of folds exceeds the number of groups")

    coordinates = group_frame[["x", "y"]].to_numpy()
    model = KMeans(n_clusters=args.folds, random_state=args.seed, n_init=50)
    raw_labels = model.fit_predict(
        coordinates, sample_weight=group_frame["lsoas"].to_numpy(dtype=float)
    )

    # Stable, human-readable fold ids ordered clockwise around the study centre.
    centre = model.cluster_centers_.mean(axis=0)
    angles = np.arctan2(
        model.cluster_centers_[:, 1] - centre[1],
        model.cluster_centers_[:, 0] - centre[0],
    )
    ordered_raw = np.argsort(angles)
    raw_to_fold = {int(raw): fold for fold, raw in enumerate(ordered_raw)}
    folds = np.array([raw_to_fold[int(raw)] for raw in raw_labels], dtype=np.int64)
    group_to_fold = np.full(group_ids.max() + 1, -1, dtype=np.int64)
    group_to_fold[group_ids] = folds

    group_frame["fold"] = folds
    lsoa_fold = points.merge(
        group_frame[["spatial_block", "fold"]], on="spatial_block", validate="many_to_one"
    )
    building_counts = buildings.groupby("spatial_block").size().rename("buildings")
    fold_stats = (
        group_frame.merge(building_counts, on="spatial_block", validate="one_to_one")
        .groupby("fold", as_index=False)
        .agg(
            spatial_blocks=("spatial_block", "size"),
            lsoas=("lsoas", "sum"),
            buildings=("buildings", "sum"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
        )
        .sort_values("fold")
    )
    if len(fold_stats) != args.folds or (fold_stats["lsoas"] == 0).any():
        raise RuntimeError("fold construction produced an empty fold")

    fold_runs = [
        {"test_fold": fold, "validation_fold": (fold + 1) % args.folds}
        for fold in range(args.folds)
    ]
    payload = {
        "group_to_fold": group_to_fold.tolist(),
        "fold_runs": fold_runs,
        "fold_stats": fold_stats.to_dict(orient="records"),
        "lsoa_to_fold": {
            str(code): int(fold)
            for code, fold in lsoa_fold[["lsoa21cd", "fold"]].itertuples(index=False)
        },
        "metadata": {
            "method": "weighted_kmeans_on_1km_spatial_block_centroids",
            "folds": args.folds,
            "seed": args.seed,
            "crs": "EPSG:27700",
            "validation_rule": "next_clockwise_fold",
            "fold_id_order": "centroid_angle_ascending",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2))
    print(fold_stats.to_string(index=False))


if __name__ == "__main__":
    main()
