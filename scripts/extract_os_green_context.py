from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from gravity_gnn.case_studies import sha256
from gravity_gnn.context import radius_counts, radius_weighted_sums


ACTIVE_FUNCTIONS = {
    "Play Space",
    "Playing Field",
    "Other Sports Facility",
    "Tennis Court",
    "Bowling Green",
    "Golf Course",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract OS Open Greenspace LSOA context")
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--greenspace",
        type=Path,
        default=Path("data/raw/os_open_greenspace_2026_04/extracted/Data/opgrsp_gb.gpkg"),
    )
    parser.add_argument("--radii", default="1000,3000,5000")
    return parser


def coverage_fraction(lsoas: gpd.GeoDataFrame, sites: gpd.GeoDataFrame) -> np.ndarray:
    output = np.zeros(len(lsoas), dtype=float)
    site_index = sites.sindex
    for index, geometry in enumerate(lsoas.geometry):
        candidates = sites.iloc[list(site_index.query(geometry, predicate="intersects"))]
        if len(candidates):
            covered = candidates.geometry.intersection(geometry).area.sum()
            output[index] = min(float(covered / max(geometry.area, 1.0)), 1.0)
    return output


def point_context(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    radii: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_xy):
        nearest, _ = cKDTree(source_xy).query(query_xy, k=1)
    else:
        nearest = np.full(len(query_xy), 100_000.0)
    return nearest, radius_counts(query_xy, source_xy, radii)


def main() -> None:
    args = build_parser().parse_args()
    radii = tuple(float(value) for value in args.radii.split(",") if value.strip())
    case_dir = args.case_root / args.study
    lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
    points = lsoas.geometry.representative_point()
    query_xy = np.column_stack((points.x, points.y))
    min_x, min_y, max_x, max_y = lsoas.total_bounds
    bbox = (min_x - 5000.0, min_y - 5000.0, max_x + 5000.0, max_y + 5000.0)
    access = gpd.read_file(args.greenspace, layer="access_point", bbox=bbox).to_crs(27700)
    sites = gpd.read_file(args.greenspace, layer="greenspace_site", bbox=bbox).to_crs(27700)
    active = sites[sites["function"].isin(ACTIVE_FUNCTIONS)].copy()

    output = lsoas[["lsoa21cd", "geometry"]].copy()
    output["os_green_coverage_fraction"] = coverage_fraction(lsoas, sites)
    output["os_active_green_coverage_fraction"] = coverage_fraction(lsoas, active)
    feature_names = ["os_green_coverage_fraction", "os_active_green_coverage_fraction"]

    groups = {
        "access": access.geometry,
        "site": sites.geometry.representative_point(),
        "active_site": active.geometry.representative_point(),
    }
    for group_name, geometry in groups.items():
        source_xy = np.column_stack((geometry.x, geometry.y))
        nearest, counts = point_context(query_xy, source_xy, radii)
        nearest_name = f"log_os_green_{group_name}_nearest_m"
        output[nearest_name] = np.log1p(nearest)
        feature_names.append(nearest_name)
        for column, radius in enumerate(radii):
            name = f"log_os_green_{group_name}_count_{int(radius)}m"
            output[name] = np.log1p(counts[:, column])
            feature_names.append(name)

    for group_name, frame in {"site": sites, "active_site": active}.items():
        geometry = frame.geometry.representative_point()
        source_xy = np.column_stack((geometry.x, geometry.y))
        areas = frame.geometry.area.to_numpy(dtype=float)
        area_sums = radius_weighted_sums(query_xy, source_xy, areas, radii)
        for column, radius in enumerate(radii):
            name = f"log_os_green_{group_name}_area_{int(radius)}m"
            output[name] = np.log1p(area_sums[:, column])
            feature_names.append(name)

    output_path = case_dir / "os_green_context.parquet"
    output.to_parquet(output_path, index=False)
    summary = {
        "study": args.study,
        "source": str(args.greenspace.resolve()),
        "lsoas": int(len(lsoas)),
        "access_points": int(len(access)),
        "sites": int(len(sites)),
        "active_sites": int(len(active)),
        "features": feature_names,
        "output": str(output_path.resolve()),
        "sha256": {"source": sha256(args.greenspace), "output": sha256(output_path)},
    }
    (case_dir / "os_green_context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
