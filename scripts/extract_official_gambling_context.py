from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gravity_gnn.case_studies import sha256
from gravity_gnn.context import (
    load_codepoint_coordinates,
    normalise_postcode,
    radius_counts,
    radius_weighted_sums,
)


ACTIVITY_GROUPS = {
    "betting": {"BETTING SHOP", "POOL BETTING"},
    "adult_gaming": {"ADULT GAMING CENTRE", "AGC"},
    "bingo": {"BINGO"},
    "casino": {"CASINO", "CASINO 2005"},
    "family_entertainment": {"FAMILY ENTERTAINMENT CENTRE"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Geocode official gambling premises and build LSOA gravity context"
    )
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--register",
        type=Path,
        default=Path(
            "data/raw/official_retail_2026_08_26/premises-licence-register.csv"
        ),
    )
    parser.add_argument(
        "--codepoint-dir",
        type=Path,
        default=Path("data/raw/code_point_open_2026_08/extracted/Data/CSV"),
    )
    parser.add_argument("--radii", default="1000,3000,5000")
    return parser


def add_group_features(
    output: gpd.GeoDataFrame,
    query_xy: np.ndarray,
    sites: pd.DataFrame,
    prefix: str,
    radii: tuple[float, ...],
) -> list[str]:
    source_xy = sites[["easting", "northing"]].to_numpy(dtype=float)
    if len(source_xy):
        nearest, _ = cKDTree(source_xy).query(query_xy, k=1)
    else:
        nearest = np.full(len(query_xy), 100_000.0)
    site_counts = radius_counts(query_xy, source_xy, radii)
    licence_counts = radius_weighted_sums(
        query_xy,
        source_xy,
        sites["licences"].to_numpy(dtype=float),
        radii,
    )
    names = [f"log_official_gambling_{prefix}_nearest_m"]
    output[names[0]] = np.log1p(nearest)
    for column, radius in enumerate(radii):
        site_name = f"log_official_gambling_{prefix}_sites_{int(radius)}m"
        licence_name = f"log_official_gambling_{prefix}_licences_{int(radius)}m"
        output[site_name] = np.log1p(site_counts[:, column])
        output[licence_name] = np.log1p(licence_counts[:, column])
        names.extend((site_name, licence_name))
    return names


def main() -> None:
    args = build_parser().parse_args()
    radii = tuple(float(value) for value in args.radii.split(",") if value.strip())
    register = pd.read_csv(args.register, dtype=str)
    register["postcode"] = normalise_postcode(register["Postcode"])
    register["activity"] = (
        register["Premises Activity"].fillna("UNKNOWN").str.strip().str.upper()
    )
    register = register[register["postcode"].ne("")].copy()
    coordinates = load_codepoint_coordinates(
        args.codepoint_dir, set(register["postcode"])
    )
    located = register.merge(
        coordinates, on="postcode", how="inner", validate="many_to_one"
    )

    case_dir = args.case_root / args.study
    lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
    points = lsoas.geometry.representative_point()
    query_xy = np.column_stack((points.x, points.y))
    min_x, min_y, max_x, max_y = lsoas.total_bounds
    nearby = located[
        located["easting"].between(min_x - 10_000.0, max_x + 10_000.0)
        & located["northing"].between(min_y - 10_000.0, max_y + 10_000.0)
    ].copy()

    output = lsoas[["lsoa21cd", "geometry"]].copy()
    feature_names = []
    group_counts = {}
    selections = {"all": nearby}
    selections.update(
        {
            name: nearby[nearby["activity"].isin(activities)]
            for name, activities in ACTIVITY_GROUPS.items()
        }
    )
    for name, group in selections.items():
        sites = (
            group.groupby(["postcode", "easting", "northing"], as_index=False)
            .size()
            .rename(columns={"size": "licences"})
        )
        group_counts[name] = {
            "premises_records": int(len(group)),
            "geocoded_postcode_sites": int(len(sites)),
        }
        feature_names.extend(
            add_group_features(output, query_xy, sites, name, radii)
        )

    output_path = case_dir / "official_gambling_context.parquet"
    output.to_parquet(output_path, index=False)
    summary = {
        "study": args.study,
        "source": str(args.register.resolve()),
        "source_records": int(len(register)),
        "geocoded_records": int(len(located)),
        "geocoding_rate": float(len(located) / max(len(register), 1)),
        "nearby_groups": group_counts,
        "features": feature_names,
        "target_derived_features": False,
        "output": str(output_path.resolve()),
        "sha256": {
            "source": sha256(args.register),
            "output": sha256(output_path),
        },
    }
    (case_dir / "official_gambling_context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
