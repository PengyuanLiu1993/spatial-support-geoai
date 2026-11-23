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
    area_weighted_grid_values,
    load_active_nhs_coordinates,
    load_codepoint_coordinates,
    load_pcm_grid,
    normalise_postcode,
    radius_counts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build target-free official LSOA context for the two case studies"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--defra-dir", type=Path, default=Path("data/raw/defra_pcm_2024"))
    parser.add_argument(
        "--nhs-gp", type=Path, default=Path("data/raw/nhs_ods_2026_08_26/gp_practices.csv")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    organisation_paths = {"nhs_gp": args.nhs_gp, "nhs_dentist": args.nhs_dentist}
    needed_postcodes: set[str] = set()
    for path in organisation_paths.values():
        frame = pd.read_csv(path, header=None, dtype=str)
        frame = frame[frame.iloc[:, 12].eq("ACTIVE")].copy()
        frame["postcode"] = normalise_postcode(frame.iloc[:, 9])
        frame = frame[frame["postcode"].ne("")].drop_duplicates("postcode")
        needed_postcodes.update(frame["postcode"])
    postcode_coordinates = load_codepoint_coordinates(args.codepoint_dir, needed_postcodes)
    nhs_coordinates = {
        name: load_active_nhs_coordinates(path, postcode_coordinates)
        for name, path in organisation_paths.items()
    }

    summaries = []
    for study in args.study:
        case_dir = args.case_root / study
        lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
        points = lsoas.geometry.representative_point()
        point_xy = np.column_stack((points.x, points.y))
        context = lsoas.copy()
        context["log_lsoa_area_km2"] = np.log1p(context.geometry.area / 1_000_000.0)
        context["lsoa_compactness"] = np.clip(
            4.0 * np.pi * context.geometry.area / np.maximum(context.geometry.length**2, 1.0),
            0.0,
            1.0,
        )
        feature_names = ["log_lsoa_area_km2", "lsoa_compactness"]

        pollutant_files = {
            "defra_no2_2024": args.defra_dir / "no2_2024.csv",
            "defra_pm10_2024": args.defra_dir / "pm10_2024.csv",
            "defra_so2_2024": args.defra_dir / "so2_2024.csv",
        }
        for feature_name, path in pollutant_files.items():
            grid = load_pcm_grid(path, tuple(lsoas.total_bounds))
            value_column = str(grid.columns[-2])
            values = area_weighted_grid_values(lsoas, grid, value_column, "lsoa21cd")
            context[feature_name] = context["lsoa21cd"].map(values)
            feature_names.append(feature_name)

        min_x, min_y, max_x, max_y = lsoas.total_bounds
        for source_name, located in nhs_coordinates.items():
            nearby = located[
                located["easting"].between(min_x - 10_000.0, max_x + 10_000.0)
                & located["northing"].between(min_y - 10_000.0, max_y + 10_000.0)
            ]
            source_xy = nearby[["easting", "northing"]].to_numpy(dtype=float)
            if not len(source_xy):
                raise ValueError(f"no geocoded {source_name} sites near {study}")
            nearest, _ = cKDTree(source_xy).query(point_xy, k=1)
            counts = radius_counts(point_xy, source_xy, (1000.0, 3000.0, 5000.0))
            names = [
                f"log_{source_name}_nearest_m",
                f"log_{source_name}_count_1000m",
                f"log_{source_name}_count_3000m",
                f"log_{source_name}_count_5000m",
            ]
            context[names[0]] = np.log1p(nearest)
            context[names[1:]] = np.log1p(counts)
            feature_names.extend(names)

        if not np.isfinite(context[feature_names].to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite official context values for {study}")
        output = case_dir / "official_context.parquet"
        context.to_parquet(output, index=False)
        summary = {
            "study": study,
            "output": str(output.resolve()),
            "lsoas": int(len(context)),
            "features": feature_names,
            "nhs_sites_with_coordinates": {
                name: int(len(frame)) for name, frame in nhs_coordinates.items()
            },
            "sha256": {
                "output": sha256(output),
                "defra_no2_2024": sha256(pollutant_files["defra_no2_2024"]),
                "defra_pm10_2024": sha256(pollutant_files["defra_pm10_2024"]),
                "defra_so2_2024": sha256(pollutant_files["defra_so2_2024"]),
                "nhs_gp": sha256(args.nhs_gp),
                "nhs_dentist": sha256(args.nhs_dentist),
            },
        }
        (case_dir / "official_context_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
