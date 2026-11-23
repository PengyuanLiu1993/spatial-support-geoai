from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from gravity_gnn.case_studies import (
    boundary_area_km2,
    load_administrative_boundary,
    make_spatial_fold_plan,
    select_lsoas,
    sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fixed Greater London and Greater Manchester AHAH case studies"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/case_studies.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", help="study id; repeat to select more")
    return parser


def target_statistics(lsoas: gpd.GeoDataFrame, columns: list[str]) -> dict:
    result = {}
    for column in columns:
        values = lsoas[column].to_numpy(dtype=float)
        result[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    project_root = args.config.resolve().parent.parent
    ahah_path = project_root / config["ahah"]
    ahah = gpd.read_parquet(ahah_path)
    target_crs = config.get("crs", "EPSG:27700")
    if ahah.crs is None:
        raise ValueError("AHAH GeoParquet has no CRS")
    if str(ahah.crs) != target_crs:
        ahah = ahah.to_crs(target_crs)

    selected_ids = set(args.study or [])
    studies = [
        study for study in config["studies"] if not selected_ids or study["id"] in selected_ids
    ]
    missing_ids = selected_ids - {study["id"] for study in studies}
    if missing_ids:
        raise ValueError(f"unknown study ids: {sorted(missing_ids)}")

    summaries = []
    for study in studies:
        pbf_path = project_root / study["source_pbf"]
        output_dir = args.output_root / study["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        boundary = load_administrative_boundary(
            pbf_path,
            study["area_name"],
            study["admin_level"],
            target_crs,
        )
        lsoas = select_lsoas(ahah, boundary)
        fold_plan = make_spatial_fold_plan(
            lsoas,
            folds=int(config.get("folds", 5)),
            seed=int(config.get("seed", 20260826)),
        )
        boundary_path = output_dir / "boundary.parquet"
        lsoa_path = output_dir / "lsoas.parquet"
        folds_path = output_dir / "spatial_folds_5.json"
        boundary.to_parquet(boundary_path, index=False)
        lsoas.to_parquet(lsoa_path, index=False)
        folds_path.write_text(
            json.dumps(fold_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "id": study["id"],
            "label": study["label"],
            "role": study["role"],
            "area_name": study["area_name"],
            "admin_level": study["admin_level"],
            "snapshot": study["snapshot"],
            "source_url": study["source_url"],
            "source_pbf": str(pbf_path.resolve()),
            "boundary_area_km2": boundary_area_km2(boundary),
            "lsoas": int(len(lsoas)),
            "ahah_statistics": target_statistics(
                lsoas,
                ["ahah", "domain_h", "domain_g", "domain_e", "domain_r"],
            ),
            "files": {
                "boundary": str(boundary_path.resolve()),
                "lsoas": str(lsoa_path.resolve()),
                "fold_plan": str(folds_path.resolve()),
            },
            "sha256": {
                "ahah": sha256(ahah_path),
                "source_pbf": sha256(pbf_path),
                "boundary": sha256(boundary_path),
                "lsoas": sha256(lsoa_path),
            },
        }
        (output_dir / "case_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
