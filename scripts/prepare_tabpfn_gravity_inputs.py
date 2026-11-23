from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_distance_effect_experiment import load_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare dependency-light inputs for the TabPFN gravity experiment"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/tabpfn_gravity/input_bundle.npz")
    )
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    if len(args.study) != 2:
        raise ValueError("the pooled TabPFN experiment requires exactly two cities")

    bundles = {
        study: load_case(args.case_root, args.oof_dir, study) for study in args.study
    }
    first_names = bundles[args.study[0]]["base_metadata"]["feature_names"]
    for study in args.study[1:]:
        if bundles[study]["base_metadata"]["feature_names"] != first_names:
            raise ValueError("base feature schemas differ between cities")

    arrays: dict[str, np.ndarray] = {
        "studies": np.asarray(args.study, dtype=str),
        "base_feature_names": np.asarray(first_names, dtype=str),
    }
    summary = {
        "studies": args.study,
        "base_feature_count": len(first_names),
        "gravity_feature_count": 24,
        "gravity_variant": "gravity_scale_15",
        "gravity_scale_minutes": 15.0,
        "gravity_beta": 1.5,
        "validation": "five frozen contiguous spatial folds",
        "target_derived_features": False,
        "absolute_coordinate_features": False,
        "cities": {},
    }
    for index, study in enumerate(args.study):
        bundle = bundles[study]
        gravity = bundle["variants"]["gravity_scale_15"]
        prefix = f"city_{index}"
        arrays[f"{prefix}_base_x"] = bundle["base_x"].astype(np.float32)
        arrays[f"{prefix}_gravity_x"] = gravity.astype(np.float32)
        arrays[f"{prefix}_y"] = bundle["y"].astype(np.float32)
        arrays[f"{prefix}_fold"] = bundle["fold"].astype(np.int8)
        arrays[f"{prefix}_lsoa"] = np.asarray(
            bundle["frame"]["lsoa21cd"].astype(str).tolist(), dtype=str
        )
        arrays[f"{prefix}_hgb_gravity15_prediction"] = bundle["frame"][
            "prediction_facility_gravity_scale15"
        ].to_numpy(dtype=np.float32)
        summary["cities"][study] = {
            "rows": int(len(bundle["y"])),
            "fold_counts": {
                str(int(fold)): int(np.sum(bundle["fold"] == fold))
                for fold in np.unique(bundle["fold"])
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary["bundle"] = str(args.output)
    summary["bundle_sha256"] = file_sha256(args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
