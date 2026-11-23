from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from gravity_gnn.semi_synthetic import (
    GRAVITY_COLUMNS,
    TRUE_SCALES_MINUTES,
    TRUE_WEIGHTS,
    raw_gravity_signal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible semi-synthetic gravity targets on real AHAH geographies"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=0.20,
        help="noise standard deviation as a fraction of the final 10-point signal SD",
    )
    parser.add_argument("--output-name", default="semi_synthetic_gravity.npz")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.noise_fraction < 0:
        raise ValueError("noise fraction must be non-negative")

    frames = []
    for study in args.study:
        case_dir = args.case_root / study
        lsoas = gpd.read_parquet(case_dir / "lsoas.parquet")
        folds = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
        missing = sorted(set(GRAVITY_COLUMNS) - set(lsoas.columns))
        if missing:
            raise ValueError(f"{study} is missing gravity columns: {missing}")
        points = lsoas.geometry.representative_point()
        frame = {
            "study": study,
            "case_dir": case_dir,
            "lsoas": lsoas,
            "codes": np.asarray(lsoas["lsoa21cd"].astype(str), dtype=str),
            "travel": lsoas[list(GRAVITY_COLUMNS)].to_numpy(dtype=float),
            "coordinates": np.column_stack((points.x, points.y)),
            "fold": lsoas["lsoa21cd"].map(folds["lsoa_to_fold"]).to_numpy(dtype=int),
        }
        frame["raw_signal"] = raw_gravity_signal(frame["travel"], frame["coordinates"])
        frames.append(frame)

    combined_signal = np.concatenate([frame["raw_signal"] for frame in frames])
    signal_mean = float(combined_signal.mean())
    signal_std = float(combined_signal.std(ddof=0))
    if signal_std <= 0:
        raise ValueError("semi-synthetic signal has zero variance")

    summaries = []
    for index, frame in enumerate(frames):
        clean_target = 50.0 + 10.0 * (frame["raw_signal"] - signal_mean) / signal_std
        rng = np.random.default_rng(args.seed + 1009 * index)
        target = clean_target + rng.normal(
            0.0, args.noise_fraction * 10.0, size=len(clean_target)
        )
        output = frame["case_dir"] / args.output_name
        np.savez_compressed(
            output,
            lsoa_code=frame["codes"],
            travel_minutes=frame["travel"].astype(np.float32),
            coordinates=frame["coordinates"].astype(np.float32),
            fold=frame["fold"].astype(np.int64),
            y_clean=clean_target.astype(np.float32),
            y=target.astype(np.float32),
            feature_names=np.asarray(GRAVITY_COLUMNS),
        )
        summary = {
            "study": frame["study"],
            "output": str(output.resolve()),
            "lsoas": int(len(target)),
            "seed": args.seed + 1009 * index,
            "noise_fraction": args.noise_fraction,
            "global_signal_mean": signal_mean,
            "global_signal_std": signal_std,
            "target": {
                "mean": float(target.mean()),
                "std": float(target.std(ddof=1)),
                "min": float(target.min()),
                "max": float(target.max()),
            },
            "true_scales_minutes": TRUE_SCALES_MINUTES,
            "true_weights": TRUE_WEIGHTS,
        }
        output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
