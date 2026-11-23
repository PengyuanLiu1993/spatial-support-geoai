from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from run_poi_distance_breakpoint import load_case, make_spatial_blocks
except ModuleNotFoundError:
    from scripts.run_poi_distance_breakpoint import load_case, make_spatial_blocks


STUDIES = ("greater_london", "greater_manchester")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare paired full and relation-ablated graph OOF predictions."
    )
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--ablated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, default=Path("data/processed/case_studies"))
    parser.add_argument("--oof-dir", type=Path, default=Path("results/oof_predictions"))
    parser.add_argument("--spatial-blocks", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - observed
    total = np.sum((observed - observed.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(error**2) / max(float(total), 1e-12)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def macro_metrics(
    observed: np.ndarray, prediction: np.ndarray, folds: np.ndarray
) -> dict[str, float]:
    rows = [
        metrics(observed[folds == fold], prediction[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("r2", "rmse", "mae")
    }


def block_bootstrap_difference(
    observed: np.ndarray,
    full_prediction: np.ndarray,
    ablated_prediction: np.ndarray,
    blocks: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    unique_blocks = np.unique(blocks)
    n_by_block = np.asarray([np.sum(blocks == block) for block in unique_blocks], dtype=float)
    full_sse = np.asarray(
        [
            np.sum((observed[blocks == block] - full_prediction[blocks == block]) ** 2)
            for block in unique_blocks
        ]
    )
    ablated_sse = np.asarray(
        [
            np.sum((observed[blocks == block] - ablated_prediction[blocks == block]) ** 2)
            for block in unique_blocks
        ]
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique_blocks),
        np.full(len(unique_blocks), 1.0 / len(unique_blocks)),
        size=replicates,
    )
    sample_n = counts @ n_by_block
    difference = np.sqrt((counts @ ablated_sse) / sample_n) - np.sqrt(
        (counts @ full_sse) / sample_n
    )
    return {
        "pooled_ablated_minus_full_rmse": float(
            metrics(observed, ablated_prediction)["rmse"]
            - metrics(observed, full_prediction)["rmse"]
        ),
        "bootstrap_lower": float(np.quantile(difference, 0.025)),
        "bootstrap_upper": float(np.quantile(difference, 0.975)),
        "bootstrap_share_above_zero": float(np.mean(difference > 0.0)),
    }


def load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []

    for study_index, study in enumerate(STUDIES):
        full = load_payload(args.full_dir / f"{study}.npz")
        ablated = load_payload(args.ablated_dir / f"{study}.npz")
        for key in ("lsoa_codes", "observed", "fold", "seed"):
            if not np.array_equal(full[key], ablated[key]):
                raise ValueError(f"full and ablated outputs differ for {study} {key}")

        observed = full["observed"].astype(float)
        folds = full["fold"].astype(int)
        for index, seed_value in enumerate(full["seed"].astype(int)):
            full_metrics = macro_metrics(observed, full["seed_predictions"][index], folds)
            ablated_metrics = macro_metrics(observed, ablated["seed_predictions"][index], folds)
            seed_rows.append(
                {
                    "study": study,
                    "seed": int(seed_value),
                    **{f"full_{key}": value for key, value in full_metrics.items()},
                    **{f"ablated_{key}": value for key, value in ablated_metrics.items()},
                    "ablated_minus_full_rmse": ablated_metrics["rmse"] - full_metrics["rmse"],
                }
            )

        full_prediction = full["ensemble_prediction"].astype(float)
        ablated_prediction = ablated["ensemble_prediction"].astype(float)
        full_metrics = macro_metrics(observed, full_prediction, folds)
        ablated_metrics = macro_metrics(observed, ablated_prediction, folds)
        case = load_case(args.case_root, args.oof_dir, study)
        blocks = make_spatial_blocks(
            np.asarray(case["coordinates"]), args.spatial_blocks, args.seed + study_index
        )
        ensemble_rows.append(
            {
                "study": study,
                **{f"full_{key}": value for key, value in full_metrics.items()},
                **{f"ablated_{key}": value for key, value in ablated_metrics.items()},
                "macro_ablated_minus_full_rmse": ablated_metrics["rmse"]
                - full_metrics["rmse"],
                **block_bootstrap_difference(
                    observed,
                    full_prediction,
                    ablated_prediction,
                    blocks,
                    args.bootstrap_replicates,
                    args.seed + study_index,
                ),
            }
        )

    seeds = pd.DataFrame(seed_rows)
    ensembles = pd.DataFrame(ensemble_rows)
    seeds.to_csv(args.output_dir / "paired_seed_metrics.csv", index=False)
    ensembles.to_csv(args.output_dir / "paired_ensemble_metrics.csv", index=False)
    summary = {
        "design": {
            "full_dir": str(args.full_dir),
            "ablated_dir": str(args.ablated_dir),
            "spatial_blocks": args.spatial_blocks,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "paired_seed_metrics": seed_rows,
        "paired_ensemble_metrics": ensemble_rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(ensembles.to_string(index=False))


if __name__ == "__main__":
    main()
