from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDIES = ("greater_london", "greater_manchester")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge disjoint HGGE seed runs without retraining completed seeds."
    )
    parser.add_argument(
        "--prediction-dir",
        action="append",
        type=Path,
        required=True,
        help="Prediction directory to merge; repeat for every disjoint seed run.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        type=Path,
        default=[],
        help="Optional run JSON corresponding to a prediction directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions_gated_gravity_cv5_relation_access_seeds5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/heterogeneous_facility_gated_gravity_gnn_hgb_residual_cv5_relation_access_seeds5.json"),
    )
    return parser


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - observed
    total = np.sum((observed - observed.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / max(float(total), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(payloads) < 2:
        raise ValueError("at least two seed payloads are required")
    reference = payloads[0]
    # Relation counterfactuals can be merged only when every source exported them.
    # Mixed sources remain valid for prediction-only robustness comparisons; in
    # that case omit the incomplete auxiliary arrays rather than discarding seeds.
    relation_access_exported = all(
        "seed_relation_access" in payload for payload in payloads
    )
    invariant_keys = ["lsoa_codes", "observed", "fold"]
    if relation_access_exported:
        invariant_keys.extend(("relation_names", "relation_access_shift_minutes"))
    for payload in payloads[1:]:
        for key in invariant_keys:
            if not np.array_equal(reference[key], payload[key]):
                raise ValueError(f"seed payloads differ for {key}")

    seeds = np.concatenate([payload["seed"].astype(int) for payload in payloads])
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("seed payloads contain duplicate seeds")
    order = np.argsort(seeds)
    seed_predictions = np.concatenate(
        [payload["seed_predictions"].astype(float) for payload in payloads], axis=0
    )[order]
    if not np.isfinite(seed_predictions).all():
        raise ValueError("merged seed predictions are incomplete")

    merged = {
        "lsoa_codes": reference["lsoa_codes"],
        "observed": reference["observed"].astype(float),
        "fold": reference["fold"].astype(int),
        "seed": seeds[order],
        "seed_predictions": seed_predictions,
        "ensemble_prediction": np.mean(seed_predictions, axis=0),
    }
    if relation_access_exported:
        seed_relation_access = np.concatenate(
            [payload["seed_relation_access"].astype(float) for payload in payloads],
            axis=0,
        )[order]
        if seed_relation_access.size and not np.isfinite(seed_relation_access).all():
            raise ValueError("merged relation counterfactuals are incomplete")
        merged.update(
            {
                "relation_names": reference["relation_names"],
                "relation_access_shift_minutes": reference[
                    "relation_access_shift_minutes"
                ].astype(float),
                "seed_relation_access": seed_relation_access,
                "ensemble_relation_access": np.mean(seed_relation_access, axis=0),
            }
        )
    return merged


def main() -> None:
    args = build_parser().parse_args()
    if args.metadata and len(args.metadata) != len(args.prediction_dir):
        raise ValueError("metadata count must match prediction-directory count")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.metadata
    ]
    summary: dict[str, object] = {
        "settings": {
            "design": "five-seed strict-OOF HGGE robustness ensemble",
            "prediction_dirs": [str(path.resolve()) for path in args.prediction_dir],
            "metadata": [str(path.resolve()) for path in args.metadata],
        },
        "within_city": {},
        "prediction_outputs": {},
    }
    performance_rows: list[dict[str, object]] = []

    for study in STUDIES:
        loaded: list[dict[str, np.ndarray]] = []
        for directory in args.prediction_dir:
            with np.load(directory / f"{study}.npz", allow_pickle=False) as source:
                loaded.append({key: source[key] for key in source.files})
        merged = merge_payloads(loaded)
        output_path = args.output_dir / f"{study}.npz"
        np.savez_compressed(output_path, **merged)

        observed = merged["observed"]
        folds = merged["fold"]
        per_seed: list[dict[str, object]] = []
        for seed, prediction in zip(
            merged["seed"], merged["seed_predictions"], strict=True
        ):
            row = {
                "study": study,
                "seed": int(seed),
                **macro_metrics(observed, prediction, folds),
            }
            per_seed.append(row)
            performance_rows.append(row)
        ensemble_macro = macro_metrics(
            observed, merged["ensemble_prediction"], folds
        )
        summary["within_city"][study] = {
            "seeds": merged["seed"].astype(int).tolist(),
            "per_seed_macro": per_seed,
            "seed_mean": {
                name: float(np.mean([row[name] for row in per_seed]))
                for name in ("r2", "rmse", "mae")
            },
            "seed_sd": {
                name: float(np.std([row[name] for row in per_seed], ddof=1))
                for name in ("r2", "rmse", "mae")
            },
            "ensemble_macro": ensemble_macro,
            "ensemble_pooled_oof": metrics(
                observed, merged["ensemble_prediction"]
            ),
            "source_seed_records": [
                record
                for source in metadata
                for record in source["within_city"][study]["seed_runs"]
            ],
        }
        summary["prediction_outputs"][study] = {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "complete_oof": True,
            "relation_access_exported": "seed_relation_access" in merged,
        }

    performance_path = args.output_dir / "graph_hybrid_performance_by_seed.csv"
    pd.DataFrame(performance_rows).to_csv(performance_path, index=False)
    summary["performance_by_seed"] = {
        "path": str(performance_path.resolve()),
        "sha256": sha256(performance_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                study: {
                    key: value
                    for key, value in city.items()
                    if key != "source_seed_records"
                }
                for study, city in summary["within_city"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
