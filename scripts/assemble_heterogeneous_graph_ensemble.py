from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


STUDIES = ("greater_london", "greater_manchester")
PROPOSED_WEIGHT = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble the fixed context-TabPFN/heterogeneous-GNN ensemble"
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_predictions_gated_gravity_cv5"
        ),
    )
    parser.add_argument(
        "--hard-gravity-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions_hgb_residual_cv5"),
    )
    parser.add_argument(
        "--no-gravity-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions_no_gravity_cv5"),
    )
    parser.add_argument(
        "--tabpfn-dir",
        type=Path,
        default=Path("results/tabpfn_gravity/screen_v2_e1_oof"),
    )
    parser.add_argument(
        "--tabpfn-estimators",
        type=int,
        default=1,
        help="Number of estimators used for the supplied TabPFN OOF predictions.",
    )
    parser.add_argument(
        "--nested-reference",
        type=Path,
        default=Path("results/tabpfn_gravity/nested_blend_v2_e1.json"),
        help="Strict nested Gravity HGB--TabPFN comparator summary.",
    )
    parser.add_argument(
        "--distance-oof-dir",
        type=Path,
        default=Path("results/distance_effect/oof"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/heterogeneous_graph_ensemble"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/heterogeneous_graph_ensemble.json"),
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


def evaluate(
    observed: np.ndarray, prediction: np.ndarray, fold: np.ndarray
) -> dict[str, object]:
    fold_rows = [
        {
            "fold": int(value),
            **metrics(observed[fold == value], prediction[fold == value]),
        }
        for value in sorted(np.unique(fold))
    ]
    return {
        "folds": fold_rows,
        "macro": {
            name: {
                "mean": float(np.mean([row[name] for row in fold_rows])),
                "std": float(np.std([row[name] for row in fold_rows], ddof=1)),
            }
            for name in ("r2", "rmse", "mae")
        },
        "pooled_oof": metrics(observed, prediction),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "settings": {
            "proposed_model": (
                "fixed equal-weight context-only TabPFN plus context-HGB with "
                "soft-gated heterogeneous facility Gravity-GNN residual"
            ),
            "tabpfn_weight": PROPOSED_WEIGHT,
            "heterogeneous_graph_hybrid_weight": 1.0 - PROPOSED_WEIGHT,
            "facility_representation": (
                "facility nodes and routed travel-time edges enter only through "
                "relation-preserving vector message passing; no aggregated gravity "
                "features enter HGB or TabPFN"
            ),
            "evaluation": "five frozen contiguous spatial folds",
            "status": (
                "frozen comparative candidate evaluated with repeated neural seeds; "
                "a new locked case study remains external confirmatory work"
            ),
            "tabpfn_estimators": int(args.tabpfn_estimators),
            "absolute_coordinate_features": False,
            "target_derived_features": False,
        },
        "studies": {},
        "outputs": {},
    }
    fold_table_rows: list[dict[str, object]] = []
    macro_table_rows: list[dict[str, object]] = []
    seed_table_rows: list[dict[str, object]] = []

    for study in STUDIES:
        graph_path = args.graph_dir / f"{study}.npz"
        hard_gravity_path = args.hard_gravity_dir / f"{study}.npz"
        no_gravity_path = args.no_gravity_dir / f"{study}.npz"
        tabpfn_path = args.tabpfn_dir / f"{study}_context_only.csv"
        reference_path = args.distance_oof_dir / f"{study}.parquet"
        graph = np.load(graph_path, allow_pickle=False)
        hard_gravity = np.load(hard_gravity_path, allow_pickle=False)
        no_gravity = np.load(no_gravity_path, allow_pickle=False)
        tabpfn = pd.read_csv(tabpfn_path)
        reference = gpd.read_parquet(reference_path)
        codes = graph["lsoa_codes"].astype(str)
        if not np.array_equal(codes, tabpfn["lsoa21cd"].astype(str).to_numpy()):
            raise ValueError(f"graph and TabPFN row order differs for {study}")
        if not np.array_equal(codes, reference["lsoa21cd"].astype(str).to_numpy()):
            raise ValueError(f"graph and reference row order differs for {study}")
        observed = graph["observed"].astype(float)
        folds = graph["fold"].astype(int)
        if not np.allclose(observed, tabpfn["observed"].to_numpy(dtype=float)):
            raise ValueError(f"observed targets differ for {study}")
        graph_prediction = graph["ensemble_prediction"].astype(float)
        tabpfn_prediction = tabpfn["prediction"].to_numpy(dtype=float)
        proposed_prediction = (
            PROPOSED_WEIGHT * tabpfn_prediction
            + (1.0 - PROPOSED_WEIGHT) * graph_prediction
        )
        models = {
            "context_hgb": reference["prediction_context_only"].to_numpy(dtype=float),
            "aggregated_gravity_hgb_scale15": reference[
                "prediction_gravity_scale_15"
            ].to_numpy(dtype=float),
            "context_tabpfn": tabpfn_prediction,
            "heterogeneous_facility_hard_gravity_gnn_hybrid": hard_gravity[
                "ensemble_prediction"
            ].astype(float),
            "heterogeneous_facility_no_gravity_gnn_hybrid": no_gravity[
                "ensemble_prediction"
            ].astype(float),
            "heterogeneous_facility_gated_gravity_gnn_hybrid": graph_prediction,
            "proposed_fixed_equal_ensemble": proposed_prediction,
        }
        evaluations = {
            name: evaluate(observed, prediction, folds)
            for name, prediction in models.items()
        }
        proposed_macro = evaluations["proposed_fixed_equal_ensemble"]["macro"]
        strongest_prior = {
            "r2": max(
                evaluations[name]["macro"]["r2"]["mean"]
                for name in models
                if name != "proposed_fixed_equal_ensemble"
            ),
            "rmse": min(
                evaluations[name]["macro"]["rmse"]["mean"]
                for name in models
                if name != "proposed_fixed_equal_ensemble"
            ),
        }
        result["studies"][study] = {
            "models": evaluations,
            "delta_vs_strongest_component": {
                "r2": proposed_macro["r2"]["mean"] - strongest_prior["r2"],
                "rmse": proposed_macro["rmse"]["mean"] - strongest_prior["rmse"],
            },
        }
        if "seed_predictions" in graph.files and "seed" in graph.files:
            seeds = graph["seed"].astype(int)
            graph_seed_predictions = graph["seed_predictions"].astype(float)
            if graph_seed_predictions.shape != (len(seeds), len(observed)):
                raise ValueError(f"invalid seed prediction shape for {study}")
            seed_records: list[dict[str, object]] = []
            for seed, graph_seed_prediction in zip(
                seeds, graph_seed_predictions, strict=True
            ):
                prediction = (
                    PROPOSED_WEIGHT * tabpfn_prediction
                    + (1.0 - PROPOSED_WEIGHT) * graph_seed_prediction
                )
                seed_evaluation = evaluate(observed, prediction, folds)
                record = {
                    "study": study,
                    "seed": int(seed),
                    **{
                        name: seed_evaluation["macro"][name]["mean"]
                        for name in ("r2", "rmse", "mae")
                    },
                }
                seed_records.append(record)
                seed_table_rows.append(record)
            result["studies"][study]["seed_robustness"] = {
                "per_seed_macro": seed_records,
                "mean": {
                    name: float(np.mean([row[name] for row in seed_records]))
                    for name in ("r2", "rmse", "mae")
                },
                "sd": {
                    name: float(np.std([row[name] for row in seed_records], ddof=1))
                    if len(seed_records) > 1
                    else 0.0
                    for name in ("r2", "rmse", "mae")
                },
                "minimum": {
                    name: float(np.min([row[name] for row in seed_records]))
                    for name in ("r2", "rmse", "mae")
                },
                "maximum": {
                    name: float(np.max([row[name] for row in seed_records]))
                    for name in ("r2", "rmse", "mae")
                },
            }
        for model_name, evaluation in evaluations.items():
            for row in evaluation["folds"]:
                fold_table_rows.append(
                    {"study": study, "model": model_name, **row}
                )
            macro_table_rows.append(
                {
                    "study": study,
                    "model": model_name,
                    **{
                        name: evaluation["macro"][name]["mean"]
                        for name in ("r2", "rmse", "mae")
                    },
                }
            )

        output = reference[["lsoa21cd", "geometry", "fold", "observed"]].copy()
        output["prediction_context_tabpfn"] = tabpfn_prediction
        output["prediction_heterogeneous_facility_gated_gravity_gnn_hybrid"] = (
            graph_prediction
        )
        output["prediction_heterogeneous_graph_ensemble"] = proposed_prediction
        output["residual_heterogeneous_graph_ensemble"] = (
            observed - proposed_prediction
        )
        output_path = args.output_dir / f"{study}.parquet"
        output.to_parquet(output_path, index=False)
        result["outputs"][study] = {
            "path": str(output_path.resolve()),
            "rows": len(output),
            "sha256": sha256(output_path),
        }

    nested_reference = json.loads(args.nested_reference.read_text(encoding="utf-8"))
    result["prior_strict_nested_gravity_hgb_tabpfn"] = {
        study: nested_reference["studies"][study]["macro"] for study in STUDIES
    }
    for study in STUDIES:
        proposed_macro = result["studies"][study]["models"][
            "proposed_fixed_equal_ensemble"
        ]["macro"]
        prior_macro = result["prior_strict_nested_gravity_hgb_tabpfn"][study]
        result["studies"][study]["delta_vs_prior_strict_nested_candidate"] = {
            "r2": proposed_macro["r2"]["mean"] - prior_macro["r2"]["mean"],
            "rmse": proposed_macro["rmse"]["mean"] - prior_macro["rmse"]["mean"],
            "mae": proposed_macro["mae"]["mean"] - prior_macro["mae"]["mean"],
        }
    fold_table_path = args.output_dir / "performance_by_fold.csv"
    macro_table_path = args.output_dir / "performance_macro.csv"
    pd.DataFrame(fold_table_rows).to_csv(fold_table_path, index=False)
    pd.DataFrame(macro_table_rows).to_csv(macro_table_path, index=False)
    result["outputs"]["performance_by_fold"] = {
        "path": str(fold_table_path.resolve()),
        "sha256": sha256(fold_table_path),
    }
    result["outputs"]["performance_macro"] = {
        "path": str(macro_table_path.resolve()),
        "sha256": sha256(macro_table_path),
    }
    if seed_table_rows:
        seed_table_path = args.output_dir / "performance_by_seed.csv"
        pd.DataFrame(seed_table_rows).to_csv(seed_table_path, index=False)
        result["outputs"]["performance_by_seed"] = {
            "path": str(seed_table_path.resolve()),
            "sha256": sha256(seed_table_path),
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                study: {
                    metric: result["studies"][study]["models"][
                        "proposed_fixed_equal_ensemble"
                    ]["macro"][metric]["mean"]
                    for metric in ("r2", "rmse", "mae")
                }
                for study in STUDIES
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
