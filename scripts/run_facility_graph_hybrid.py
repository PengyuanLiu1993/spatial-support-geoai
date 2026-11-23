from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gravity_gnn.case_studies import sha256


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    auxiliary_weight: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


HGB_CONFIGS = (
    ("balanced15", 0.035, 400, 15, 25, 3.0),
    ("fine15", 0.025, 600, 15, 18, 4.0),
    ("wide31", 0.025, 550, 31, 25, 6.0),
    ("smooth7", 0.035, 500, 7, 18, 2.0),
)
AUXILIARY_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested spatial evaluation of a facility-gravity/HGB hybrid"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/facility_graph_hybrid.json")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    return parser


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for name in ("r2", "rmse", "mae"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return result


def load_case(case_root: Path, study: str) -> dict:
    case_dir = case_root / study
    sources = [
        ("official_context.parquet", "official_context_summary.json"),
        ("osm_context.parquet", "osm_context_summary.json"),
        ("os_green_context.parquet", "os_green_context_summary.json"),
        ("sentinel_ndvi_context.parquet", "sentinel_ndvi_context_summary.json"),
    ]
    frame = gpd.read_parquet(case_dir / sources[0][0])
    base_names: list[str] = []
    for position, (parquet_name, summary_name) in enumerate(sources):
        if position:
            extra = gpd.read_parquet(case_dir / parquet_name).drop(columns="geometry")
            frame = frame.merge(extra, on="lsoa21cd", validate="one_to_one")
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        base_names.extend(summary["features"])
    facility = gpd.read_parquet(case_dir / "facility_gravity_features.parquet").drop(
        columns="geometry"
    )
    frame = frame.merge(facility, on="lsoa21cd", validate="one_to_one")
    facility_names = [name for name in facility.columns if name != "lsoa21cd"]
    core_relations = ("nhs_gp", "osm_hospital", "os_green", "osm_fast_food")
    core_names = [
        name
        for name in facility_names
        if any(f"facility_{relation}_" in name for relation in core_relations)
    ]
    fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
    return {
        "frame": frame,
        "base_x": frame[base_names].to_numpy(dtype=float),
        "full_x": frame[base_names + facility_names].to_numpy(dtype=float),
        "core_x": frame[base_names + core_names].to_numpy(dtype=float),
        "y": frame["ahah"].to_numpy(dtype=float),
        "fold": frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int),
        "base_names": base_names,
        "facility_names": facility_names,
        "core_names": core_names,
    }


def candidates() -> tuple[Candidate, ...]:
    output = []
    for feature_set in ("core", "full"):
        for auxiliary_weight in AUXILIARY_WEIGHTS:
            for config in HGB_CONFIGS:
                name, learning_rate, max_iter, leaves, minimum, l2 = config
                output.append(
                    Candidate(
                        name=f"{feature_set}_{name}_aux{auxiliary_weight:g}",
                        feature_set=feature_set,
                        auxiliary_weight=auxiliary_weight,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        max_leaf_nodes=leaves,
                        min_samples_leaf=minimum,
                        l2_regularization=l2,
                    )
                )
    return tuple(output)


def design_matrix(
    bundles: dict[str, dict],
    studies: list[str],
    target_study: str,
    feature_set: str,
    target_selector: np.ndarray,
    auxiliary_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_blocks = []
    y_blocks = []
    weight_blocks = []
    for city_index, study in enumerate(studies):
        bundle = bundles[study]
        selected = (
            target_selector
            if study == target_study
            else np.ones(len(bundle["y"]), dtype=bool)
        )
        city = np.zeros((int(selected.sum()), len(studies)), dtype=float)
        city[:, city_index] = 1.0
        x_blocks.append(np.column_stack((bundle[f"{feature_set}_x"][selected], city)))
        y_blocks.append(bundle["y"][selected])
        weight_blocks.append(
            np.full(
                int(selected.sum()),
                1.0 if study == target_study else auxiliary_weight,
                dtype=float,
            )
        )
    return (
        np.concatenate(x_blocks),
        np.concatenate(y_blocks),
        np.concatenate(weight_blocks),
    )


def fit_candidate(
    candidate: Candidate,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    seed: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=candidate.learning_rate,
        max_iter=candidate.max_iter,
        max_leaf_nodes=candidate.max_leaf_nodes,
        min_samples_leaf=candidate.min_samples_leaf,
        l2_regularization=candidate.l2_regularization,
        random_state=seed,
    )
    return model.fit(train_x, train_y, sample_weight=train_weight)


def baseline_candidate() -> Candidate:
    return Candidate(
        name="frozen_context_hgb",
        feature_set="base",
        auxiliary_weight=1.0,
        learning_rate=0.035,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
    )


def main() -> None:
    args = build_parser().parse_args()
    if len(args.study) != 2:
        raise ValueError("this frozen design requires exactly two case studies")
    if args.top_candidates < 1:
        raise ValueError("top-candidates must be positive")
    bundles = {study: load_case(args.case_root, study) for study in args.study}
    if bundles[args.study[0]]["base_names"] != bundles[args.study[1]]["base_names"]:
        raise ValueError("base feature schemas differ")
    if bundles[args.study[0]]["facility_names"] != bundles[args.study[1]]["facility_names"]:
        raise ValueError("facility feature schemas differ")

    candidate_pool = candidates()
    result = {
        "settings": {
            "studies": args.study,
            "target_derived_features": False,
            "absolute_coordinate_features": False,
            "outer_validation": "five frozen contiguous spatial folds",
            "inner_selection": (
                "the next target-city spatial fold selects candidates and blend weight; "
                "the held-out outer fold is never used for selection"
            ),
            "candidate_count": len(candidate_pool),
            "top_candidates": args.top_candidates,
            "blend_weights": BLEND_WEIGHTS,
            "facility_feature_count": len(bundles[args.study[0]]["facility_names"]),
            "core_facility_relations": [
                "nhs_gp",
                "osm_hospital",
                "os_green",
                "osm_fast_food",
            ],
        },
        "within_city": {},
        "oof_outputs": {},
    }
    args.oof_dir.mkdir(parents=True, exist_ok=True)

    for target_index, target_study in enumerate(args.study):
        bundle = bundles[target_study]
        prediction = np.full(len(bundle["y"]), np.nan, dtype=float)
        fold_records = []
        unique_folds = sorted(int(value) for value in np.unique(bundle["fold"]))
        for fold_position, test_fold in enumerate(unique_folds):
            validation_fold = unique_folds[(fold_position + 1) % len(unique_folds)]
            inner_train = (bundle["fold"] != test_fold) & (
                bundle["fold"] != validation_fold
            )
            validation = bundle["fold"] == validation_fold
            outer_train = bundle["fold"] != test_fold
            test = bundle["fold"] == test_fold
            validation_records = []

            for candidate_index, candidate in enumerate(candidate_pool):
                train_x, train_y, train_weight = design_matrix(
                    bundles,
                    args.study,
                    target_study,
                    candidate.feature_set,
                    inner_train,
                    candidate.auxiliary_weight,
                )
                validation_city = np.zeros((int(validation.sum()), len(args.study)))
                validation_city[:, target_index] = 1.0
                validation_x = np.column_stack(
                    (bundle[f"{candidate.feature_set}_x"][validation], validation_city)
                )
                fitted = fit_candidate(
                    candidate,
                    train_x,
                    train_y,
                    train_weight,
                    args.seed + test_fold * 1000 + candidate_index,
                )
                candidate_prediction = fitted.predict(validation_x)
                validation_records.append(
                    {
                        "candidate": candidate,
                        "prediction": candidate_prediction,
                        "rmse": float(
                            mean_squared_error(bundle["y"][validation], candidate_prediction)
                            ** 0.5
                        ),
                    }
                )

            selected = sorted(validation_records, key=lambda row: row["rmse"])[
                : args.top_candidates
            ]
            graph_validation_prediction = np.mean(
                [row["prediction"] for row in selected], axis=0
            )
            base = baseline_candidate()
            base_train_x, base_train_y, base_train_weight = design_matrix(
                bundles,
                args.study,
                target_study,
                "base",
                inner_train,
                1.0,
            )
            validation_city = np.zeros((int(validation.sum()), len(args.study)))
            validation_city[:, target_index] = 1.0
            base_validation_x = np.column_stack(
                (bundle["base_x"][validation], validation_city)
            )
            base_fitted = fit_candidate(
                base,
                base_train_x,
                base_train_y,
                base_train_weight,
                args.seed + test_fold * 1000 + 999,
            )
            base_validation_prediction = base_fitted.predict(base_validation_x)
            blend_scores = []
            for graph_weight in BLEND_WEIGHTS:
                blended = (
                    (1.0 - graph_weight) * base_validation_prediction
                    + graph_weight * graph_validation_prediction
                )
                blend_scores.append(
                    (
                        float(mean_squared_error(bundle["y"][validation], blended) ** 0.5),
                        graph_weight,
                    )
                )
            _, graph_weight = min(blend_scores)

            test_city = np.zeros((int(test.sum()), len(args.study)))
            test_city[:, target_index] = 1.0
            graph_test_predictions = []
            for selected_index, row in enumerate(selected):
                candidate = row["candidate"]
                train_x, train_y, train_weight = design_matrix(
                    bundles,
                    args.study,
                    target_study,
                    candidate.feature_set,
                    outer_train,
                    candidate.auxiliary_weight,
                )
                test_x = np.column_stack(
                    (bundle[f"{candidate.feature_set}_x"][test], test_city)
                )
                fitted = fit_candidate(
                    candidate,
                    train_x,
                    train_y,
                    train_weight,
                    args.seed + test_fold * 1000 + 2000 + selected_index,
                )
                graph_test_predictions.append(fitted.predict(test_x))
            graph_test_prediction = np.mean(graph_test_predictions, axis=0)

            base_train_x, base_train_y, base_train_weight = design_matrix(
                bundles,
                args.study,
                target_study,
                "base",
                outer_train,
                1.0,
            )
            base_test_x = np.column_stack((bundle["base_x"][test], test_city))
            base_fitted = fit_candidate(
                base,
                base_train_x,
                base_train_y,
                base_train_weight,
                args.seed + test_fold * 1000 + 2999,
            )
            base_test_prediction = base_fitted.predict(base_test_x)
            fold_prediction = (
                (1.0 - graph_weight) * base_test_prediction
                + graph_weight * graph_test_prediction
            )
            prediction[test] = fold_prediction
            fold_records.append(
                {
                    "fold": test_fold,
                    "validation_fold": validation_fold,
                    "selected_candidates": [
                        {
                            "name": row["candidate"].name,
                            "inner_validation_rmse": row["rmse"],
                        }
                        for row in selected
                    ],
                    "selected_graph_blend_weight": graph_weight,
                    **metrics(bundle["y"][test], fold_prediction),
                }
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"incomplete OOF prediction for {target_study}")
        result["within_city"][target_study] = {
            "folds": fold_records,
            "macro": aggregate(fold_records),
            "pooled_oof": metrics(bundle["y"], prediction),
        }
        oof_path = args.oof_dir / f"{target_study}.parquet"
        oof = gpd.read_parquet(oof_path)
        if not np.array_equal(
            oof["lsoa21cd"].to_numpy(), bundle["frame"]["lsoa21cd"].to_numpy()
        ):
            raise ValueError(f"OOF row order differs for {target_study}")
        oof["prediction_facility_gravity_hybrid"] = prediction
        oof["residual_facility_gravity_hybrid"] = bundle["y"] - prediction
        oof.to_parquet(oof_path, index=False)
        result["oof_outputs"][target_study] = {
            "path": str(oof_path.resolve()),
            "rows": len(oof),
            "sha256": sha256(oof_path),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                study: {
                    "r2": values["macro"]["r2"]["mean"],
                    "rmse": values["macro"]["rmse"]["mean"],
                }
                for study, values in result["within_city"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
