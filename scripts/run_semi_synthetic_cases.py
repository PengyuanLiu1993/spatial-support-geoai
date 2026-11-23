from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gravity_gnn.semi_synthetic import (
    GRAVITY_COLUMNS,
    TRUE_SCALES_MINUTES,
    coordinate_features,
    gravity_features,
    multiscale_gravity_features,
)


ALPHAS = np.logspace(-4, 4, 25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark gravity recovery within and across semi-synthetic case studies"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--bundle-name", default="semi_synthetic_gravity.npz")
    parser.add_argument(
        "--output", type=Path, default=Path("results/semi_synthetic_two_case.json")
    )
    return parser


def features(bundle: dict[str, np.ndarray], model: str) -> np.ndarray:
    travel = bundle["travel_minutes"]
    coordinates = bundle["coordinates"]
    if model == "raw_ridge":
        return travel
    if model == "multiscale_gravity_ridge":
        return multiscale_gravity_features(travel)
    if model == "multiscale_gravity_spatial_ridge":
        return np.column_stack(
            (multiscale_gravity_features(travel), coordinate_features(coordinates))
        )
    if model == "oracle_gravity_ridge":
        return gravity_features(travel, GRAVITY_COLUMNS, TRUE_SCALES_MINUTES)
    raise ValueError(f"unknown model: {model}")


def estimator():
    regressor = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=ALPHAS),
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for key in ("r2", "rmse", "mae"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def main() -> None:
    args = build_parser().parse_args()
    bundles = {
        study: load_bundle(args.case_root / study / args.bundle_name) for study in args.study
    }
    model_names = (
        "raw_ridge",
        "multiscale_gravity_ridge",
        "multiscale_gravity_spatial_ridge",
        "oracle_gravity_ridge",
    )
    result = {
        "settings": {
            "studies": args.study,
            "bundle_name": args.bundle_name,
            "models": list(model_names),
            "primary_metric": "macro mean over fixed spatial folds",
        },
        "within_city": {},
        "cross_city": {},
    }

    for study, bundle in bundles.items():
        study_result = {}
        for model_name in model_names:
            matrix = features(bundle, model_name)
            rows = []
            for fold in sorted(np.unique(bundle["fold"])):
                train = bundle["fold"] != fold
                test = ~train
                fitted = estimator().fit(matrix[train], bundle["y"][train])
                prediction = fitted.predict(matrix[test])
                rows.append({"fold": int(fold), **metrics(bundle["y"][test], prediction)})
            study_result[model_name] = {"folds": rows, "aggregate": aggregate(rows)}
        result["within_city"][study] = study_result

    for source_name, source in bundles.items():
        for target_name, target in bundles.items():
            if source_name == target_name:
                continue
            direction = f"{source_name}_to_{target_name}"
            direction_result = {}
            for model_name in model_names:
                source_x = features(source, model_name)
                target_x = features(target, model_name)
                fitted = estimator().fit(source_x, source["y"])
                prediction = fitted.predict(target_x)
                direction_result[model_name] = metrics(target["y"], prediction)
            result["cross_city"][direction] = direction_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
