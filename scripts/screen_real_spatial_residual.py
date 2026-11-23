from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gravity_gnn.semi_synthetic import relative_spatial_basis


CONTEXT_FEATURES = [
    "log_lsoa_area_km2",
    "lsoa_compactness",
    "defra_no2_2024",
    "defra_pm10_2024",
    "defra_so2_2024",
    "log_nhs_gp_nearest_m",
    "log_nhs_gp_count_1000m",
    "log_nhs_gp_count_3000m",
    "log_nhs_gp_count_5000m",
    "log_nhs_dentist_nearest_m",
    "log_nhs_dentist_count_1000m",
    "log_nhs_dentist_count_3000m",
    "log_nhs_dentist_count_5000m",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen controlled relative-coordinate residuals")
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_two_case_spatial_residual.json")
    )
    return parser


def build_estimator(name: str, seed: int):
    if name.endswith("ridge"):
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 6, 40)),
        )
    elif name.endswith("extra_trees"):
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=400,
                max_features=0.8,
                min_samples_leaf=8,
                n_jobs=-1,
                random_state=seed,
            ),
        )
    elif name.endswith("hist_gradient_boosting"):
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=3.0,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(name)
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return result


def main() -> None:
    args = build_parser().parse_args()
    bundles = {}
    for study in args.study:
        case_dir = args.case_root / study
        frame = gpd.read_parquet(case_dir / "official_context.parquet")
        points = frame.geometry.representative_point()
        coordinates = np.column_stack((points.x, points.y))
        context = frame[CONTEXT_FEATURES].to_numpy(dtype=float)
        spatial = relative_spatial_basis(coordinates)
        fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
        bundles[study] = {
            "context": context,
            "spatial": spatial,
            "combined": np.column_stack((context, spatial)),
            "y": frame["ahah"].to_numpy(dtype=float),
            "fold": frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int),
        }

    model_inputs = {
        "context_ridge": "context",
        "spatial_only_ridge": "spatial",
        "context_spatial_ridge": "combined",
        "context_spatial_extra_trees": "combined",
        "context_spatial_hist_gradient_boosting": "combined",
    }
    result = {
        "settings": {
            "studies": args.study,
            "models": model_inputs,
            "seed": args.seed,
            "target": "ahah",
            "coordinate_design": "city-relative polynomial + Fourier + RBF basis",
            "target_derived_features": False,
        },
        "within_city": {},
        "cross_city": {},
    }
    for study, bundle in bundles.items():
        study_result = {}
        for model_name, input_name in model_inputs.items():
            matrix = bundle[input_name]
            rows = []
            for fold in sorted(np.unique(bundle["fold"])):
                train = bundle["fold"] != fold
                test = ~train
                fitted = build_estimator(model_name, args.seed + int(fold)).fit(
                    matrix[train], bundle["y"][train]
                )
                rows.append(
                    {
                        "fold": int(fold),
                        **metrics(bundle["y"][test], fitted.predict(matrix[test])),
                    }
                )
            study_result[model_name] = {"folds": rows, "aggregate": aggregate(rows)}
        result["within_city"][study] = study_result

    for source_name, source in bundles.items():
        for target_name, target in bundles.items():
            if source_name == target_name:
                continue
            direction = {}
            for model_name, input_name in model_inputs.items():
                fitted = build_estimator(model_name, args.seed).fit(
                    source[input_name], source["y"]
                )
                direction[model_name] = metrics(
                    target["y"], fitted.predict(target[input_name])
                )
            result["cross_city"][f"{source_name}_to_{target_name}"] = direction

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
