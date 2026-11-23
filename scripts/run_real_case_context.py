from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ALPHAS = np.logspace(-4, 5, 30)
POLLUTION_FEATURES = ["defra_no2_2024", "defra_pm10_2024", "defra_so2_2024"]
HEALTH_FEATURES = [
    "log_nhs_gp_nearest_m",
    "log_nhs_gp_count_1000m",
    "log_nhs_gp_count_3000m",
    "log_nhs_gp_count_5000m",
    "log_nhs_dentist_nearest_m",
    "log_nhs_dentist_count_1000m",
    "log_nhs_dentist_count_3000m",
    "log_nhs_dentist_count_5000m",
]
SHAPE_FEATURES = ["log_lsoa_area_km2", "lsoa_compactness"]
RAW_COMPONENTS = [
    "GP",
    "dentist",
    "pharmacy",
    "hospital",
    "leisure",
    "greenspace_active",
    "greenspace",
    "bluespace",
    "NO2",
    "PM10",
    "SO2",
    "fast_food",
    "gambling",
    "pub_bar",
    "tobacco",
]
DOMAIN_COLUMNS = ["domain_h", "domain_g", "domain_e", "domain_r"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate real official context and labelled AHAH reconstruction ceilings"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_two_case_context.json")
    )
    return parser


def estimator():
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS)
        ),
        transformer=StandardScaler(),
    )


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in rows])
        output[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return output


def main() -> None:
    args = build_parser().parse_args()
    bundles = {}
    for study in args.study:
        case_dir = args.case_root / study
        frame = gpd.read_parquet(case_dir / "official_context.parquet")
        fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
        fold = frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int)
        bundles[study] = {"frame": frame, "fold": fold}

    experiments = {
        "domain_e_official_pollution": ("domain_e", POLLUTION_FEATURES),
        "domain_h_partial_official_health": ("domain_h", HEALTH_FEATURES),
        "ahah_partial_official_context": (
            "ahah",
            SHAPE_FEATURES + POLLUTION_FEATURES + HEALTH_FEATURES,
        ),
        # These are index-reconstruction ceilings, not deployable predictors.
        "ahah_published_components_oracle": ("ahah", RAW_COMPONENTS),
        "ahah_published_domains_oracle": ("ahah", DOMAIN_COLUMNS),
    }
    result = {
        "settings": {
            "studies": args.study,
            "primary_metric": "macro mean over fixed spatial folds",
            "oracle_warning": (
                "Published AHAH components/domains are target-derived reconstruction inputs and "
                "must not be reported as predictive model features."
            ),
        },
        "within_city": {},
        "cross_city": {},
    }

    for study, bundle in bundles.items():
        frame = bundle["frame"]
        study_result = {}
        for name, (target, feature_names) in experiments.items():
            matrix = frame[feature_names].to_numpy(dtype=float)
            values = frame[target].to_numpy(dtype=float)
            rows = []
            for fold in sorted(np.unique(bundle["fold"])):
                train = bundle["fold"] != fold
                test = ~train
                fitted = estimator().fit(matrix[train], values[train])
                prediction = fitted.predict(matrix[test])
                rows.append({"fold": int(fold), **metrics(values[test], prediction)})
            study_result[name] = {"folds": rows, "aggregate": aggregate(rows)}
        result["within_city"][study] = study_result

    for source_name, source in bundles.items():
        for target_name, target_bundle in bundles.items():
            if source_name == target_name:
                continue
            direction_result = {}
            for name, (target_column, feature_names) in experiments.items():
                source_x = source["frame"][feature_names].to_numpy(dtype=float)
                target_x = target_bundle["frame"][feature_names].to_numpy(dtype=float)
                source_y = source["frame"][target_column].to_numpy(dtype=float)
                target_y = target_bundle["frame"][target_column].to_numpy(dtype=float)
                fitted = estimator().fit(source_x, source_y)
                direction_result[name] = metrics(target_y, fitted.predict(target_x))
            result["cross_city"][f"{source_name}_to_{target_name}"] = direction_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
