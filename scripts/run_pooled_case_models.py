from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gravity_gnn.spatial import gravity_neighborhood_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pooled hierarchical two-city evaluation")
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--target", default="ahah")
    parser.add_argument("--include-gravity-neighborhood", action="store_true")
    parser.add_argument("--include-official-gambling", action="store_true")
    parser.add_argument("--include-osm-blue", action="store_true")
    parser.add_argument(
        "--models",
        default=(
            "ridge_raw,ridge_city_standardised,hist_gradient_boosting_raw,"
            "hist_gradient_boosting_city_standardised,extra_trees_city_standardised"
        ),
    )
    parser.add_argument(
        "--auxiliary-weights",
        default="1.0",
        help="Comma-separated sample weights for observations from auxiliary cities",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_two_case_pooled_models.json")
    )
    return parser


def build_estimator(name: str, seed: int):
    if name.startswith("ridge"):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 6, 40)),
        )
    if name.startswith("hist_gradient_boosting"):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=400,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=3.0,
                random_state=seed,
            ),
        )
    if name.startswith("extra_trees"):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=400,
                max_features=0.8,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=seed,
            ),
        )
    raise ValueError(name)


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


def load_case(
    case_root: Path,
    study: str,
    target: str,
    include_gravity_neighborhood: bool,
    include_official_gambling: bool,
    include_osm_blue: bool,
) -> dict:
    case_dir = case_root / study
    sources = [
        ("official_context.parquet", "official_context_summary.json"),
        ("osm_context.parquet", "osm_context_summary.json"),
        ("os_green_context.parquet", "os_green_context_summary.json"),
        ("sentinel_ndvi_context.parquet", "sentinel_ndvi_context_summary.json"),
    ]
    if include_official_gambling:
        sources.append(
            ("official_gambling_context.parquet", "official_gambling_context_summary.json")
        )
    if include_osm_blue:
        sources.append(("osm_blue_context.parquet", "osm_blue_context_summary.json"))

    frame = gpd.read_parquet(case_dir / sources[0][0])
    feature_names = []
    for position, (parquet_name, summary_name) in enumerate(sources):
        if position:
            extra = gpd.read_parquet(case_dir / parquet_name).drop(columns="geometry")
            frame = frame.merge(extra, on="lsoa21cd", validate="one_to_one")
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        feature_names.extend(summary["features"])
    x = frame[feature_names].to_numpy(dtype=float)
    gravity_names: list[str] = []
    if include_gravity_neighborhood:
        core_names = [
            "defra_no2_2024",
            "defra_pm10_2024",
            "defra_so2_2024",
            "log_osm_buildings_per_km2",
            "log_nhs_gp_nearest_m",
            "log_nhs_dentist_nearest_m",
            "os_green_coverage_fraction",
            "os_active_green_coverage_fraction",
            "s2_ndvi_mean",
            "s2_ndvi_fraction_ge_0_4",
            "log_osm_fast_food_count_1000m",
            "log_osm_gambling_count_1000m",
            "log_osm_pub_bar_count_1000m",
            "log_osm_tobacco_count_1000m",
        ]
        core_indices = [feature_names.index(name) for name in core_names]
        points = frame.geometry.representative_point()
        coordinates = np.column_stack((points.x, points.y))
        building_index = feature_names.index("log_osm_building_count")
        source_mass = np.expm1(np.maximum(x[:, building_index], 0.0))
        scales = (1000.0, 3000.0, 5000.0, 10000.0)
        gravity_x = gravity_neighborhood_features(
            coordinates,
            x[:, core_indices],
            source_mass,
            scales_m=scales,
            neighbors=256,
        )
        for scale in scales:
            gravity_names.append(f"gravity_potential_{int(scale)}m")
            gravity_names.extend(
                f"gravity_neighbor_mean_{name}_{int(scale)}m" for name in core_names
            )
            gravity_names.extend(
                f"gravity_local_contrast_{name}_{int(scale)}m" for name in core_names
            )
        x = np.column_stack((x, gravity_x))
        feature_names.extend(gravity_names)
    fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
    return {
        "x": x,
        "y": frame[target].to_numpy(dtype=float),
        "fold": frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int),
        "feature_names": feature_names,
    }


def main() -> None:
    args = build_parser().parse_args()
    if len(args.study) < 2:
        raise ValueError("pooled evaluation requires at least two studies")
    bundles = {
        study: load_case(
            args.case_root,
            study,
            args.target,
            args.include_gravity_neighborhood,
            args.include_official_gambling,
            args.include_osm_blue,
        )
        for study in args.study
    }
    feature_names = bundles[args.study[0]]["feature_names"]
    if any(bundle["feature_names"] != feature_names for bundle in bundles.values()):
        raise ValueError("case-study feature schemas differ")

    base_model_names = tuple(
        name.strip() for name in args.models.split(",") if name.strip()
    )
    auxiliary_weights = tuple(
        float(value) for value in args.auxiliary_weights.split(",") if value.strip()
    )
    if not auxiliary_weights or any(weight <= 0 for weight in auxiliary_weights):
        raise ValueError("auxiliary weights must be positive")
    model_names = tuple(
        f"{name}__auxw_{weight:g}"
        for weight in auxiliary_weights
        for name in base_model_names
    )
    result = {
        "settings": {
            "studies": args.study,
            "target": args.target,
            "models": list(model_names),
            "features": feature_names,
            "target_derived_features": False,
            "design": "target-city spatial fold plus all auxiliary-city observations",
            "gravity_neighborhood": args.include_gravity_neighborhood,
            "official_gambling": args.include_official_gambling,
            "osm_blue": args.include_osm_blue,
        },
        "within_city": {},
    }

    for target_index, target_study in enumerate(args.study):
        target_bundle = bundles[target_study]
        study_result = {}
        for model_name in model_names:
            rows = []
            base_model_name, weight_text = model_name.rsplit("__auxw_", 1)
            auxiliary_weight = float(weight_text)
            city_standardised = base_model_name.endswith("city_standardised")
            for fold in sorted(np.unique(target_bundle["fold"])):
                train_blocks = []
                target_blocks = []
                weight_blocks = []
                city_blocks = []
                target_location = None
                target_scale = None
                for city_index, study in enumerate(args.study):
                    bundle = bundles[study]
                    selected = bundle["fold"] != fold if study == target_study else np.ones(
                        len(bundle["y"]), dtype=bool
                    )
                    city_column = np.zeros((int(selected.sum()), len(args.study)), dtype=float)
                    city_column[:, city_index] = 1.0
                    train_blocks.append(np.column_stack((bundle["x"][selected], city_column)))
                    city_y = bundle["y"][selected]
                    location = float(city_y.mean())
                    scale = max(float(city_y.std(ddof=0)), 1e-6)
                    target_blocks.append((city_y - location) / scale if city_standardised else city_y)
                    weight_blocks.append(
                        np.full(
                            len(city_y),
                            1.0 if study == target_study else auxiliary_weight,
                            dtype=float,
                        )
                    )
                    city_blocks.append(study)
                    if study == target_study:
                        target_location = location
                        target_scale = scale
                train_x = np.concatenate(train_blocks, axis=0)
                train_y = np.concatenate(target_blocks)
                train_weight = np.concatenate(weight_blocks)
                test = target_bundle["fold"] == fold
                test_city = np.zeros((int(test.sum()), len(args.study)), dtype=float)
                test_city[:, target_index] = 1.0
                test_x = np.column_stack((target_bundle["x"][test], test_city))
                estimator = build_estimator(base_model_name, args.seed + int(fold))
                if auxiliary_weight == 1.0:
                    fitted = estimator.fit(train_x, train_y)
                elif base_model_name.startswith("hist_gradient_boosting"):
                    fitted = estimator.fit(
                        train_x,
                        train_y,
                        histgradientboostingregressor__sample_weight=train_weight,
                    )
                elif base_model_name.startswith("extra_trees"):
                    fitted = estimator.fit(
                        train_x,
                        train_y,
                        extratreesregressor__sample_weight=train_weight,
                    )
                elif base_model_name.startswith("ridge"):
                    fitted = estimator.fit(
                        train_x, train_y, ridgecv__sample_weight=train_weight
                    )
                else:
                    raise ValueError(base_model_name)
                prediction = fitted.predict(test_x)
                if city_standardised:
                    prediction = prediction * target_scale + target_location
                rows.append(
                    {
                        "fold": int(fold),
                        **metrics(target_bundle["y"][test], prediction),
                    }
                )
            study_result[model_name] = {"folds": rows, "aggregate": aggregate(rows)}
        result["within_city"][target_study] = study_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
