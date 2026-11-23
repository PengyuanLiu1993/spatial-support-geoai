from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen official + OSM gravity context models")
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--target", default="ahah")
    parser.add_argument("--include-os-green", action="store_true")
    parser.add_argument("--include-sentinel", action="store_true")
    parser.add_argument("--include-official-gambling", action="store_true")
    parser.add_argument("--include-osm-blue", action="store_true")
    parser.add_argument(
        "--models",
        default=(
            "ridge,hist_gradient_boosting,extra_trees_leaf4,extra_trees_leaf8,"
            "extra_trees_leaf16,random_forest_leaf4,random_forest_leaf8"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_two_case_osm_context.json")
    )
    return parser


def build_estimator(name: str, seed: int):
    if name == "ridge":
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 6, 40)),
        )
    elif name == "hist_gradient_boosting":
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=350,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=3.0,
                random_state=seed,
            ),
        )
    elif name.startswith("extra_trees_leaf"):
        leaf = int(name.removeprefix("extra_trees_leaf"))
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=350,
                max_features=0.8,
                min_samples_leaf=leaf,
                n_jobs=-1,
                random_state=seed,
            ),
        )
    elif name.startswith("random_forest_leaf"):
        leaf = int(name.removeprefix("random_forest_leaf"))
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=350,
                max_features=0.8,
                min_samples_leaf=leaf,
                n_jobs=-1,
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
    official_features: list[str] | None = None
    osm_features: list[str] | None = None
    green_features: list[str] | None = None
    sentinel_features: list[str] | None = None
    gambling_features: list[str] | None = None
    blue_features: list[str] | None = None
    for study in args.study:
        case_dir = args.case_root / study
        official = gpd.read_parquet(case_dir / "official_context.parquet")
        osm = gpd.read_parquet(case_dir / "osm_context.parquet").drop(columns="geometry")
        frame = official.merge(osm, on="lsoa21cd", how="left", validate="one_to_one")
        official_summary = json.loads(
            (case_dir / "official_context_summary.json").read_text(encoding="utf-8")
        )
        osm_summary = json.loads(
            (case_dir / "osm_context_summary.json").read_text(encoding="utf-8")
        )
        current_official = official_summary["features"]
        current_osm = osm_summary["features"]
        current_green: list[str] = []
        if args.include_os_green:
            green = gpd.read_parquet(case_dir / "os_green_context.parquet").drop(
                columns="geometry"
            )
            frame = frame.merge(green, on="lsoa21cd", how="left", validate="one_to_one")
            green_summary = json.loads(
                (case_dir / "os_green_context_summary.json").read_text(encoding="utf-8")
            )
            current_green = green_summary["features"]
        current_sentinel: list[str] = []
        if args.include_sentinel:
            sentinel = gpd.read_parquet(case_dir / "sentinel_ndvi_context.parquet").drop(
                columns="geometry"
            )
            frame = frame.merge(
                sentinel, on="lsoa21cd", how="left", validate="one_to_one"
            )
            sentinel_summary = json.loads(
                (case_dir / "sentinel_ndvi_context_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            current_sentinel = sentinel_summary["features"]
        current_gambling: list[str] = []
        if args.include_official_gambling:
            gambling = gpd.read_parquet(
                case_dir / "official_gambling_context.parquet"
            ).drop(columns="geometry")
            frame = frame.merge(
                gambling, on="lsoa21cd", how="left", validate="one_to_one"
            )
            gambling_summary = json.loads(
                (case_dir / "official_gambling_context_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            current_gambling = gambling_summary["features"]
        current_blue: list[str] = []
        if args.include_osm_blue:
            blue = gpd.read_parquet(case_dir / "osm_blue_context.parquet").drop(
                columns="geometry"
            )
            frame = frame.merge(blue, on="lsoa21cd", how="left", validate="one_to_one")
            blue_summary = json.loads(
                (case_dir / "osm_blue_context_summary.json").read_text(encoding="utf-8")
            )
            current_blue = blue_summary["features"]
        if official_features is None:
            official_features = current_official
            osm_features = current_osm
            green_features = current_green
            sentinel_features = current_sentinel
            gambling_features = current_gambling
            blue_features = current_blue
        elif (
            official_features != current_official
            or osm_features != current_osm
            or green_features != current_green
            or sentinel_features != current_sentinel
            or gambling_features != current_gambling
            or blue_features != current_blue
        ):
            raise ValueError("case studies do not share an identical feature schema")
        feature_names = (
            current_official
            + current_osm
            + current_green
            + current_sentinel
            + current_gambling
            + current_blue
        )
        fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
        bundles[study] = {
            "x": frame[feature_names].to_numpy(dtype=float),
            "y": frame[args.target].to_numpy(dtype=float),
            "fold": frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int),
        }

    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    result = {
        "settings": {
            "studies": args.study,
            "models": list(model_names),
            "seed": args.seed,
            "target": args.target,
            "official_features": official_features,
            "osm_features": osm_features,
            "green_features": green_features,
            "sentinel_features": sentinel_features,
            "official_gambling_features": gambling_features,
            "osm_blue_features": blue_features,
            "target_derived_features": False,
        },
        "within_city": {},
        "cross_city": {},
    }
    for study, bundle in bundles.items():
        study_result = {}
        for model_name in model_names:
            rows = []
            for fold in sorted(np.unique(bundle["fold"])):
                train = bundle["fold"] != fold
                test = ~train
                fitted = build_estimator(model_name, args.seed + int(fold)).fit(
                    bundle["x"][train], bundle["y"][train]
                )
                rows.append(
                    {
                        "fold": int(fold),
                        **metrics(bundle["y"][test], fitted.predict(bundle["x"][test])),
                    }
                )
            study_result[model_name] = {"folds": rows, "aggregate": aggregate(rows)}
        result["within_city"][study] = study_result

    for source_name, source in bundles.items():
        for target_name, target in bundles.items():
            if source_name == target_name:
                continue
            direction = {}
            for model_name in model_names:
                fitted = build_estimator(model_name, args.seed).fit(source["x"], source["y"])
                direction[model_name] = metrics(target["y"], fitted.predict(target["x"]))
            result["cross_city"][f"{source_name}_to_{target_name}"] = direction

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
