from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gravity_gnn.case_studies import sha256


BASELINES = ("mean", "ridge", "random_forest", "local_hgb")
PROPOSED = "proposed_pooled_hgb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen simple baseline suite and save strict OOF residuals"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--target", default="ahah")
    parser.add_argument(
        "--output", type=Path, default=Path("results/final_baselines.json")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    return parser


def build_model(name: str, seed: int):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 6, 40)),
        )
    elif name == "random_forest":
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_features=0.7,
                min_samples_leaf=8,
                n_jobs=-1,
                random_state=seed,
            ),
        )
    elif name in {"local_hgb", PROPOSED}:
        regressor = make_pipeline(
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
    else:
        raise ValueError(name)
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def macro_metrics(rows: list[dict]) -> dict:
    output = {}
    for name in ("r2", "rmse", "mae"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return output


def load_case(case_root: Path, study: str, target: str) -> dict:
    case_dir = case_root / study
    sources = [
        ("official_context.parquet", "official_context_summary.json"),
        ("osm_context.parquet", "osm_context_summary.json"),
        ("os_green_context.parquet", "os_green_context_summary.json"),
        ("sentinel_ndvi_context.parquet", "sentinel_ndvi_context_summary.json"),
    ]
    frame = gpd.read_parquet(case_dir / sources[0][0])
    feature_names = []
    for position, (parquet_name, summary_name) in enumerate(sources):
        if position:
            extra = gpd.read_parquet(case_dir / parquet_name).drop(columns="geometry")
            frame = frame.merge(extra, on="lsoa21cd", validate="one_to_one")
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        feature_names.extend(summary["features"])
    fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
    fold = frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int)
    return {
        "frame": frame,
        "x": frame[feature_names].to_numpy(dtype=float),
        "y": frame[target].to_numpy(dtype=float),
        "fold": fold,
        "feature_names": feature_names,
    }


def pooled_training_data(
    bundles: dict[str, dict],
    studies: list[str],
    target_study: str,
    held_out_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_blocks = []
    y_blocks = []
    for city_index, study in enumerate(studies):
        bundle = bundles[study]
        selected = (
            bundle["fold"] != held_out_fold
            if study == target_study
            else np.ones(len(bundle["y"]), dtype=bool)
        )
        city_one_hot = np.zeros((int(selected.sum()), len(studies)), dtype=float)
        city_one_hot[:, city_index] = 1.0
        x_blocks.append(np.column_stack((bundle["x"][selected], city_one_hot)))
        y_blocks.append(bundle["y"][selected])
    return np.concatenate(x_blocks), np.concatenate(y_blocks)


def main() -> None:
    args = build_parser().parse_args()
    if len(args.study) < 2:
        raise ValueError("the proposed pooled model requires at least two studies")
    bundles = {
        study: load_case(args.case_root, study, args.target) for study in args.study
    }
    feature_names = bundles[args.study[0]]["feature_names"]
    if any(bundle["feature_names"] != feature_names for bundle in bundles.values()):
        raise ValueError("case-study feature schemas differ")

    result = {
        "settings": {
            "studies": args.study,
            "target": args.target,
            "baseline_models": list(BASELINES),
            "proposed_model": PROPOSED,
            "features": feature_names,
            "target_derived_features": False,
            "validation": "five frozen contiguous spatial folds; macro fold mean is primary",
            "fairness": (
                "all models use identical target-independent features and target-city folds; "
                "only the proposed model additionally uses all auxiliary-city observations"
            ),
        },
        "within_city": {},
        "oof_outputs": {},
    }
    args.oof_dir.mkdir(parents=True, exist_ok=True)

    for target_index, study in enumerate(args.study):
        bundle = bundles[study]
        predictions = {
            name: np.full(len(bundle["y"]), np.nan, dtype=float)
            for name in (*BASELINES, PROPOSED)
        }
        model_results = {}
        for model_name in (*BASELINES, PROPOSED):
            folds = []
            for fold in sorted(np.unique(bundle["fold"])):
                train = bundle["fold"] != fold
                test = ~train
                if model_name == PROPOSED:
                    train_x, train_y = pooled_training_data(
                        bundles, args.study, study, int(fold)
                    )
                    test_city = np.zeros((int(test.sum()), len(args.study)), dtype=float)
                    test_city[:, target_index] = 1.0
                    test_x = np.column_stack((bundle["x"][test], test_city))
                else:
                    train_x, train_y = bundle["x"][train], bundle["y"][train]
                    test_x = bundle["x"][test]
                fitted = build_model(
                    model_name, args.seed + int(fold) * 100 + target_index
                ).fit(train_x, train_y)
                prediction = fitted.predict(test_x)
                predictions[model_name][test] = prediction
                folds.append(
                    {
                        "fold": int(fold),
                        **metrics(bundle["y"][test], prediction),
                    }
                )
            if not np.isfinite(predictions[model_name]).all():
                raise RuntimeError(f"incomplete OOF predictions for {study}/{model_name}")
            model_results[model_name] = {
                "folds": folds,
                "macro": macro_metrics(folds),
                "pooled_oof": metrics(bundle["y"], predictions[model_name]),
            }
        result["within_city"][study] = model_results

        oof = bundle["frame"][["lsoa21cd", "geometry"]].copy()
        oof["fold"] = bundle["fold"]
        oof["observed"] = bundle["y"]
        for model_name, prediction in predictions.items():
            oof[f"prediction_{model_name}"] = prediction
            oof[f"residual_{model_name}"] = bundle["y"] - prediction
        oof_path = args.oof_dir / f"{study}.parquet"
        oof.to_parquet(oof_path, index=False)
        result["oof_outputs"][study] = {
            "path": str(oof_path.resolve()),
            "sha256": sha256(oof_path),
            "rows": len(oof),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    concise = {
        study: {
            name: {
                "r2": values["macro"]["r2"]["mean"],
                "rmse": values["macro"]["rmse"]["mean"],
            }
            for name, values in models.items()
        }
        for study, models in result["within_city"].items()
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
