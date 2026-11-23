from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DOMAINS = ("domain_h", "domain_g", "domain_e", "domain_r")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate domain-decomposed AHAH prediction without test-label inputs"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--models", default="ridge,hist_gradient_boosting,extra_trees"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_two_case_domain_decomposed.json"),
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
    elif name == "extra_trees":
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=350,
                max_features=0.8,
                min_samples_leaf=4,
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


def feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    def matching(*tokens: str) -> list[int]:
        indices = [
            index
            for index, name in enumerate(feature_names)
            if any(token in name for token in tokens)
        ]
        return sorted(set(indices))

    common = matching("lsoa_area", "compactness", "building")
    groups = {
        "domain_h": common
        + matching(
            "nhs_",
            "primary_care",
            "dentist",
            "pharmacy",
            "hospital",
            "leisure",
            "education",
        ),
        "domain_g": common
        + matching("s2_ndvi", "os_green", "active_green", "green_blue"),
        "domain_e": common + matching("defra_"),
        "domain_r": common
        + matching(
            "fast_food",
            "gambling",
            "pub_bar",
            "tobacco",
            "food_retail",
            "social_food",
        ),
    }
    return {name: sorted(set(indices)) for name, indices in groups.items()}


def load_case(case_root: Path, study: str) -> dict:
    case_dir = case_root / study
    paths = [
        ("official_context.parquet", "official_context_summary.json"),
        ("osm_context.parquet", "osm_context_summary.json"),
        ("os_green_context.parquet", "os_green_context_summary.json"),
        ("sentinel_ndvi_context.parquet", "sentinel_ndvi_context_summary.json"),
    ]
    frame = gpd.read_parquet(case_dir / paths[0][0])
    feature_names = []
    for parquet_name, summary_name in paths:
        if parquet_name != paths[0][0]:
            extra = gpd.read_parquet(case_dir / parquet_name).drop(columns="geometry")
            frame = frame.merge(extra, on="lsoa21cd", validate="one_to_one")
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        feature_names.extend(summary["features"])
    fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
    return {
        "x": frame[feature_names].to_numpy(dtype=float),
        "ahah": frame["ahah"].to_numpy(dtype=float),
        "domains": frame[list(DOMAINS)].to_numpy(dtype=float),
        "fold": frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int),
        "feature_names": feature_names,
    }


def main() -> None:
    args = build_parser().parse_args()
    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    bundles = {study: load_case(args.case_root, study) for study in args.study}
    feature_names = bundles[args.study[0]]["feature_names"]
    if any(bundle["feature_names"] != feature_names for bundle in bundles.values()):
        raise ValueError("case-study feature schemas differ")
    scoped = feature_groups(feature_names)
    all_features = {domain: list(range(len(feature_names))) for domain in DOMAINS}

    result = {
        "settings": {
            "studies": args.study,
            "models": list(model_names),
            "features": feature_names,
            "domain_feature_indices": scoped,
            "auxiliary_training_labels": list(DOMAINS),
            "test_label_inputs": False,
            "target_derived_features": False,
            "composite_calibration": (
                "training-fold isotonic mapping from mean predicted domain score to AHAH"
            ),
        },
        "within_city": {},
    }

    for study, bundle in bundles.items():
        study_result = {}
        for design_name, groups in {"scoped": scoped, "all_features": all_features}.items():
            design_result = {}
            for model_name in model_names:
                rows = []
                for fold in sorted(np.unique(bundle["fold"])):
                    train = bundle["fold"] != fold
                    test = ~train
                    train_domain_predictions = []
                    test_domain_predictions = []
                    domain_metrics = {}
                    for domain_index, domain in enumerate(DOMAINS):
                        indices = groups[domain]
                        fitted = build_estimator(
                            model_name, args.seed + int(fold) * 10 + domain_index
                        ).fit(
                            bundle["x"][train][:, indices],
                            bundle["domains"][train, domain_index],
                        )
                        train_prediction = fitted.predict(
                            bundle["x"][train][:, indices]
                        )
                        test_prediction = fitted.predict(bundle["x"][test][:, indices])
                        train_domain_predictions.append(train_prediction)
                        test_domain_predictions.append(test_prediction)
                        domain_metrics[domain] = metrics(
                            bundle["domains"][test, domain_index], test_prediction
                        )

                    train_mean = np.mean(train_domain_predictions, axis=0)
                    test_mean = np.mean(test_domain_predictions, axis=0)
                    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                        train_mean, bundle["ahah"][train]
                    )
                    prediction = calibrator.predict(test_mean)
                    rows.append(
                        {
                            "fold": int(fold),
                            **metrics(bundle["ahah"][test], prediction),
                            "domain_metrics": domain_metrics,
                        }
                    )
                design_result[model_name] = {
                    "folds": rows,
                    "aggregate": aggregate(rows),
                }
            study_result[design_name] = design_result
        result["within_city"][study] = study_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
