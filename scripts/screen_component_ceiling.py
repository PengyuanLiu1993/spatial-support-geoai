from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ORACLE_GROUPS = {
    "green": ["greenspace", "greenspace_active", "bluespace"],
    "retail": ["fast_food", "gambling", "pub_bar", "tobacco"],
    "health": ["GP", "dentist", "pharmacy", "hospital", "leisure"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantify missing-component data ceilings")
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_two_case_component_ceiling.json")
    )
    return parser


def estimator(seed: int):
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=350,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=3.0,
                random_state=seed,
            ),
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
    result = {}
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return result


def load_case(case_root: Path, study: str) -> tuple[gpd.GeoDataFrame, list[str], np.ndarray]:
    case_dir = case_root / study
    official = gpd.read_parquet(case_dir / "official_context.parquet")
    osm = gpd.read_parquet(case_dir / "osm_context.parquet").drop(columns="geometry")
    green = gpd.read_parquet(case_dir / "os_green_context.parquet").drop(columns="geometry")
    frame = official.merge(osm, on="lsoa21cd", validate="one_to_one").merge(
        green, on="lsoa21cd", validate="one_to_one"
    )
    independent_features = []
    for summary_name in (
        "official_context_summary.json",
        "osm_context_summary.json",
        "os_green_context_summary.json",
    ):
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        independent_features.extend(summary["features"])
    fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
    fold = frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=int)
    return frame, independent_features, fold


def main() -> None:
    args = build_parser().parse_args()
    scenarios = {
        "independent_only": [],
        "plus_green_oracle": ORACLE_GROUPS["green"],
        "plus_retail_oracle": ORACLE_GROUPS["retail"],
        "plus_health_oracle": ORACLE_GROUPS["health"],
        "plus_green_retail_oracle": ORACLE_GROUPS["green"] + ORACLE_GROUPS["retail"],
        "plus_all_missing_domain_oracles": (
            ORACLE_GROUPS["green"] + ORACLE_GROUPS["retail"] + ORACLE_GROUPS["health"]
        ),
    }
    result = {
        "settings": {
            "studies": args.study,
            "target": "ahah",
            "scenarios": scenarios,
            "warning": (
                "Any scenario containing oracle features is a data ceiling only. The raw component "
                "values come from the same AHAH release and are not valid predictive inputs."
            ),
        },
        "within_city": {},
    }
    for study in args.study:
        frame, independent_features, fold = load_case(args.case_root, study)
        values = frame["ahah"].to_numpy(dtype=float)
        study_result = {}
        for scenario, oracle_features in scenarios.items():
            feature_names = independent_features + oracle_features
            matrix = frame[feature_names].to_numpy(dtype=float)
            rows = []
            for test_fold in sorted(np.unique(fold)):
                train = fold != test_fold
                test = ~train
                fitted = estimator(args.seed + int(test_fold)).fit(matrix[train], values[train])
                rows.append(
                    {
                        "fold": int(test_fold),
                        **metrics(values[test], fitted.predict(matrix[test])),
                    }
                )
            study_result[scenario] = {
                "features": feature_names,
                "folds": rows,
                "aggregate": aggregate(rows),
            }
        result["within_city"][study] = study_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
