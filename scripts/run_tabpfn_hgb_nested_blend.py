from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


MODEL_VERSIONS = {
    "v2": ModelVersion.V2,
    "v2.5": ModelVersion.V2_5,
    "v2.6": ModelVersion.V2_6,
    "v3": ModelVersion.V3,
}
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested spatial selection of a gravity HGB-TabPFN blend"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("results/tabpfn_gravity/input_bundle.npz")
    )
    parser.add_argument(
        "--tabpfn-oof-dir",
        type=Path,
        default=Path("results/tabpfn_gravity/screen_v2_e1_oof"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tabpfn_gravity/nested_blend_v2_e1.json"),
    )
    parser.add_argument("--model-version", choices=tuple(MODEL_VERSIONS), default="v2")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    return parser


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        output[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return output


def load_bundle(path: Path) -> tuple[list[str], dict[str, dict]]:
    payload = np.load(path, allow_pickle=False)
    studies = payload["studies"].astype(str).tolist()
    bundles = {}
    for index, study in enumerate(studies):
        prefix = f"city_{index}"
        bundles[study] = {
            "x": np.column_stack(
                (
                    payload[f"{prefix}_base_x"].astype(np.float32),
                    payload[f"{prefix}_gravity_x"].astype(np.float32),
                )
            ),
            "y": payload[f"{prefix}_y"].astype(np.float32),
            "fold": payload[f"{prefix}_fold"].astype(int),
            "lsoa": payload[f"{prefix}_lsoa"].astype(str),
            "hgb_outer": payload[f"{prefix}_hgb_gravity15_prediction"].astype(
                np.float32
            ),
        }
    return studies, bundles


def pooled_train(
    bundles: dict[str, dict],
    studies: list[str],
    target_study: str,
    target_selector: np.ndarray,
    tabpfn_design: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_blocks = []
    y_blocks = []
    for city_index, study in enumerate(studies):
        bundle = bundles[study]
        selector = (
            target_selector
            if study == target_study
            else np.ones(len(bundle["y"]), dtype=bool)
        )
        values = bundle["x"][selector]
        if tabpfn_design:
            city = np.full((len(values), 1), city_index, dtype=np.float32)
        else:
            city = np.zeros((len(values), len(studies)), dtype=np.float32)
            city[:, city_index] = 1.0
        x_blocks.append(np.column_stack((values, city)))
        y_blocks.append(bundle["y"][selector])
    return (
        np.concatenate(x_blocks).astype(np.float32, copy=False),
        np.concatenate(y_blocks).astype(np.float32, copy=False),
    )


def target_x(bundle: dict, selector: np.ndarray, city_index: int, cities: int, tabpfn_design: bool) -> np.ndarray:
    values = bundle["x"][selector]
    if tabpfn_design:
        city = np.full((len(values), 1), city_index, dtype=np.float32)
    else:
        city = np.zeros((len(values), cities), dtype=np.float32)
        city[:, city_index] = 1.0
    return np.column_stack((values, city)).astype(np.float32, copy=False)


def fit_hgb(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=seed,
    ).fit(x, y)


def fit_tabpfn(args: argparse.Namespace, x: np.ndarray, y: np.ndarray, seed: int) -> TabPFNRegressor:
    model = TabPFNRegressor.create_default_for_version(
        MODEL_VERSIONS[args.model_version],
        n_estimators=args.n_estimators,
        auto_scale_n_estimators=False,
        categorical_features_indices=None,
        device=args.device,
        fit_mode="low_memory",
        memory_saving_mode="auto",
        keep_cache_on_device=False,
        random_state=seed,
        show_progress_bar=False,
    )
    return model.fit(x, y)


def load_tabpfn_outer(path: Path, bundle: dict) -> np.ndarray:
    frame = pd.read_csv(path)
    if not np.array_equal(frame["lsoa21cd"].astype(str).to_numpy(), bundle["lsoa"]):
        raise ValueError(f"TabPFN OOF row mismatch: {path}")
    prediction = frame["prediction"].to_numpy(dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError(f"TabPFN OOF is incomplete: {path}")
    return prediction


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    studies, bundles = load_bundle(args.input)
    if len(studies) != 2:
        raise ValueError("the pooled nested blend requires exactly two cities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "settings": {
            "studies": studies,
            "feature_set": "103 context plus 24 continuous gravity scale-15 features",
            "outer_validation": "five frozen contiguous spatial folds",
            "inner_selection": (
                "the next target-city spatial fold selects the HGB/TabPFN blend weight; "
                "the outer test fold is excluded from both inner fitting and selection"
            ),
            "blend_definition": "(1-weight)*HGB + weight*TabPFN",
            "blend_weights": BLEND_WEIGHTS,
            "tabpfn_version": args.model_version,
            "tabpfn_n_estimators": args.n_estimators,
            "target_derived_features": False,
            "absolute_coordinate_features": False,
            "seed": args.seed,
        },
        "studies": {},
    }
    for city_index, study in enumerate(studies):
        bundle = bundles[study]
        tabpfn_outer = load_tabpfn_outer(
            args.tabpfn_oof_dir / f"{study}_gravity_scale_15.csv", bundle
        )
        blended = np.full(len(bundle["y"]), np.nan, dtype=np.float32)
        fold_rows = []
        folds = sorted(int(value) for value in np.unique(bundle["fold"]))
        for position, test_fold in enumerate(folds):
            validation_fold = folds[(position + 1) % len(folds)]
            inner_train = (bundle["fold"] != test_fold) & (
                bundle["fold"] != validation_fold
            )
            validation = bundle["fold"] == validation_fold
            test = bundle["fold"] == test_fold

            hgb_train_x, hgb_train_y = pooled_train(
                bundles, studies, study, inner_train, tabpfn_design=False
            )
            tabpfn_train_x, tabpfn_train_y = pooled_train(
                bundles, studies, study, inner_train, tabpfn_design=True
            )
            hgb_validation_x = target_x(
                bundle, validation, city_index, len(studies), tabpfn_design=False
            )
            tabpfn_validation_x = target_x(
                bundle, validation, city_index, len(studies), tabpfn_design=True
            )
            seed = args.seed + city_index * 1000 + test_fold
            started = time.perf_counter()
            hgb = fit_hgb(hgb_train_x, hgb_train_y, seed)
            hgb_validation = hgb.predict(hgb_validation_x)
            tabpfn = fit_tabpfn(args, tabpfn_train_x, tabpfn_train_y, seed)
            tabpfn_validation = np.asarray(
                tabpfn.predict(tabpfn_validation_x), dtype=np.float32
            )
            validation_y = bundle["y"][validation]
            candidates = []
            for weight in BLEND_WEIGHTS:
                candidate = (1.0 - weight) * hgb_validation + weight * tabpfn_validation
                candidates.append(
                    {
                        "weight": weight,
                        "rmse": float(mean_squared_error(validation_y, candidate) ** 0.5),
                    }
                )
            selected = min(candidates, key=lambda row: (row["rmse"], row["weight"]))
            weight = float(selected["weight"])
            outer_prediction = (
                (1.0 - weight) * bundle["hgb_outer"][test]
                + weight * tabpfn_outer[test]
            )
            blended[test] = outer_prediction
            elapsed = time.perf_counter() - started
            row = {
                "test_fold": test_fold,
                "validation_fold": validation_fold,
                "selected_tabpfn_weight": weight,
                "validation_candidates": candidates,
                "elapsed_seconds": float(elapsed),
                **metrics(bundle["y"][test], outer_prediction),
            }
            fold_rows.append(row)
            print(
                json.dumps(
                    {"event": "nested_fold_complete", "study": study, **row},
                    sort_keys=True,
                ),
                flush=True,
            )
            del tabpfn
            torch.cuda.empty_cache()

        if not np.isfinite(blended).all():
            raise RuntimeError(f"incomplete nested blend predictions: {study}")
        study_result = {
            "folds": fold_rows,
            "macro": aggregate(fold_rows),
            "pooled_oof": metrics(bundle["y"], blended),
            "selected_weight_counts": {
                str(weight): int(
                    sum(row["selected_tabpfn_weight"] == weight for row in fold_rows)
                )
                for weight in BLEND_WEIGHTS
            },
        }
        result["studies"][study] = study_result
        pd.DataFrame(
            {
                "lsoa21cd": bundle["lsoa"],
                "fold": bundle["fold"],
                "observed": bundle["y"],
                "prediction_hgb": bundle["hgb_outer"],
                "prediction_tabpfn": tabpfn_outer,
                "prediction_nested_blend": blended,
                "residual_nested_blend": bundle["y"] - blended,
            }
        ).to_csv(args.output.parent / f"nested_blend_{study}.csv", index=False)

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
