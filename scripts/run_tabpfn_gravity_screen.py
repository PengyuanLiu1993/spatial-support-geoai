from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


MODEL_VERSIONS = {
    "v2": ModelVersion.V2,
    "v2.5": ModelVersion.V2_5,
    "v2.6": ModelVersion.V2_6,
    "v3": ModelVersion.V3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen spatial-CV screen of TabPFN with gravity features"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("results/tabpfn_gravity/input_bundle.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/tabpfn_gravity/screen_v2.json")
    )
    parser.add_argument(
        "--hgb-reference",
        type=Path,
        default=Path("results/distance_effect/distance_effect_results.json"),
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=("context_only", "gravity_scale_15"),
        required=True,
    )
    parser.add_argument("--model-version", choices=tuple(MODEL_VERSIONS), default="v2")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-fold", type=int, action="append")
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
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return output


def load_bundle(path: Path) -> tuple[list[str], dict[str, dict]]:
    payload = np.load(path, allow_pickle=False)
    studies = payload["studies"].astype(str).tolist()
    bundles = {}
    for index, study in enumerate(studies):
        prefix = f"city_{index}"
        bundles[study] = {
            "base_x": payload[f"{prefix}_base_x"].astype(np.float32),
            "gravity_x": payload[f"{prefix}_gravity_x"].astype(np.float32),
            "y": payload[f"{prefix}_y"].astype(np.float32),
            "fold": payload[f"{prefix}_fold"].astype(int),
            "lsoa": payload[f"{prefix}_lsoa"].astype(str),
        }
    return studies, bundles


def city_feature(rows: int, city_index: int) -> np.ndarray:
    return np.full((rows, 1), city_index, dtype=np.float32)


def feature_block(bundle: dict, variant: str) -> np.ndarray:
    if variant == "context_only":
        return bundle["base_x"]
    if variant == "gravity_scale_15":
        return np.column_stack((bundle["base_x"], bundle["gravity_x"]))
    raise ValueError(f"unknown variant: {variant}")


def pooled_train(
    bundles: dict[str, dict],
    studies: list[str],
    target_study: str,
    target_train: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_blocks = []
    y_blocks = []
    for city_index, study in enumerate(studies):
        bundle = bundles[study]
        selector = (
            target_train
            if study == target_study
            else np.ones(len(bundle["y"]), dtype=bool)
        )
        values = feature_block(bundle, variant)[selector]
        x_blocks.append(np.column_stack((values, city_feature(len(values), city_index))))
        y_blocks.append(bundle["y"][selector])
    train_x = np.concatenate(x_blocks).astype(np.float32, copy=False)
    train_y = np.concatenate(y_blocks).astype(np.float32, copy=False)
    if not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
        raise ValueError("non-finite TabPFN training input")
    return train_x, train_y


def make_model(args: argparse.Namespace, seed: int) -> TabPFNRegressor:
    return TabPFNRegressor.create_default_for_version(
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


def load_hgb_reference(path: Path, study: str, variant: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["studies"][study]["variants"][variant]["macro"]


def main() -> None:
    args = build_parser().parse_args()
    if args.n_estimators < 1:
        raise ValueError("n-estimators must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    studies, bundles = load_bundle(args.input)
    if len(studies) != 2:
        raise ValueError("the pooled screen requires exactly two cities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    oof_dir = args.output.parent / f"{args.output.stem}_oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "settings": {
            "studies": studies,
            "variants": args.variant,
            "model": "TabPFNRegressor",
            "model_version": args.model_version,
            "n_estimators": args.n_estimators,
            "device": args.device,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "validation": "five frozen contiguous spatial folds",
            "pooled_design": (
                "target-city outer training folds plus all observations from the "
                "other city; held-out target-city fold is used only for testing"
            ),
            "city_indicator": "one numeric 0/1 column; no scaling or one-hot encoding",
            "target_derived_features": False,
            "absolute_coordinate_features": False,
            "seed": args.seed,
            "screening_note": (
                "low-cost screen; n_estimators is fixed and automatic estimator "
                "scaling is disabled"
            ),
        },
        "studies": {},
    }

    for study_index, study in enumerate(studies):
        bundle = bundles[study]
        all_folds = sorted(int(value) for value in np.unique(bundle["fold"]))
        selected_folds = all_folds if args.test_fold is None else args.test_fold
        if any(fold not in all_folds for fold in selected_folds):
            raise ValueError(f"invalid test fold for {study}: {selected_folds}")
        study_result = {"variants": {}}
        for variant_index, variant in enumerate(args.variant):
            prediction = np.full(len(bundle["y"]), np.nan, dtype=np.float32)
            fold_rows = []
            for test_fold in selected_folds:
                train_selector = bundle["fold"] != test_fold
                test_selector = bundle["fold"] == test_fold
                train_x, train_y = pooled_train(
                    bundles, studies, study, train_selector, variant
                )
                test_values = feature_block(bundle, variant)[test_selector]
                test_x = np.column_stack(
                    (test_values, city_feature(len(test_values), study_index))
                ).astype(np.float32, copy=False)
                seed = args.seed + study_index * 1000 + variant_index * 100 + test_fold
                started = time.perf_counter()
                model = make_model(args, seed)
                model.fit(train_x, train_y)
                fold_prediction = np.asarray(model.predict(test_x), dtype=np.float32)
                elapsed = time.perf_counter() - started
                prediction[test_selector] = fold_prediction
                row = {
                    "fold": test_fold,
                    "train_rows": int(len(train_y)),
                    "test_rows": int(test_selector.sum()),
                    "elapsed_seconds": float(elapsed),
                    **metrics(bundle["y"][test_selector], fold_prediction),
                }
                fold_rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "fold_complete",
                            "study": study,
                            "variant": variant,
                            **row,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            evaluated = np.isfinite(prediction)
            if int(evaluated.sum()) != int(
                np.isin(bundle["fold"], selected_folds).sum()
            ):
                raise RuntimeError(f"incomplete predictions: {study}/{variant}")
            macro = aggregate(fold_rows)
            pooled = metrics(bundle["y"][evaluated], prediction[evaluated])
            hgb = load_hgb_reference(args.hgb_reference, study, variant)
            comparison = None
            if hgb is not None and len(selected_folds) == len(all_folds):
                comparison = {
                    "delta_r2_tabpfn_minus_hgb": float(
                        macro["r2"]["mean"] - hgb["r2"]["mean"]
                    ),
                    "delta_rmse_tabpfn_minus_hgb": float(
                        macro["rmse"]["mean"] - hgb["rmse"]["mean"]
                    ),
                }
            study_result["variants"][variant] = {
                "folds": fold_rows,
                "macro": macro,
                "pooled_oof": pooled,
                "hgb_macro_reference": hgb,
                "comparison_to_hgb": comparison,
            }
            frame = pd.DataFrame(
                {
                    "lsoa21cd": bundle["lsoa"],
                    "fold": bundle["fold"],
                    "observed": bundle["y"],
                    "prediction": prediction,
                    "residual": bundle["y"] - prediction,
                    "evaluated": evaluated,
                }
            )
            frame.to_csv(oof_dir / f"{study}_{variant}.csv", index=False)
        result["studies"][study] = study_result

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
