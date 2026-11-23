from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRAVITY_SCALES = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
GRAVITY_BETA = 1.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen spatial-CV comparison of facility distance encodings"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/distance_effect")
    )
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


def binned_sum(
    destination: np.ndarray, values: np.ndarray, n_lsoa: int
) -> np.ndarray:
    return np.bincount(destination, weights=values, minlength=n_lsoa).astype(float)


def graph_distance_features(graph_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    payload = np.load(graph_path, allow_pickle=False)
    relation_names = payload["relation_names"].astype(str).tolist()
    facility_relations = relation_names[:-1]
    n_relations = len(facility_relations)
    n_lsoa = len(payload["lsoa_codes"])
    source, destination = payload["edge_index"]
    edge_type = payload["edge_type"]
    impedance = payload["edge_impedance"].astype(float)
    facility_edge = (edge_type < n_relations) & (destination < n_lsoa)
    source = source[facility_edge]
    destination = destination[facility_edge]
    edge_type = edge_type[facility_edge]
    impedance = impedance[facility_edge]
    mass = payload["node_mass"][source].astype(float)

    variants: dict[str, np.ndarray] = {
        "context_only": np.empty((n_lsoa, 0), dtype=float),
        "no_distance_40": np.zeros((n_lsoa, 2 * n_relations), dtype=float),
        "threshold_15": np.zeros((n_lsoa, 2 * n_relations), dtype=float),
        "threshold_20": np.zeros((n_lsoa, 2 * n_relations), dtype=float),
        "segmented_15_20": np.zeros((n_lsoa, 6 * n_relations), dtype=float),
    }
    for scale in GRAVITY_SCALES:
        variants[f"gravity_scale_{int(scale)}"] = np.zeros(
            (n_lsoa, 2 * n_relations), dtype=float
        )

    for relation in range(n_relations):
        selected = edge_type == relation
        destinations = destination[selected]
        times = impedance[selected]
        masses = mass[selected]
        variants["no_distance_40"][:, 2 * relation] = binned_sum(
            destinations, np.ones(len(destinations)), n_lsoa
        )
        variants["no_distance_40"][:, 2 * relation + 1] = binned_sum(
            destinations, masses, n_lsoa
        )
        for threshold in (15, 20):
            within = times <= threshold
            name = f"threshold_{threshold}"
            variants[name][:, 2 * relation] = binned_sum(
                destinations[within], np.ones(int(within.sum())), n_lsoa
            )
            variants[name][:, 2 * relation + 1] = binned_sum(
                destinations[within], masses[within], n_lsoa
            )
        segment_masks = (times <= 15, (times > 15) & (times <= 20), times > 20)
        for segment, within in enumerate(segment_masks):
            offset = relation * 6 + segment * 2
            variants["segmented_15_20"][:, offset] = binned_sum(
                destinations[within], np.ones(int(within.sum())), n_lsoa
            )
            variants["segmented_15_20"][:, offset + 1] = binned_sum(
                destinations[within], masses[within], n_lsoa
            )
        for scale in GRAVITY_SCALES:
            weight = np.exp(-np.power(times / scale, GRAVITY_BETA))
            name = f"gravity_scale_{int(scale)}"
            variants[name][:, 2 * relation] = binned_sum(
                destinations, weight, n_lsoa
            )
            variants[name][:, 2 * relation + 1] = binned_sum(
                destinations, weight * masses, n_lsoa
            )

    for name, values in variants.items():
        variants[name] = np.log1p(values)
        if not np.isfinite(variants[name]).all():
            raise ValueError(f"non-finite distance features in {graph_path}: {name}")
    base = {
        "x": payload["x"][:n_lsoa].astype(float),
        "feature_names": payload["feature_names"].astype(str).tolist(),
        "relation_names": facility_relations,
        "lsoa_codes": payload["lsoa_codes"].astype(str),
        "edge_minutes": {
            "minimum": float(impedance.min()),
            "maximum": float(impedance.max()),
        },
    }
    return base, variants


def load_case(case_root: Path, oof_dir: Path, study: str) -> dict:
    base, variants = graph_distance_features(
        case_root / study / "facility_lsoa_graph.npz"
    )
    oof = gpd.read_parquet(oof_dir / f"{study}.parquet")
    if not np.array_equal(oof["lsoa21cd"].astype(str).to_numpy(), base["lsoa_codes"]):
        raise ValueError(f"OOF/graph row order mismatch for {study}")
    return {
        "frame": oof,
        "base_x": base["x"],
        "y": oof["observed"].to_numpy(dtype=float),
        "fold": oof["fold"].to_numpy(dtype=int),
        "variants": variants,
        "base_metadata": base,
    }


def design_matrix(
    bundles: dict[str, dict],
    studies: list[str],
    target_study: str,
    variant: str,
    target_selector: np.ndarray,
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
        city = np.zeros((int(selector.sum()), len(studies)), dtype=float)
        city[:, city_index] = 1.0
        x_blocks.append(
            np.column_stack(
                (
                    bundle["base_x"][selector],
                    bundle["variants"][variant][selector],
                    city,
                )
            )
        )
        y_blocks.append(bundle["y"][selector])
    return np.concatenate(x_blocks), np.concatenate(y_blocks)


def fit_model(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=seed,
    ).fit(x, y)


def block_bootstrap(
    delta_by_fold: np.ndarray, replicates: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        np.asarray(delta_by_fold, dtype=float),
        size=(replicates, len(delta_by_fold)),
        replace=True,
    ).mean(axis=1)
    probability_nonpositive = float(np.mean(sampled <= 0))
    probability_nonnegative = float(np.mean(sampled >= 0))
    return {
        "mean_delta_rmse": float(np.mean(delta_by_fold)),
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
        "two_sided_p_bootstrap": float(
            min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        ),
        "interpretation": "negative delta favours the first-named model",
    }


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def add_contrast_fdr(result: dict) -> dict:
    families: dict[str, list[dict]] = {
        "versus_no_distance": [],
        "threshold_20_vs_15": [],
        "adjacent_gravity_scales": [],
    }
    for study in result["studies"].values():
        for name, contrast in study["contrasts"].items():
            if name == "threshold_20_vs_threshold_15":
                families["threshold_20_vs_15"].append(contrast)
            elif name.startswith("scale_"):
                families["adjacent_gravity_scales"].append(contrast)
            elif name.endswith("_vs_no_distance_40"):
                families["versus_no_distance"].append(contrast)
    for family, contrasts in families.items():
        if not contrasts:
            continue
        adjusted = bh_fdr(
            np.asarray([row["two_sided_p_bootstrap"] for row in contrasts])
        )
        for row, q_value in zip(contrasts, adjusted, strict=True):
            row["q_bh_family"] = float(q_value)
            row["fdr_family"] = family
    result["settings"]["contrast_multiple_testing"] = (
        "Benjamini-Hochberg within contrast family across both cities"
    )
    return result


def plot_results(performance: pd.DataFrame, output: Path) -> None:
    studies = performance["study"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, len(studies), figsize=(12, 8), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(studies))
    for city_index, study in enumerate(studies):
        city = performance[performance["study"] == study].copy()
        gravity = city[city["variant"].str.startswith("gravity_scale_")].copy()
        gravity["scale"] = gravity["variant"].str.extract(r"(\d+)$").astype(int)
        axis = axes[0, city_index]
        axis.plot(gravity["scale"], gravity["r2"], marker="o", color="#1b9e77")
        axis.set_title(study.replace("_", " ").title())
        axis.set_xlabel("Continuous gravity scale (min)")
        axis.set_ylabel("Macro spatial-CV R²")
        axis.grid(alpha=0.2)
        categorical = city[
            city["variant"].isin(
                [
                    "context_only",
                    "no_distance_40",
                    "threshold_15",
                    "threshold_20",
                    "segmented_15_20",
                ]
            )
        ]
        axis = axes[1, city_index]
        labels = [value.replace("_", "\n") for value in categorical["variant"]]
        axis.bar(labels, categorical["r2"], color="#7570b3")
        axis.set_ylabel("Macro spatial-CV R²")
        axis.tick_params(axis="x", labelrotation=0)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Facility distance-encoding comparison")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if len(args.study) != 2:
        raise ValueError("this frozen experiment requires exactly two cities")
    bundles = {
        study: load_case(args.case_root, args.oof_dir, study) for study in args.study
    }
    variant_names = list(bundles[args.study[0]]["variants"])
    if any(list(bundles[study]["variants"]) != variant_names for study in args.study):
        raise ValueError("distance variant schemas differ between cities")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oof").mkdir(parents=True, exist_ok=True)
    result = {
        "settings": {
            "studies": args.study,
            "variants": variant_names,
            "gravity_scales_minutes": GRAVITY_SCALES,
            "gravity_beta": GRAVITY_BETA,
            "feature_definition": (
                "two log1p facility channels per relation: weighted count and "
                "weighted source mass; segmented variant has two channels per bin"
            ),
            "segments_minutes": [0, 15, 20, 40],
            "validation": "five frozen contiguous spatial folds",
            "model": "fixed two-city HGB; no outer-test model selection",
            "randomness_control": (
                "all distance variants use the same HGB seed within a city/fold "
                "so the internal early-stopping split is paired"
            ),
            "hgb": {
                "learning_rate": 0.035,
                "max_iter": 400,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 25,
                "l2_regularization": 3.0,
            },
            "bootstrap": (
                "resample the five frozen test folds with replacement; "
                "delta is first model RMSE minus comparator RMSE"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "target_derived_features": False,
            "absolute_coordinate_features": False,
        },
        "studies": {},
    }
    performance_rows: list[dict] = []
    for study_index, study in enumerate(args.study):
        bundle = bundles[study]
        predictions: dict[str, np.ndarray] = {}
        study_result = {"variants": {}, "contrasts": {}}
        unique_folds = sorted(int(value) for value in np.unique(bundle["fold"]))
        for variant_index, variant in enumerate(variant_names):
            prediction = np.full(len(bundle["y"]), np.nan, dtype=float)
            fold_rows = []
            for fold in unique_folds:
                train_selector = bundle["fold"] != fold
                test_selector = bundle["fold"] == fold
                train_x, train_y = design_matrix(
                    bundles, args.study, study, variant, train_selector
                )
                test_city = np.zeros((int(test_selector.sum()), len(args.study)))
                test_city[:, study_index] = 1.0
                test_x = np.column_stack(
                    (
                        bundle["base_x"][test_selector],
                        bundle["variants"][variant][test_selector],
                        test_city,
                    )
                )
                fitted = fit_model(
                    train_x,
                    train_y,
                    args.seed + study_index * 10000 + fold,
                )
                fold_prediction = fitted.predict(test_x)
                prediction[test_selector] = fold_prediction
                fold_rows.append(
                    {
                        "fold": fold,
                        **metrics(bundle["y"][test_selector], fold_prediction),
                    }
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"incomplete predictions: {study}/{variant}")
            predictions[variant] = prediction
            macro = aggregate(fold_rows)
            pooled = metrics(bundle["y"], prediction)
            study_result["variants"][variant] = {
                "folds": fold_rows,
                "macro": macro,
                "pooled_oof": pooled,
            }
            performance_rows.append(
                {
                    "study": study,
                    "variant": variant,
                    "r2": macro["r2"]["mean"],
                    "rmse": macro["rmse"]["mean"],
                    "mae": macro["mae"]["mean"],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "variant_complete",
                        "study": study,
                        "variant": variant,
                        "r2": macro["r2"]["mean"],
                        "rmse": macro["rmse"]["mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        reference = study_result["variants"]["no_distance_40"]
        reference_rmse = np.asarray([row["rmse"] for row in reference["folds"]])
        for contrast_index, variant in enumerate(variant_names):
            if variant == "no_distance_40":
                continue
            variant_rmse = np.asarray(
                [row["rmse"] for row in study_result["variants"][variant]["folds"]]
            )
            study_result["contrasts"][f"{variant}_vs_no_distance_40"] = block_bootstrap(
                variant_rmse - reference_rmse,
                args.bootstrap_replicates,
                args.seed + study_index * 100 + contrast_index,
            )
        threshold_15_rmse = np.asarray(
            [
                row["rmse"]
                for row in study_result["variants"]["threshold_15"]["folds"]
            ]
        )
        threshold_20_rmse = np.asarray(
            [
                row["rmse"]
                for row in study_result["variants"]["threshold_20"]["folds"]
            ]
        )
        study_result["contrasts"]["threshold_20_vs_threshold_15"] = block_bootstrap(
            threshold_20_rmse - threshold_15_rmse,
            args.bootstrap_replicates,
            args.seed + study_index * 100 + 99,
        )
        for scale_index, (first_scale, second_scale) in enumerate(
            zip(GRAVITY_SCALES[:-1], GRAVITY_SCALES[1:]), start=1
        ):
            first_name = f"gravity_scale_{int(first_scale)}"
            second_name = f"gravity_scale_{int(second_scale)}"
            first_rmse = np.asarray(
                [row["rmse"] for row in study_result["variants"][first_name]["folds"]]
            )
            second_rmse = np.asarray(
                [row["rmse"] for row in study_result["variants"][second_name]["folds"]]
            )
            study_result["contrasts"][
                f"scale_{int(second_scale)}_vs_{int(first_scale)}"
            ] = block_bootstrap(
                second_rmse - first_rmse,
                args.bootstrap_replicates,
                args.seed + study_index * 100 + 200 + scale_index,
            )
        oof = bundle["frame"][["lsoa21cd", "geometry", "fold", "observed"]].copy()
        for variant, prediction in predictions.items():
            oof[f"prediction_{variant}"] = prediction
            oof[f"residual_{variant}"] = oof["observed"].to_numpy() - prediction
        oof_path = args.output_dir / "oof" / f"{study}.parquet"
        oof.to_parquet(oof_path, index=False)
        study_result["oof"] = str(oof_path.resolve())
        result["studies"][study] = study_result

    result = add_contrast_fdr(result)
    performance = pd.DataFrame(performance_rows)
    performance_path = args.output_dir / "performance.csv"
    performance.to_csv(performance_path, index=False)
    figure_path = args.output_dir / "distance_effect.png"
    plot_results(performance, figure_path)
    result["outputs"] = {
        "performance": str(performance_path.resolve()),
        "figure": str(figure_path.resolve()),
    }
    output_path = args.output_dir / "distance_effect_results.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
