from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scripts.run_distance_effect_experiment import (
    GRAVITY_SCALES,
    bh_fdr,
    aggregate,
    block_bootstrap,
    fit_model,
    graph_distance_features,
    metrics,
)


CORE_RELATIONS = ("nhs_gp", "osm_hospital", "os_green", "osm_fast_food")
VARIANTS = (
    "no_distance_40",
    "threshold_15",
    "threshold_20",
    "segmented_15_20",
    "gravity_scale_5",
    "gravity_scale_10",
    "gravity_scale_15",
    "gravity_scale_20",
    "gravity_scale_30",
    "gravity_scale_40",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relation-specific facility distance-effect experiment"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--relation", action="append")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--overall-results",
        type=Path,
        default=Path("results/distance_effect/distance_effect_results.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/distance_effect/relation_specific_results.json")
    )
    return parser


def load_case(case_root: Path, oof_dir: Path, study: str) -> dict:
    base, variants = graph_distance_features(
        case_root / study / "facility_lsoa_graph.npz"
    )
    oof = gpd.read_parquet(oof_dir / f"{study}.parquet")
    if not np.array_equal(oof["lsoa21cd"].astype(str).to_numpy(), base["lsoa_codes"]):
        raise ValueError(f"OOF/graph order mismatch for {study}")
    return {
        "base_x": base["x"],
        "y": oof["observed"].to_numpy(dtype=float),
        "fold": oof["fold"].to_numpy(dtype=int),
        "relations": base["relation_names"],
        "variants": variants,
    }


def relation_features(bundle: dict, relation: str, variant: str) -> np.ndarray:
    relation_index = bundle["relations"].index(relation)
    values = bundle["variants"][variant]
    width = 6 if variant == "segmented_15_20" else 2
    start = relation_index * width
    return values[:, start : start + width]


def design_matrix(
    bundles: dict[str, dict],
    studies: list[str],
    target_study: str,
    relation: str,
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
                    relation_features(bundle, relation, variant)[selector],
                    city,
                )
            )
        )
        y_blocks.append(bundle["y"][selector])
    return np.concatenate(x_blocks), np.concatenate(y_blocks)


def rmse_by_fold(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["rmse"] for row in rows], dtype=float)


def add_relation_contrast_fdr(result: dict) -> dict:
    families: dict[str, list[dict]] = {
        "versus_context": [],
        "versus_no_distance": [],
        "threshold_20_vs_15": [],
        "adjacent_gravity_scales": [],
    }
    for study in result["studies"].values():
        for relation in study["relations"].values():
            for name, contrast in relation["contrasts"].items():
                if name == "threshold_20_vs_threshold_15":
                    families["threshold_20_vs_15"].append(contrast)
                elif name.startswith("scale_"):
                    families["adjacent_gravity_scales"].append(contrast)
                elif name.endswith("_vs_context_only"):
                    families["versus_context"].append(contrast)
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
        "Benjamini-Hochberg within contrast family across both cities and all requested relations"
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    if len(args.study) != 2:
        raise ValueError("this frozen experiment requires exactly two cities")
    relations = tuple(args.relation or CORE_RELATIONS)
    bundles = {
        study: load_case(args.case_root, args.oof_dir, study) for study in args.study
    }
    for study, bundle in bundles.items():
        missing = [relation for relation in relations if relation not in bundle["relations"]]
        if missing:
            raise ValueError(f"missing relations for {study}: {missing}")
    overall = json.loads(args.overall_results.read_text(encoding="utf-8"))
    result = {
        "settings": {
            "studies": args.study,
            "relations": relations,
            "variants": VARIANTS,
            "gravity_scales_minutes": GRAVITY_SCALES,
            "paired_seed_within_relation_city_fold": True,
            "validation": "five frozen contiguous spatial folds",
            "model": "fixed two-city HGB; one relation added to 103 context features",
            "bootstrap": "resample five frozen folds with replacement",
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "target_derived_features": False,
            "absolute_coordinate_features": False,
        },
        "studies": {},
    }
    performance_rows = []
    for study_index, study in enumerate(args.study):
        bundle = bundles[study]
        unique_folds = sorted(int(value) for value in np.unique(bundle["fold"]))
        city_result = {"relations": {}}
        context_rows = overall["studies"][study]["variants"]["context_only"]["folds"]
        context_rmse = rmse_by_fold(context_rows)
        for relation_index, relation in enumerate(relations):
            relation_result = {"variants": {}, "contrasts": {}}
            for variant_index, variant in enumerate(VARIANTS):
                prediction = np.full(len(bundle["y"]), np.nan, dtype=float)
                fold_rows = []
                for fold in unique_folds:
                    train_selector = bundle["fold"] != fold
                    test_selector = bundle["fold"] == fold
                    train_x, train_y = design_matrix(
                        bundles,
                        args.study,
                        study,
                        relation,
                        variant,
                        train_selector,
                    )
                    test_city = np.zeros((int(test_selector.sum()), len(args.study)))
                    test_city[:, study_index] = 1.0
                    test_x = np.column_stack(
                        (
                            bundle["base_x"][test_selector],
                            relation_features(bundle, relation, variant)[test_selector],
                            test_city,
                        )
                    )
                    fitted = fit_model(
                        train_x,
                        train_y,
                        args.seed
                        + study_index * 10000
                        + relation_index * 100
                        + fold,
                    )
                    fold_prediction = fitted.predict(test_x)
                    prediction[test_selector] = fold_prediction
                    fold_rows.append(
                        {
                            "fold": fold,
                            **metrics(bundle["y"][test_selector], fold_prediction),
                        }
                    )
                relation_result["variants"][variant] = {
                    "folds": fold_rows,
                    "macro": aggregate(fold_rows),
                    "pooled_oof": metrics(bundle["y"], prediction),
                }
                performance_rows.append(
                    {
                        "study": study,
                        "relation": relation,
                        "variant": variant,
                        "r2": relation_result["variants"][variant]["macro"]["r2"]["mean"],
                        "rmse": relation_result["variants"][variant]["macro"]["rmse"]["mean"],
                    }
                )
            no_distance_rmse = rmse_by_fold(
                relation_result["variants"]["no_distance_40"]["folds"]
            )
            for variant_index, variant in enumerate(VARIANTS):
                variant_rmse = rmse_by_fold(relation_result["variants"][variant]["folds"])
                relation_result["contrasts"][f"{variant}_vs_context_only"] = block_bootstrap(
                    variant_rmse - context_rmse,
                    args.bootstrap_replicates,
                    args.seed + study_index * 1000 + relation_index * 100 + variant_index,
                )
                if variant != "no_distance_40":
                    relation_result["contrasts"][f"{variant}_vs_no_distance_40"] = block_bootstrap(
                        variant_rmse - no_distance_rmse,
                        args.bootstrap_replicates,
                        args.seed
                        + study_index * 1000
                        + relation_index * 100
                        + 20
                        + variant_index,
                    )
            threshold_15_rmse = rmse_by_fold(
                relation_result["variants"]["threshold_15"]["folds"]
            )
            threshold_20_rmse = rmse_by_fold(
                relation_result["variants"]["threshold_20"]["folds"]
            )
            relation_result["contrasts"]["threshold_20_vs_threshold_15"] = block_bootstrap(
                threshold_20_rmse - threshold_15_rmse,
                args.bootstrap_replicates,
                args.seed + study_index * 1000 + relation_index * 100 + 90,
            )
            scale_rows = {
                int(scale): rmse_by_fold(
                    relation_result["variants"][f"gravity_scale_{int(scale)}"]["folds"]
                )
                for scale in GRAVITY_SCALES
            }
            for scale_index, (first_scale, second_scale) in enumerate(
                zip(GRAVITY_SCALES[:-1], GRAVITY_SCALES[1:])
            ):
                relation_result["contrasts"][
                    f"scale_{int(second_scale)}_vs_{int(first_scale)}"
                ] = block_bootstrap(
                    scale_rows[int(second_scale)] - scale_rows[int(first_scale)],
                    args.bootstrap_replicates,
                    args.seed
                    + study_index * 1000
                    + relation_index * 100
                    + 100
                    + scale_index,
                )
            gravity_performance = {
                int(scale): relation_result["variants"][f"gravity_scale_{int(scale)}"][
                    "macro"
                ]["rmse"]["mean"]
                for scale in GRAVITY_SCALES
            }
            relation_result["exploratory_best_gravity_scale"] = min(
                gravity_performance, key=gravity_performance.get
            )
            city_result["relations"][relation] = relation_result
            best_scale = relation_result["exploratory_best_gravity_scale"]
            print(
                json.dumps(
                    {
                        "event": "relation_complete",
                        "study": study,
                        "relation": relation,
                        "best_gravity_scale": best_scale,
                        "best_gravity_r2": relation_result["variants"][
                            f"gravity_scale_{best_scale}"
                        ]["macro"]["r2"]["mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        result["studies"][study] = city_result
    result = add_relation_contrast_fdr(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    performance_path = args.output.with_name("relation_specific_performance.csv")
    pd.DataFrame(performance_rows).to_csv(performance_path, index=False)
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
