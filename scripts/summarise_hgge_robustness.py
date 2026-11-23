from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from run_poi_distance_breakpoint import load_case, make_spatial_blocks
except ModuleNotFoundError:
    from scripts.run_poi_distance_breakpoint import load_case, make_spatial_blocks


STUDIES = ("greater_london", "greater_manchester")
HGGE_TABPFN_WEIGHT = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise five-seed HGGE stability and healthcare-edge ablation."
    )
    parser.add_argument(
        "--full-graph-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions_gated_gravity_cv5_relation_access_seeds5"),
    )
    parser.add_argument(
        "--no-healthcare-graph-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions_no_ahah_healthcare_relations_cv5_seeds5"),
    )
    parser.add_argument(
        "--tabpfn-dir",
        type=Path,
        default=Path("results/tabpfn_gravity/screen_v2_e1_oof"),
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/hgge_robustness")
    )
    parser.add_argument("--spatial-blocks", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - observed
    total = np.sum((observed - observed.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / max(float(total), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def macro_metrics(
    observed: np.ndarray, prediction: np.ndarray, folds: np.ndarray
) -> dict[str, float]:
    rows = [
        metrics(observed[folds == fold], prediction[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("r2", "rmse", "mae")
    }


def block_bootstrap_rmse_difference(
    observed: np.ndarray,
    full_prediction: np.ndarray,
    ablated_prediction: np.ndarray,
    spatial_block: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    unique_blocks = np.unique(spatial_block)
    n_by_block = np.asarray(
        [np.sum(spatial_block == block) for block in unique_blocks], dtype=float
    )
    full_sse = np.asarray(
        [
            np.sum((observed[spatial_block == block] - full_prediction[spatial_block == block]) ** 2)
            for block in unique_blocks
        ]
    )
    ablated_sse = np.asarray(
        [
            np.sum((observed[spatial_block == block] - ablated_prediction[spatial_block == block]) ** 2)
            for block in unique_blocks
        ]
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique_blocks),
        np.full(len(unique_blocks), 1.0 / len(unique_blocks)),
        size=replicates,
    )
    sample_n = counts @ n_by_block
    full_rmse = np.sqrt((counts @ full_sse) / sample_n)
    ablated_rmse = np.sqrt((counts @ ablated_sse) / sample_n)
    difference = ablated_rmse - full_rmse
    return {
        "ablated_minus_full_rmse": float(
            metrics(observed, ablated_prediction)["rmse"]
            - metrics(observed, full_prediction)["rmse"]
        ),
        "bootstrap_lower": float(np.quantile(difference, 0.025)),
        "bootstrap_upper": float(np.quantile(difference, 0.975)),
        "bootstrap_share_above_zero": float(np.mean(difference > 0.0)),
    }


def load_graph_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []

    for study_index, study in enumerate(STUDIES):
        full = load_graph_payload(args.full_graph_dir / f"{study}.npz")
        ablated = load_graph_payload(
            args.no_healthcare_graph_dir / f"{study}.npz"
        )
        invariant_keys = ("lsoa_codes", "observed", "fold", "seed")
        for key in invariant_keys:
            if not np.array_equal(full[key], ablated[key]):
                raise ValueError(f"full and ablated outputs differ for {study} {key}")
        tabpfn = pd.read_csv(args.tabpfn_dir / f"{study}_context_only.csv")
        codes = full["lsoa_codes"].astype(str)
        if not np.array_equal(codes, tabpfn["lsoa21cd"].astype(str).to_numpy()):
            raise ValueError(f"TabPFN row order differs for {study}")
        observed = full["observed"].astype(float)
        folds = full["fold"].astype(int)
        tabpfn_prediction = tabpfn["prediction"].to_numpy(float)

        for seed, full_graph, ablated_graph in zip(
            full["seed"].astype(int),
            full["seed_predictions"].astype(float),
            ablated["seed_predictions"].astype(float),
            strict=True,
        ):
            full_prediction = (
                HGGE_TABPFN_WEIGHT * tabpfn_prediction
                + (1.0 - HGGE_TABPFN_WEIGHT) * full_graph
            )
            ablated_prediction = (
                HGGE_TABPFN_WEIGHT * tabpfn_prediction
                + (1.0 - HGGE_TABPFN_WEIGHT) * ablated_graph
            )
            full_metrics = macro_metrics(observed, full_prediction, folds)
            ablated_metrics = macro_metrics(observed, ablated_prediction, folds)
            seed_rows.append(
                {
                    "study": study,
                    "seed": int(seed),
                    **{f"full_{key}": value for key, value in full_metrics.items()},
                    **{
                        f"no_ahah_healthcare_{key}": value
                        for key, value in ablated_metrics.items()
                    },
                    "no_ahah_healthcare_minus_full_rmse": (
                        ablated_metrics["rmse"] - full_metrics["rmse"]
                    ),
                    "no_ahah_healthcare_minus_full_r2": (
                        ablated_metrics["r2"] - full_metrics["r2"]
                    ),
                }
            )

        full_prediction = (
            HGGE_TABPFN_WEIGHT * tabpfn_prediction
            + (1.0 - HGGE_TABPFN_WEIGHT)
            * full["ensemble_prediction"].astype(float)
        )
        ablated_prediction = (
            HGGE_TABPFN_WEIGHT * tabpfn_prediction
            + (1.0 - HGGE_TABPFN_WEIGHT)
            * ablated["ensemble_prediction"].astype(float)
        )
        case = load_case(args.case_root, args.oof_dir, study)
        blocks = make_spatial_blocks(
            np.asarray(case["coordinates"]),
            args.spatial_blocks,
            args.seed + study_index,
        )
        comparison = block_bootstrap_rmse_difference(
            observed,
            full_prediction,
            ablated_prediction,
            blocks,
            args.bootstrap_replicates,
            args.seed + study_index,
        )
        full_metrics = macro_metrics(observed, full_prediction, folds)
        ablated_metrics = macro_metrics(observed, ablated_prediction, folds)
        improvement = full_metrics["rmse"] - ablated_metrics["rmse"]
        ablation_rows.append(
            {
                "study": study,
                **{f"full_{key}": value for key, value in full_metrics.items()},
                **{
                    f"no_ahah_healthcare_{key}": value
                    for key, value in ablated_metrics.items()
                },
                "full_minus_no_ahah_healthcare_rmse": improvement,
                **comparison,
            }
        )

    seeds = pd.DataFrame(seed_rows)
    ablation = pd.DataFrame(ablation_rows)
    seeds.to_csv(args.output_dir / "hgge_performance_by_seed.csv", index=False)
    ablation.to_csv(
        args.output_dir / "hgge_no_ahah_healthcare_relation_ablation.csv",
        index=False,
    )
    summary = {
        "design": {
            "seeds": sorted(seeds["seed"].unique().astype(int).tolist()),
            "excluded_relations": [
                "nhs_gp",
                "nhs_dentist",
                "osm_hospital",
                "osm_pharmacy",
            ],
            "ensemble": "fixed 50:50 TabPFN and graph-hybrid prediction",
            "uncertainty": (
                f"{args.bootstrap_replicates} resamples of {args.spatial_blocks} "
                "spatial blocks for the five-seed ensemble RMSE difference"
            ),
        },
        "seed_robustness": {
            study: {
                metric: {
                    "mean": float(frame[metric].mean()),
                    "sd": float(frame[metric].std(ddof=1)),
                    "minimum": float(frame[metric].min()),
                    "maximum": float(frame[metric].max()),
                }
                for metric in ("full_r2", "full_rmse", "full_mae")
            }
            for study, frame in seeds.groupby("study", sort=False)
        },
        "healthcare_relation_ablation": ablation.to_dict(orient="records"),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
