from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CITIES = ("greater_london", "greater_manchester")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit across-seed reliability of HGGE relation counterfactuals, "
            "fold-specific graph weights, gravity-exposure agreement, and "
            "distance-kernel seed dispersion."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_predictions_"
            "gated_gravity_cv5_relation_access_seeds5"
        ),
    )
    parser.add_argument(
        "--graph-result",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_gated_gravity_gnn_"
            "hgb_residual_cv5_relation_access_seeds5.json"
        ),
    )
    parser.add_argument(
        "--hgge-adjusted",
        type=Path,
        default=Path(
            "results/hgge_informed_associations_seeds5_e5/"
            "adjusted_hgge_oof.parquet"
        ),
    )
    parser.add_argument(
        "--gravity-adjusted",
        type=Path,
        default=Path(
            "results/gravity_derived_gradients/adjusted_gravity_oof.parquet"
        ),
    )
    parser.add_argument(
        "--gated-seeds",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_predictions_"
            "gated_gravity_cv5_relation_access_seeds5/"
            "graph_hybrid_performance_by_seed.csv"
        ),
    )
    parser.add_argument(
        "--hard-seeds",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_predictions_"
            "hard_gravity_cv5_seeds5/graph_hybrid_performance_by_seed.csv"
        ),
    )
    parser.add_argument(
        "--no-gravity-seeds",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_predictions_"
            "no_gravity_cv5_seeds5/graph_hybrid_performance_by_seed.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hgge_exposure_reliability_audit"),
    )
    return parser


def one_way_random_effects_icc(values: np.ndarray) -> dict[str, float]:
    """Return one-way random-effects ICCs for items by repeated model seeds."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("ICC input must contain at least two areas and two seeds")
    if not np.isfinite(data).all():
        raise ValueError("ICC input contains non-finite values")
    area_count, seed_count = data.shape
    area_mean = data.mean(axis=1)
    grand_mean = float(data.mean())
    ms_between = float(
        seed_count
        * np.square(area_mean - grand_mean).sum()
        / (area_count - 1)
    )
    ms_within = float(
        np.square(data - area_mean[:, None]).sum()
        / (area_count * (seed_count - 1))
    )
    single_denominator = ms_between + (seed_count - 1) * ms_within
    icc_single = (
        (ms_between - ms_within) / single_denominator
        if single_denominator > 0
        else float("nan")
    )
    icc_seed_mean = (
        (ms_between - ms_within) / ms_between
        if ms_between > 0
        else float("nan")
    )
    return {
        "ms_between": ms_between,
        "ms_within": ms_within,
        "icc_single_seed": float(icc_single),
        "icc_seed_mean": float(icc_seed_mean),
    }


def finite_pairwise_correlations(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    correlations = []
    for first in range(data.shape[1]):
        for second in range(first + 1, data.shape[1]):
            left = data[:, first]
            right = data[:, second]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return np.asarray(correlations, dtype=float)


def reliability_rows(prediction_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for city in DEFAULT_CITIES:
        with np.load(prediction_dir / f"{city}.npz") as payload:
            relation_names = payload["relation_names"].astype(str)
            shifts = payload["relation_access_shift_minutes"].astype(float)
            access = payload["seed_relation_access"].astype(float)
            if access.ndim != 4:
                raise ValueError(f"invalid counterfactual array for {city}")
            for relation_index, relation in enumerate(relation_names):
                for shift_index, shift in enumerate(shifts):
                    # Stored order is seed, relation, area, shift. ICC uses area, seed.
                    values = access[:, relation_index, :, shift_index].T
                    area_mean = values.mean(axis=1)
                    within_area_variance = values.var(axis=1, ddof=1)
                    pairwise = finite_pairwise_correlations(values)
                    icc = one_way_random_effects_icc(values)
                    between_sd = float(area_mean.std(ddof=1))
                    within_rms_sd = float(np.sqrt(within_area_variance.mean()))
                    rows.append(
                        {
                            "study": city,
                            "relation": relation,
                            "shift_minutes": float(shift),
                            "area_count": int(values.shape[0]),
                            "seed_count": int(values.shape[1]),
                            "ensemble_mean": float(area_mean.mean()),
                            "ensemble_mean_absolute": float(np.abs(area_mean).mean()),
                            "between_area_sd_of_seed_mean": between_sd,
                            "within_area_seed_sd_mean": float(
                                np.sqrt(within_area_variance).mean()
                            ),
                            "within_area_seed_sd_rms": within_rms_sd,
                            "between_to_within_sd_ratio": (
                                between_sd / within_rms_sd
                                if within_rms_sd > 0
                                else float("nan")
                            ),
                            "pairwise_seed_correlation_mean": (
                                float(pairwise.mean())
                                if pairwise.size
                                else float("nan")
                            ),
                            "pairwise_seed_correlation_min": (
                                float(pairwise.min())
                                if pairwise.size
                                else float("nan")
                            ),
                            **icc,
                        }
                    )
    return rows


def fold_weight_rows(graph_result: Path) -> list[dict[str, object]]:
    result = json.loads(graph_result.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for city in DEFAULT_CITIES:
        weights: dict[int, list[float]] = {}
        city_result = result["within_city"][city]
        seed_runs = city_result.get(
            "source_seed_records", city_result.get("seed_runs", [])
        )
        if not seed_runs:
            raise ValueError(f"missing seed-level fold records for {city}")
        for seed_run in seed_runs:
            for fold in seed_run["folds"]:
                weights.setdefault(int(fold["test_fold"]), []).append(
                    float(fold["selected_residual_weight"])
                )
        for fold, values_list in sorted(weights.items()):
            values = np.asarray(values_list, dtype=float)
            rows.append(
                {
                    "study": city,
                    "test_fold": fold,
                    "seed_count": int(len(values)),
                    "weights": ";".join(f"{value:.2f}" for value in values),
                    "weight_mean": float(values.mean()),
                    "weight_sd": float(values.std(ddof=1)),
                    "weight_min": float(values.min()),
                    "weight_max": float(values.max()),
                    "zero_weight_count": int(np.sum(values == 0)),
                }
            )
    return rows


def exposure_correlation_rows(
    hgge_adjusted: Path, gravity_adjusted: Path
) -> list[dict[str, object]]:
    def correlation(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
        if rank:
            left = left.rank(method="average")
            right = right.rank(method="average")
        return float(left.corr(right, method="pearson"))

    keys = ["study", "lsoa21cd", "relation"]
    graph = pd.read_parquet(hgge_adjusted)[
        keys + ["hgge_access_exposure_z", "hgge_access_delta_5min"]
    ].drop_duplicates(keys)
    gravity = pd.read_parquet(gravity_adjusted)[
        keys + ["gravity_exposure_z", "gravity_delta_5min"]
    ].drop_duplicates(keys)
    merged = graph.merge(gravity, on=keys, how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for (city, relation), frame in merged.groupby(["study", "relation"]):
        rows.append(
            {
                "study": city,
                "relation": relation,
                "area_count": int(len(frame)),
                "level_pearson": correlation(
                    frame["hgge_access_exposure_z"],
                    frame["gravity_exposure_z"],
                ),
                "level_spearman": correlation(
                    frame["hgge_access_exposure_z"],
                    frame["gravity_exposure_z"],
                    rank=True,
                ),
                "delta_5min_pearson": correlation(
                    frame["hgge_access_delta_5min"],
                    frame["gravity_delta_5min"],
                ),
                "delta_5min_spearman": correlation(
                    frame["hgge_access_delta_5min"],
                    frame["gravity_delta_5min"],
                    rank=True,
                ),
            }
        )
    return rows


def kernel_seed_dispersion_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, path in paths.items():
        frame = pd.read_csv(path)
        for city, group in frame.groupby("study"):
            for metric in ("r2", "rmse", "mae"):
                values = group[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "variant": variant,
                        "study": city,
                        "metric": metric,
                        "seed_count": int(len(values)),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
    return rows


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reliability = pd.DataFrame(reliability_rows(args.prediction_dir))
    fold_weights = pd.DataFrame(fold_weight_rows(args.graph_result))
    exposure_correlations = pd.DataFrame(
        exposure_correlation_rows(args.hgge_adjusted, args.gravity_adjusted)
    )
    kernel_dispersion = pd.DataFrame(
        kernel_seed_dispersion_rows(
            {
                "gated_gravity": args.gated_seeds,
                "hard_gravity": args.hard_seeds,
                "no_gravity": args.no_gravity_seeds,
            }
        )
    )

    outputs = {
        "counterfactual_reliability": reliability,
        "fold_residual_weights": fold_weights,
        "graph_gravity_exposure_correlations": exposure_correlations,
        "kernel_seed_dispersion": kernel_dispersion,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)

    observed_access = reliability[reliability["shift_minutes"] == 0].copy()
    primary = observed_access[
        observed_access["relation"].isin(("nhs_gp", "osm_pharmacy"))
    ]
    summary = {
        "design": (
            "Across-seed measurement reliability audit of strict-OOF relation "
            "counterfactuals; negative ICC estimates are retained rather than clipped."
        ),
        "reliability_interpretation": (
            "icc_seed_mean measures reliability of the five-seed mean under a "
            "one-way random-effects model; it does not establish construct validity."
        ),
        "primary_care_observed_access": records(primary),
        "fold_residual_weights": records(fold_weights),
        "graph_gravity_exposure_correlations": records(exposure_correlations),
        "kernel_rmse_seed_dispersion": records(
            kernel_dispersion[kernel_dispersion["metric"] == "rmse"]
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
