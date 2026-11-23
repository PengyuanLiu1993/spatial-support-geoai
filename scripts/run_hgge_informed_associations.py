from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MPL_CACHE = Path("tmp/matplotlib").resolve()
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats

try:
    from run_gravity_derived_gradients import (
        TIME_BANDS,
        block_bootstrap_primary,
        contrast,
        independent_city_difference,
        relation_edges,
    )
    from run_poi_distance_breakpoint import (
        FIXED_THRESHOLDS,
        RELATION_LABELS,
        STUDIES,
        STUDY_LABELS,
        bh_fdr,
        cluster_robust_inference,
        load_case,
        make_spatial_blocks,
        relation_context,
    )
    from run_vulnerability_distance_heterogeneity import (
        MODERATORS,
        MODERATOR_LABELS,
        PRIMARY_RELATIONS,
        fit_nuisance,
        load_census_features,
    )
except ModuleNotFoundError:
    from scripts.run_gravity_derived_gradients import (
        TIME_BANDS,
        block_bootstrap_primary,
        contrast,
        independent_city_difference,
        relation_edges,
    )
    from scripts.run_poi_distance_breakpoint import (
        FIXED_THRESHOLDS,
        RELATION_LABELS,
        STUDIES,
        STUDY_LABELS,
        bh_fdr,
        cluster_robust_inference,
        load_case,
        make_spatial_blocks,
        relation_context,
    )
    from scripts.run_vulnerability_distance_heterogeneity import (
        MODERATORS,
        MODERATOR_LABELS,
        PRIMARY_RELATIONS,
        fit_nuisance,
        load_census_features,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use strict-OOF HGGE predictions and relation counterfactuals in the "
            "threshold and Census-health association stages."
        )
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument(
        "--hgge-relation-dir",
        type=Path,
        default=Path(
            "results/heterogeneous_facility_predictions_gated_gravity_cv5_relation_access"
        ),
    )
    parser.add_argument(
        "--hgge-ensemble-dir",
        type=Path,
        default=Path("results/heterogeneous_graph_ensemble_relation_access"),
    )
    parser.add_argument(
        "--census-root",
        type=Path,
        default=Path("data/raw/census2021_vulnerability_2026_08_28"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hgge_informed_associations"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--spatial-blocks", type=int, default=25)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument(
        "--relations",
        nargs="+",
        choices=tuple(RELATION_LABELS),
        default=list(RELATION_LABELS),
    )
    return parser


def load_hgge_outputs(
    relation_dir: Path,
    ensemble_dir: Path,
    study: str,
    expected_codes: np.ndarray,
    expected_folds: np.ndarray,
) -> dict[str, object]:
    relation_path = relation_dir / f"{study}.npz"
    relation = np.load(relation_path, allow_pickle=False)
    codes = relation["lsoa_codes"].astype(str)
    folds = relation["fold"].astype(int)
    if not np.array_equal(codes, expected_codes.astype(str)):
        raise ValueError(f"HGGE relation row order differs for {study}")
    if not np.array_equal(folds, expected_folds):
        raise ValueError(f"HGGE relation fold plan differs for {study}")
    access = relation["ensemble_relation_access"].astype(float)
    relation_names = relation["relation_names"].astype(str).tolist()
    shifts = relation["relation_access_shift_minutes"].astype(float)
    if access.shape != (len(relation_names), len(codes), len(shifts)):
        raise ValueError(f"invalid HGGE relation-access shape for {study}")
    if len(shifts) < 2 or shifts[0] != 0.0 or 5.0 not in shifts:
        raise ValueError("HGGE relation access must include 0- and 5-minute shifts")

    ensemble = pd.read_parquet(ensemble_dir / f"{study}.parquet")
    if not np.array_equal(
        ensemble["lsoa21cd"].astype(str).to_numpy(), expected_codes.astype(str)
    ):
        raise ValueError(f"HGGE ensemble row order differs for {study}")
    prediction = ensemble[
        "prediction_heterogeneous_graph_ensemble"
    ].to_numpy(float)
    return {
        "relation_names": relation_names,
        "shifts": shifts,
        "access": access,
        "prediction": prediction,
        "relation_path": relation_path,
    }


def cross_fitted_standardize(
    raw: np.ndarray,
    shifts: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, dict[float, np.ndarray], list[dict[str, float]]]:
    if raw.shape != (len(folds), len(shifts)):
        raise ValueError("relation counterfactual array has an invalid shape")
    exposure = np.full(len(folds), np.nan, dtype=float)
    shifted = {
        float(shift): np.full(len(folds), np.nan, dtype=float) for shift in shifts
    }
    scaling: list[dict[str, float]] = []
    for outer_fold in sorted(int(value) for value in np.unique(folds)):
        train = folds != outer_fold
        test = folds == outer_fold
        mean = float(np.mean(raw[train, 0]))
        standard_deviation = float(np.std(raw[train, 0], ddof=0))
        if standard_deviation <= 1e-10:
            raise RuntimeError(
                f"degenerate HGGE relation-access contrast in fold {outer_fold}"
            )
        exposure[test] = (raw[test, 0] - mean) / standard_deviation
        for shift_index, shift in enumerate(shifts):
            shifted[float(shift)][test] = (
                raw[test, shift_index] - mean
            ) / standard_deviation
        scaling.append(
            {
                "outer_fold": float(outer_fold),
                "training_mean": mean,
                "training_sd": standard_deviation,
            }
        )
    if not np.isfinite(exposure).all() or any(
        not np.isfinite(values).all() for values in shifted.values()
    ):
        raise RuntimeError("incomplete cross-fitted HGGE relation exposure")
    return exposure, shifted, scaling


def access_design(
    exposure: np.ndarray,
    moderators_z: np.ndarray,
    selected_moderator: int,
    censored: np.ndarray,
) -> np.ndarray:
    moderator = moderators_z[:, selected_moderator]
    return np.column_stack(
        (
            np.ones(len(exposure)),
            exposure,
            moderators_z,
            exposure * moderator,
            censored,
        )
    )


def learned_access_interaction(
    study: str,
    relation: str,
    moderator: str,
    residual: np.ndarray,
    exposure: np.ndarray,
    shifted: dict[float, np.ndarray],
    nearest: np.ndarray,
    censored: np.ndarray,
    moderators_raw: np.ndarray,
    moderators_z: np.ndarray,
    moderator_index: int,
    spatial_block: np.ndarray,
    population: np.ndarray,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    design = access_design(exposure, moderators_z, moderator_index, censored)
    inference = cluster_robust_inference(design, residual, spatial_block)
    beta = inference["beta"]
    covariance = inference["covariance"]
    bootstrap_beta = spatial_block_bootstrap_coefficients(
        design,
        residual,
        spatial_block,
        bootstrap_replicates,
        bootstrap_seed,
    )
    clusters = len(np.unique(spatial_block))
    interaction_vector = np.zeros(len(beta))
    interaction_vector[5] = 1.0
    interaction = contrast(beta, covariance, interaction_vector, clusters)
    interaction_bootstrap = bootstrap_coefficient_summary(
        bootstrap_beta[:, 5], float(beta[5])
    )

    raw_moderator = moderators_raw[:, moderator_index]
    raw_q25, raw_q75 = np.quantile(raw_moderator, [0.25, 0.75])
    raw_mean = float(np.mean(raw_moderator))
    raw_sd = float(np.std(raw_moderator, ddof=0))
    z25 = float((raw_q25 - raw_mean) / raw_sd)
    z75 = float((raw_q75 - raw_mean) / raw_sd)
    connected = censored == 0
    delta_five = shifted[5.0] - exposure

    penalty_rows: list[dict[str, object]] = []
    for band_index, (band_label, lower, upper) in enumerate(TIME_BANDS):
        selected = connected & (nearest > lower) & (nearest <= upper)
        if band_index == 0:
            selected = connected & (nearest <= upper)
        if selected.sum() < 25:
            continue
        mean_delta = float(np.mean(delta_five[selected]))
        low_vector = np.zeros(len(beta))
        low_vector[1] = mean_delta
        low_vector[5] = mean_delta * z25
        high_vector = np.zeros(len(beta))
        high_vector[1] = mean_delta
        high_vector[5] = mean_delta * z75
        difference_vector = high_vector - low_vector
        low_effect = contrast(beta, covariance, low_vector, clusters)
        high_effect = contrast(beta, covariance, high_vector, clusters)
        difference = contrast(beta, covariance, difference_vector, clusters)
        low_bootstrap = bootstrap_coefficient_summary(
            bootstrap_beta @ low_vector, float(low_effect["estimate"])
        )
        high_bootstrap = bootstrap_coefficient_summary(
            bootstrap_beta @ high_vector, float(high_effect["estimate"])
        )
        difference_bootstrap = bootstrap_coefficient_summary(
            bootstrap_beta @ difference_vector, float(difference["estimate"])
        )
        median_population = float(np.median(population[selected]))
        band_population = float(np.sum(population[selected]))
        penalty_rows.append(
            {
                "study": study,
                "outcome": "bad_health_pct",
                "relation": relation,
                "moderator": moderator,
                "band": band_label,
                "band_index": band_index,
                "n_band": int(selected.sum()),
                "median_lsoa_population": median_population,
                "band_population": band_population,
                "clusters": clusters,
                "mean_hgge_access_change_for_5min_delay": mean_delta,
                "low_penalty": low_effect["estimate"],
                "low_penalty_lower": low_effect["lower"],
                "low_penalty_upper": low_effect["upper"],
                "low_penalty_block_bootstrap_lower": low_bootstrap["lower"],
                "low_penalty_block_bootstrap_upper": low_bootstrap["upper"],
                "high_penalty": high_effect["estimate"],
                "high_penalty_lower": high_effect["lower"],
                "high_penalty_upper": high_effect["upper"],
                "high_penalty_block_bootstrap_lower": high_bootstrap["lower"],
                "high_penalty_block_bootstrap_upper": high_bootstrap["upper"],
                "penalty_difference": difference["estimate"],
                "penalty_difference_se": difference["se"],
                "penalty_difference_lower": difference["lower"],
                "penalty_difference_upper": difference["upper"],
                "penalty_difference_p": difference["p"],
                "penalty_difference_block_bootstrap_se": difference_bootstrap["se"],
                "penalty_difference_block_bootstrap_lower": difference_bootstrap[
                    "lower"
                ],
                "penalty_difference_block_bootstrap_upper": difference_bootstrap[
                    "upper"
                ],
                "penalty_difference_block_bootstrap_p": difference_bootstrap["p"],
                "low_penalty_people_per_median_lsoa": (
                    low_effect["estimate"] * median_population / 100.0
                ),
                "high_penalty_people_per_median_lsoa": (
                    high_effect["estimate"] * median_population / 100.0
                ),
                "penalty_difference_people_per_median_lsoa": (
                    difference["estimate"] * median_population / 100.0
                ),
                "low_penalty_people_if_bandwide": (
                    low_effect["estimate"] * band_population / 100.0
                ),
                "high_penalty_people_if_bandwide": (
                    high_effect["estimate"] * band_population / 100.0
                ),
            }
        )

    curve_rows: list[dict[str, object]] = []
    for shift, shifted_exposure in shifted.items():
        mean_delta = float(np.mean((shifted_exposure - exposure)[connected]))
        for stratum, z_value in (
            ("lower vulnerability", z25),
            ("higher vulnerability", z75),
        ):
            vector = np.zeros(len(beta))
            vector[1] = mean_delta
            vector[5] = mean_delta * z_value
            effect = contrast(beta, covariance, vector, clusters)
            effect_bootstrap = bootstrap_coefficient_summary(
                bootstrap_beta @ vector, float(effect["estimate"])
            )
            curve_rows.append(
                {
                    "study": study,
                    "outcome": "bad_health_pct",
                    "relation": relation,
                    "moderator": moderator,
                    "stratum": stratum,
                    "additional_minutes": shift,
                    "effect": effect["estimate"],
                    "effect_lower": effect["lower"],
                    "effect_upper": effect["upper"],
                    "effect_block_bootstrap_lower": effect_bootstrap["lower"],
                    "effect_block_bootstrap_upper": effect_bootstrap["upper"],
                }
            )
    summary = {
        "study": study,
        "outcome": "bad_health_pct",
        "relation": relation,
        "relation_label": RELATION_LABELS[relation],
        "moderator": moderator,
        "moderator_label": MODERATOR_LABELS[moderator],
        "n": len(residual),
        "clusters": clusters,
        "moderator_q25": float(raw_q25),
        "moderator_q75": float(raw_q75),
        "hgge_access_main_effect": float(beta[1]),
        "hgge_access_interaction": interaction["estimate"],
        "hgge_access_interaction_se": interaction["se"],
        "hgge_access_interaction_lower": interaction["lower"],
        "hgge_access_interaction_upper": interaction["upper"],
        "hgge_access_interaction_p": interaction["p"],
        "hgge_access_interaction_block_bootstrap_se": interaction_bootstrap["se"],
        "hgge_access_interaction_block_bootstrap_lower": interaction_bootstrap[
            "lower"
        ],
        "hgge_access_interaction_block_bootstrap_upper": interaction_bootstrap[
            "upper"
        ],
        "hgge_access_interaction_block_bootstrap_p": interaction_bootstrap["p"],
    }
    return summary, penalty_rows, curve_rows


def spatial_block_bootstrap_coefficients(
    design: np.ndarray,
    outcome: np.ndarray,
    spatial_block: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample complete spatial blocks and return OLS coefficient draws."""

    if replicates < 100:
        raise ValueError("at least 100 spatial-block bootstrap replicates are required")
    unique_blocks = np.unique(spatial_block)
    block_xtx = np.stack(
        [
            design[spatial_block == block].T @ design[spatial_block == block]
            for block in unique_blocks
        ]
    )
    block_xty = np.stack(
        [
            design[spatial_block == block].T @ outcome[spatial_block == block]
            for block in unique_blocks
        ]
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique_blocks),
        np.full(len(unique_blocks), 1.0 / len(unique_blocks)),
        size=replicates,
    )
    xtx = np.einsum("bg,gij->bij", counts, block_xtx, optimize=True)
    xty = np.einsum("bg,gj->bj", counts, block_xty, optimize=True)
    return np.einsum(
        "bij,bj->bi", np.linalg.pinv(xtx, rcond=1e-10), xty, optimize=True
    )


def bootstrap_coefficient_summary(
    draws: np.ndarray, estimate: float
) -> dict[str, float]:
    centered = draws - estimate
    p_value = (1.0 + np.sum(np.abs(centered) >= abs(estimate))) / (
        len(draws) + 1.0
    )
    return {
        "se": float(np.std(draws, ddof=1)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "p": float(min(p_value, 1.0)),
    }


def minimum_detectable_effect(
    standard_error: float,
    clusters: int,
    alpha: float,
    power: float,
) -> float:
    """Two-sided MDE from the noncentral-t distribution with cluster df."""

    if standard_error <= 0 or clusters < 3:
        return float("nan")
    degrees_freedom = clusters - 1
    critical = stats.t.ppf(1.0 - alpha / 2.0, degrees_freedom)

    def achieved(noncentrality: float) -> float:
        return float(
            stats.nct.cdf(-critical, degrees_freedom, noncentrality)
            + stats.nct.sf(critical, degrees_freedom, noncentrality)
        )

    upper = 1.0
    while achieved(upper) < power and upper < 30.0:
        upper *= 1.5
    if not np.isfinite(achieved(upper)) or achieved(upper) < power:
        raise RuntimeError("unable to bracket the noncentral-t MDE")
    noncentrality = optimize.brentq(
        lambda value: achieved(value) - power, 0.0, upper
    )
    return float(noncentrality * standard_error)


def minimum_detectable_effects(
    standard_error: float, clusters: int, family_tests: int
) -> dict[str, float]:
    return {
        "mde80_unadjusted": minimum_detectable_effect(
            standard_error, clusters, 0.05, 0.80
        ),
        "mde90_unadjusted": minimum_detectable_effect(
            standard_error, clusters, 0.05, 0.90
        ),
        "mde80_bonferroni": minimum_detectable_effect(
            standard_error, clusters, 0.05 / family_tests, 0.80
        ),
    }


def hgge_residual_threshold_tests(
    study: str,
    relation: str,
    residual: np.ndarray,
    nearest: np.ndarray,
    censored: np.ndarray,
    spatial_block: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clusters = len(np.unique(spatial_block))
    residual_sd = float(np.std(residual, ddof=1))
    for threshold_index, threshold in enumerate(FIXED_THRESHOLDS):
        design = np.column_stack(
            (
                np.ones(len(residual)),
                nearest / 5.0,
                (nearest > threshold).astype(float),
                np.maximum(nearest - threshold, 0.0) / 5.0,
                censored,
            )
        )
        inference = cluster_robust_inference(design, residual, spatial_block)
        beta = inference["beta"]
        covariance = inference["covariance"]
        step_vector = np.zeros(len(beta))
        step_vector[2] = 1.0
        hinge_vector = np.zeros(len(beta))
        hinge_vector[3] = 1.0
        step = contrast(beta, covariance, step_vector, clusters)
        hinge = contrast(beta, covariance, hinge_vector, clusters)
        bootstrap_beta = spatial_block_bootstrap_coefficients(
            design,
            residual,
            spatial_block,
            bootstrap_replicates,
            seed + threshold_index,
        )
        step_bootstrap = bootstrap_coefficient_summary(
            bootstrap_beta[:, 2], float(beta[2])
        )
        hinge_bootstrap = bootstrap_coefficient_summary(
            bootstrap_beta[:, 3], float(beta[3])
        )
        step_mde = minimum_detectable_effects(
            step_bootstrap["se"], clusters, family_tests=96
        )
        hinge_mde = minimum_detectable_effects(
            hinge_bootstrap["se"], clusters, family_tests=96
        )
        rows.append(
            {
                "study": study,
                "relation": relation,
                "threshold": threshold,
                "step": step["estimate"],
                "step_se": step["se"],
                "step_lower": step["lower"],
                "step_upper": step["upper"],
                "step_p": step["p"],
                "step_block_bootstrap_se": step_bootstrap["se"],
                "step_block_bootstrap_lower": step_bootstrap["lower"],
                "step_block_bootstrap_upper": step_bootstrap["upper"],
                "step_block_bootstrap_p": step_bootstrap["p"],
                "step_mde80_unadjusted": step_mde["mde80_unadjusted"],
                "step_mde80_bonferroni96": step_mde["mde80_bonferroni"],
                "step_mde90_unadjusted": step_mde["mde90_unadjusted"],
                "step_mde80_unadjusted_residual_sd": (
                    step_mde["mde80_unadjusted"] / residual_sd
                ),
                "step_mde80_bonferroni96_residual_sd": (
                    step_mde["mde80_bonferroni"] / residual_sd
                ),
                "post_threshold_deviation_per_5min": hinge["estimate"],
                "post_threshold_deviation_se": hinge["se"],
                "post_threshold_deviation_lower": hinge["lower"],
                "post_threshold_deviation_upper": hinge["upper"],
                "post_threshold_deviation_p": hinge["p"],
                "post_threshold_deviation_block_bootstrap_se": hinge_bootstrap[
                    "se"
                ],
                "post_threshold_deviation_block_bootstrap_lower": hinge_bootstrap[
                    "lower"
                ],
                "post_threshold_deviation_block_bootstrap_upper": hinge_bootstrap[
                    "upper"
                ],
                "post_threshold_deviation_block_bootstrap_p": hinge_bootstrap[
                    "p"
                ],
                "post_threshold_deviation_mde80_unadjusted": hinge_mde[
                    "mde80_unadjusted"
                ],
                "post_threshold_deviation_mde80_bonferroni96": hinge_mde[
                    "mde80_bonferroni"
                ],
            }
        )
    return rows


def compare_cities(
    summaries: pd.DataFrame, penalties: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_rows: list[dict[str, object]] = []
    for keys, frame in summaries.groupby(["relation", "moderator"], sort=True):
        indexed = frame.set_index("study")
        london = indexed.loc["greater_london"]
        manchester = indexed.loc["greater_manchester"]
        comparison = independent_city_difference(
            float(london["hgge_access_interaction"]),
            float(london["hgge_access_interaction_block_bootstrap_se"]),
            int(london["clusters"]),
            float(manchester["hgge_access_interaction"]),
            float(manchester["hgge_access_interaction_block_bootstrap_se"]),
            int(manchester["clusters"]),
        )
        interaction_rows.append(
            {
                "outcome": "bad_health_pct",
                "relation": keys[0],
                "moderator": keys[1],
                "london_interaction": float(london["hgge_access_interaction"]),
                "manchester_interaction": float(
                    manchester["hgge_access_interaction"]
                ),
                **comparison,
            }
        )
    penalty_rows: list[dict[str, object]] = []
    for keys, frame in penalties.groupby(
        ["relation", "moderator", "band", "band_index"], sort=True
    ):
        indexed = frame.set_index("study")
        london = indexed.loc["greater_london"]
        manchester = indexed.loc["greater_manchester"]
        comparison = independent_city_difference(
            float(london["penalty_difference"]),
            float(london["penalty_difference_block_bootstrap_se"]),
            int(london["clusters"]),
            float(manchester["penalty_difference"]),
            float(manchester["penalty_difference_block_bootstrap_se"]),
            int(manchester["clusters"]),
        )
        penalty_rows.append(
            {
                "outcome": "bad_health_pct",
                "relation": keys[0],
                "moderator": keys[1],
                "band": keys[2],
                "band_index": keys[3],
                "london_penalty_difference": float(
                    london["penalty_difference"]
                ),
                "manchester_penalty_difference": float(
                    manchester["penalty_difference"]
                ),
                **comparison,
            }
        )
    interactions = pd.DataFrame(interaction_rows)
    penalty_comparisons = pd.DataFrame(penalty_rows)
    interactions["difference_q"] = bh_fdr(
        interactions["difference_p"].to_numpy(float)
    )
    penalty_comparisons["difference_q"] = bh_fdr(
        penalty_comparisons["difference_p"].to_numpy(float)
    )
    return interactions, penalty_comparisons


def save_primary_figure(curves: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    selected = curves[
        (curves["moderator"] == "no_car_pct")
        & (curves["relation"].isin(("nhs_gp", "osm_pharmacy")))
    ]
    colors = {
        "lower vulnerability": "#2b6cb0",
        "higher vulnerability": "#c53030",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), layout="constrained")
    for row_index, study in enumerate(STUDIES):
        for column_index, relation in enumerate(("nhs_gp", "osm_pharmacy")):
            axis = axes[row_index, column_index]
            panel = selected[
                (selected["study"] == study) & (selected["relation"] == relation)
            ]
            for stratum in ("lower vulnerability", "higher vulnerability"):
                line = panel[panel["stratum"] == stratum].sort_values(
                    "additional_minutes"
                )
                axis.fill_between(
                    line["additional_minutes"],
                    line["effect_block_bootstrap_lower"],
                    line["effect_block_bootstrap_upper"],
                    color=colors[stratum],
                    alpha=0.13,
                    linewidth=0,
                )
                axis.plot(
                    line["additional_minutes"],
                    line["effect"],
                    color=colors[stratum],
                    marker="o",
                    linewidth=2.0,
                    label=stratum,
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.grid(color="#eeeeee", linewidth=0.7)
            axis.set_title(
                f"{STUDY_LABELS[study]} | {RELATION_LABELS[relation]}", fontsize=11
            )
            if row_index == 1:
                axis.set_xlabel("Uniform increase in routed walking time (min)")
            if column_index == 0:
                axis.set_ylabel(
                    "Adjusted poor-health difference\n(percentage points)"
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "HGGE-derived primary-care penalties by neighbourhood car availability",
        fontsize=14,
        fontweight="bold",
    )
    pdf = output_dir / "figure_hgge_derived_no_car_primary_care.pdf"
    png = output_dir / "figure_hgge_derived_no_car_primary_care.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def save_green_figure(curves: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    selected = curves[curves["relation"] == "os_green"]
    colors = {
        "lower vulnerability": "#2b6cb0",
        "higher vulnerability": "#c53030",
    }
    moderators = ("no_car_pct", "older_65_pct")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), layout="constrained")
    for row_index, study in enumerate(STUDIES):
        for column_index, moderator in enumerate(moderators):
            axis = axes[row_index, column_index]
            panel = selected[
                (selected["study"] == study)
                & (selected["moderator"] == moderator)
            ]
            for stratum in ("lower vulnerability", "higher vulnerability"):
                line = panel[panel["stratum"] == stratum].sort_values(
                    "additional_minutes"
                )
                axis.fill_between(
                    line["additional_minutes"],
                    line["effect_block_bootstrap_lower"],
                    line["effect_block_bootstrap_upper"],
                    color=colors[stratum],
                    alpha=0.13,
                    linewidth=0,
                )
                axis.plot(
                    line["additional_minutes"],
                    line["effect"],
                    color=colors[stratum],
                    marker="o",
                    linewidth=2.0,
                    label=stratum,
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.grid(color="#eeeeee", linewidth=0.7)
            axis.set_title(
                f"{STUDY_LABELS[study]} | {MODERATOR_LABELS[moderator]}",
                fontsize=11,
            )
            if row_index == 1:
                axis.set_xlabel("Uniform increase in routed walking time (min)")
            if column_index == 0:
                axis.set_ylabel(
                    "Adjusted poor-health difference\n(percentage points)"
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "HGGE-derived green-space penalties by neighbourhood vulnerability",
        fontsize=14,
        fontweight="bold",
    )
    pdf = output_dir / "figure_hgge_derived_green_vulnerability.pdf"
    png = output_dir / "figure_hgge_derived_green_vulnerability.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    census = load_census_features(args.census_root)
    scaling_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    penalty_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    adjusted_rows: list[pd.DataFrame] = []
    city_benchmark_rows: list[dict[str, object]] = []

    for study_index, study in enumerate(args.studies):
        bundle = load_case(args.case_root, args.oof_dir, study)
        frame = bundle["frame"]
        codes = frame["lsoa21cd"].astype(str).to_numpy()
        joined = pd.DataFrame({"lsoa21cd": codes}).merge(
            census, on="lsoa21cd", how="left", validate="one_to_one"
        )
        moderators_raw = joined[list(MODERATORS)].to_numpy(float)
        moderators_z = (
            moderators_raw - moderators_raw.mean(axis=0)
        ) / moderators_raw.std(axis=0, ddof=0)
        folds = np.asarray(bundle["fold"], dtype=int)
        spatial_block = make_spatial_blocks(
            np.asarray(bundle["coordinates"]),
            args.spatial_blocks,
            args.seed + study_index,
        )
        _, edge_bundle = relation_edges(
            args.case_root / study / "facility_lsoa_graph.npz"
        )
        ahah = np.asarray(bundle["y"], dtype=float)
        health = joined["bad_health_pct"].to_numpy(float)
        population = joined["usual_residents"].to_numpy(float)
        city_benchmark_rows.append(
            {
                "study": study,
                "population": float(np.sum(population)),
                "population_weighted_bad_health_pct": float(
                    np.average(health, weights=population)
                ),
                "median_lsoa_population": float(np.median(population)),
            }
        )
        hgge = load_hgge_outputs(
            args.hgge_relation_dir,
            args.hgge_ensemble_dir,
            study,
            codes,
            folds,
        )
        hgge_residual = ahah - np.asarray(hgge["prediction"], dtype=float)
        relation_lookup = {
            relation: index
            for index, relation in enumerate(hgge["relation_names"])
        }

        for relation_index, relation in enumerate(args.relations):
            if relation not in relation_lookup:
                raise ValueError(f"HGGE output lacks relation {relation}")
            raw = np.asarray(hgge["access"], dtype=float)[
                relation_lookup[relation]
            ]
            shifts = np.asarray(hgge["shifts"], dtype=float)
            exposure, shifted, scaling = cross_fitted_standardize(
                raw, shifts, folds
            )
            for row in scaling:
                scaling_rows.append({"study": study, "relation": relation, **row})
            diagnostic_rows.append(
                {
                    "study": study,
                    "relation": relation,
                    "mean_raw_access_benefit": float(np.mean(raw[:, 0])),
                    "sd_raw_access_benefit": float(np.std(raw[:, 0], ddof=0)),
                    "share_positive_raw_access_benefit": float(
                        np.mean(raw[:, 0] > 0)
                    ),
                    "mean_raw_change_after_5min": float(
                        np.mean(raw[:, np.flatnonzero(shifts == 5.0)[0]] - raw[:, 0])
                    ),
                }
            )
            nearest = edge_bundle[relation]["nearest"]
            censored = edge_bundle[relation]["censored"]
            threshold_rows.extend(
                hgge_residual_threshold_tests(
                    study,
                    relation,
                    hgge_residual,
                    nearest,
                    censored,
                    spatial_block,
                    args.bootstrap_replicates,
                    args.seed + study_index * 100000 + relation_index * 1000,
                )
            )
            if relation not in PRIMARY_RELATIONS:
                continue
            context, _ = relation_context(bundle, relation)
            health_residual, health_context_prediction = fit_nuisance(
                context,
                health,
                folds,
                args.seed + study_index * 10000 + relation_index * 100 + 50,
                args.max_iterations,
            )
            for moderator_index, moderator in enumerate(MODERATORS):
                summary, penalties, curves = learned_access_interaction(
                    study,
                    relation,
                    moderator,
                    health_residual,
                    exposure,
                    shifted,
                    nearest,
                    censored,
                    moderators_raw,
                    moderators_z,
                    moderator_index,
                    spatial_block,
                    population,
                    args.bootstrap_replicates,
                    (
                        args.seed
                        + study_index * 100000
                        + relation_index * 1000
                        + moderator_index * 100
                        + 17
                    ),
                )
                summary_rows.append(summary)
                penalty_rows.extend(penalties)
                curve_rows.extend(curves)
                design = access_design(
                    exposure, moderators_z, moderator_index, censored
                )
                adjusted_rows.append(
                    pd.DataFrame(
                        {
                            "study": study,
                            "lsoa21cd": joined["lsoa21cd"],
                            "relation": relation,
                            "outcome": "bad_health_pct",
                            "moderator": moderator,
                            "observed": health,
                            "nuisance_prediction": health_context_prediction,
                            "adjusted_residual": health_residual,
                            "hgge_access_exposure_z": exposure,
                            "hgge_access_delta_5min": shifted[5.0] - exposure,
                            # Compatibility aliases for the shared bootstrap helper.
                            "gravity_exposure_z": exposure,
                            "gravity_delta_5min": shifted[5.0] - exposure,
                            "nearest_time": nearest,
                            "censored": censored,
                            "spatial_block": spatial_block,
                            **{
                                name: joined[name].to_numpy(float)
                                for name in MODERATORS
                            },
                            **{
                                f"design_{index}": design[:, index]
                                for index in range(design.shape[1])
                            },
                        }
                    )
                )
            print(
                json.dumps(
                    {"event": "relation_complete", "study": study, "relation": relation}
                ),
                flush=True,
            )

    scaling_frame = pd.DataFrame(scaling_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summaries = pd.DataFrame(summary_rows)
    penalties = pd.DataFrame(penalty_rows)
    curves = pd.DataFrame(curve_rows)
    thresholds = pd.DataFrame(threshold_rows)
    adjusted = pd.concat(adjusted_rows, ignore_index=True)
    city_benchmarks = pd.DataFrame(city_benchmark_rows)
    city_health_gap = float(
        abs(
            city_benchmarks.set_index("study").loc[
                "greater_manchester", "population_weighted_bad_health_pct"
            ]
            - city_benchmarks.set_index("study").loc[
                "greater_london", "population_weighted_bad_health_pct"
            ]
        )
    )
    for column in ("low_penalty", "high_penalty", "penalty_difference"):
        penalties[f"{column}_as_between_city_health_gap_pct"] = (
            100.0 * penalties[column] / city_health_gap
        )
    summaries["hgge_access_interaction_q"] = bh_fdr(
        summaries["hgge_access_interaction_block_bootstrap_p"].to_numpy(float)
    )
    penalties["penalty_difference_q"] = bh_fdr(
        penalties["penalty_difference_block_bootstrap_p"].to_numpy(float)
    )
    thresholds["step_q"] = bh_fdr(
        thresholds["step_block_bootstrap_p"].to_numpy(float)
    )
    thresholds["post_threshold_deviation_q"] = bh_fdr(
        thresholds["post_threshold_deviation_block_bootstrap_p"].to_numpy(float)
    )
    threshold_power = (
        thresholds.groupby(["study", "relation"], as_index=False)
        .agg(
            median_step_mde80_unadjusted=(
                "step_mde80_unadjusted",
                "median",
            ),
            minimum_step_mde80_unadjusted=(
                "step_mde80_unadjusted",
                "min",
            ),
            maximum_step_mde80_unadjusted=(
                "step_mde80_unadjusted",
                "max",
            ),
            median_step_mde80_bonferroni96=(
                "step_mde80_bonferroni96",
                "median",
            ),
            median_step_mde80_unadjusted_residual_sd=(
                "step_mde80_unadjusted_residual_sd",
                "median",
            ),
        )
    )
    threshold_power["primary_care_relation"] = threshold_power["relation"].isin(
        PRIMARY_RELATIONS
    )
    city_interactions, city_penalties = compare_cities(summaries, penalties)
    bootstrap, city_bootstrap = block_bootstrap_primary(
        adjusted, args.bootstrap_replicates, args.seed
    )
    bootstrap["note"] = (
        "spatial-block bootstrap conditional on strict-OOF HGGE relation "
        "counterfactuals and health nuisance predictions"
    )
    city_bootstrap["note"] = (
        "independent spatial-block resampling within each city; conditional on "
        "strict-OOF HGGE relation counterfactuals and health nuisance predictions"
    )

    scaling_frame.to_csv(args.output_dir / "hgge_access_scaling.csv", index=False)
    diagnostics.to_csv(
        args.output_dir / "hgge_relation_access_diagnostics.csv", index=False
    )
    summaries.to_csv(
        args.output_dir / "hgge_access_interaction_summary.csv", index=False
    )
    penalties.to_csv(args.output_dir / "hgge_delay_penalties.csv", index=False)
    curves.to_csv(args.output_dir / "hgge_delay_curves.csv", index=False)
    thresholds.to_csv(
        args.output_dir / "hgge_residual_threshold_tests.csv", index=False
    )
    threshold_power.to_csv(
        args.output_dir / "hgge_threshold_power_summary.csv", index=False
    )
    city_benchmarks.to_csv(
        args.output_dir / "health_effect_size_benchmarks.csv", index=False
    )
    adjusted.to_parquet(args.output_dir / "adjusted_hgge_oof.parquet", index=False)
    bootstrap.to_csv(
        args.output_dir / "primary_no_car_block_bootstrap.csv", index=False
    )
    city_interactions.to_csv(
        args.output_dir / "between_city_hgge_interactions.csv", index=False
    )
    city_penalties.to_csv(
        args.output_dir / "between_city_hgge_penalties.csv", index=False
    )
    city_bootstrap.to_csv(
        args.output_dir / "primary_no_car_between_city_bootstrap.csv", index=False
    )
    figures = {
        "primary_care_carlessness": save_primary_figure(curves, args.output_dir),
        "green_vulnerability": save_green_figure(curves, args.output_dir),
    }

    payload = {
        "design": {
            "threshold_stage": (
                "candidate fixed steps and slope deviations are tested against "
                "strict-OOF residuals from the full 50:50 HGGE prediction"
            ),
            "health_exposure": (
                "outer-fold HGGE graph prediction at a 40-minute within-relation "
                "reference minus prediction after 0/5/10/15/20-minute edge-time "
                "shifts, clipped at 40 minutes and cross-fitted-standardised"
            ),
            "health_independence": (
                "Census poor-health values do not train HGGE or construct its "
                "relation counterfactual exposure"
            ),
            "gravity_projection_role": (
                "the separately selected scalar gravity projection is retained "
                "only as a sensitivity analysis"
            ),
            "inference": (
                f"{args.spatial_blocks}-block spatial-cluster covariance, "
                f"{args.bootstrap_replicates}-replicate spatial-block bootstrap "
                "for every threshold and health-interaction coefficient, and "
                "BH-FDR based on the block-bootstrap probabilities within the "
                "pre-specified result families"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "threshold_mde": (
                "noncentral-t 80/90% minimum detectable effects using spatial-"
                "block-bootstrap standard errors; Bonferroni-96 values are a "
                "conservative multiplicity benchmark for the BH-FDR family"
            ),
        },
        "counts": {
            "hgge_access_interactions": int(len(summaries)),
            "health_interactions_fdr": int(
                (summaries["hgge_access_interaction_q"] <= 0.05).sum()
            ),
            "threshold_steps_fdr": int((thresholds["step_q"] <= 0.05).sum()),
            "between_city_interactions_fdr": int(
                (city_interactions["difference_q"] <= 0.05).sum()
            ),
            "between_city_penalties_fdr": int(
                (city_penalties["difference_q"] <= 0.05).sum()
            ),
        },
        "effect_size_benchmark": {
            "population_weighted_between_city_bad_health_gap_percentage_points": (
                city_health_gap
            ),
            "translation": (
                "five-minute penalty estimates are also expressed as residents "
                "per median LSOA and as a percentage of the between-city health gap"
            ),
        },
        "outputs": {"figures": figures, "directory": str(args.output_dir.resolve())},
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **payload["counts"]}), flush=True)


if __name__ == "__main__":
    main()
