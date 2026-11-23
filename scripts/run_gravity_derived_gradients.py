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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
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


SCALES = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
BETAS = (1.0, 1.5, 2.0)
ALPHA = 0.15
OUTCOMES = ("bad_health_pct", "ahah")
OUTCOME_LABELS = {
    "bad_health_pct": "Poor self-rated health prevalence",
    "ahah": "AHAH index",
}
TIME_BANDS = (
    ("0--10 min", 0.0, 10.0),
    ("10--20 min", 10.0, 20.0),
    (">20 min", 20.0, np.inf),
)
SHIFT_GRID = np.arange(0.0, 20.0 + 1e-9, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fitted gravity-derived distance gradients and vulnerability "
            "interactions."
        )
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument(
        "--census-root",
        type=Path,
        default=Path("data/raw/census2021_vulnerability_2026_08_28"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gravity_derived_gradients"),
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


def hgb(seed: int, max_iterations: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=max_iterations,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=seed,
    )


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = y - prediction
    total = np.sum((y - y.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / max(float(total), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def relation_edges(graph_path: Path) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    payload = np.load(graph_path, allow_pickle=False)
    n_lsoa = len(payload["lsoa_codes"])
    relations = payload["relation_names"].astype(str).tolist()[:-1]
    source, destination = payload["edge_index"]
    edge_type = payload["edge_type"]
    impedance = payload["edge_impedance"].astype(float)
    node_mass = payload["node_mass"].astype(float)
    output: dict[str, dict[str, np.ndarray]] = {}
    for relation_index, relation in enumerate(relations):
        selected = (edge_type == relation_index) & (destination < n_lsoa)
        relation_destination = destination[selected].astype(int)
        relation_time = impedance[selected]
        relation_mass = node_mass[source[selected]]
        nearest = np.full(n_lsoa, np.inf, dtype=float)
        np.minimum.at(nearest, relation_destination, relation_time)
        censored = ~np.isfinite(nearest)
        nearest[censored] = 40.0
        output[relation] = {
            "destination": relation_destination,
            "time": relation_time,
            "mass": relation_mass,
            "nearest": nearest,
            "censored": censored.astype(float),
        }
    return n_lsoa, output


def gravity_potential(
    n_lsoa: int,
    edges: dict[str, np.ndarray],
    scale: float,
    beta: float,
    shift: float = 0.0,
) -> np.ndarray:
    time = edges["time"] + shift
    mass = edges["mass"]
    weight = np.minimum(
        1.0,
        np.power(mass, ALPHA) * np.exp(-np.power(time / scale, beta)),
    )
    strength = np.bincount(
        edges["destination"], weights=weight, minlength=n_lsoa
    ).astype(float)
    return np.log1p(strength)


def inner_context_residuals(
    context: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    seed: int,
    max_iterations: int,
) -> np.ndarray:
    residual = np.full(len(y), np.nan, dtype=float)
    inner_folds = [int(value) for value in np.unique(folds) if value != outer_fold]
    for inner_fold in inner_folds:
        train = (folds != outer_fold) & (folds != inner_fold)
        validation = folds == inner_fold
        model = hgb(seed + outer_fold * 100 + inner_fold, max_iterations)
        model.fit(context[train], y[train])
        residual[validation] = y[validation] - model.predict(context[validation])
    if not np.isfinite(residual[folds != outer_fold]).all():
        raise RuntimeError(f"incomplete inner residuals for outer fold {outer_fold}")
    return residual


def candidate_cv_rmse(
    potential: np.ndarray,
    residual: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
) -> float:
    squared_error = 0.0
    observations = 0
    for inner_fold in sorted(int(value) for value in np.unique(folds)):
        if inner_fold == outer_fold:
            continue
        train = (folds != outer_fold) & (folds != inner_fold)
        validation = folds == inner_fold
        design_train = np.column_stack((np.ones(train.sum()), potential[train]))
        beta = np.linalg.pinv(design_train, rcond=1e-10) @ residual[train]
        prediction = beta[0] + beta[1] * potential[validation]
        error = residual[validation] - prediction
        squared_error += float(np.sum(error**2))
        observations += int(validation.sum())
    return float(np.sqrt(squared_error / observations))


def select_cross_fitted_gravity(
    n_lsoa: int,
    edges: dict[str, np.ndarray],
    context: np.ndarray,
    ahah: np.ndarray,
    folds: np.ndarray,
    context_oof_prediction: np.ndarray,
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray, list[dict[str, object]]]:
    candidates = [(scale, beta) for scale in SCALES for beta in BETAS]
    potential = {
        candidate: gravity_potential(n_lsoa, edges, *candidate)
        for candidate in candidates
    }
    exposure = np.full(n_lsoa, np.nan, dtype=float)
    shifted = {
        float(shift): np.full(n_lsoa, np.nan, dtype=float) for shift in SHIFT_GRID
    }
    projection_prediction = np.full(n_lsoa, np.nan, dtype=float)
    selection_rows: list[dict[str, object]] = []
    for outer_fold in sorted(int(value) for value in np.unique(folds)):
        outer_train = folds != outer_fold
        outer_test = folds == outer_fold
        inner_residual = inner_context_residuals(
            context,
            ahah,
            folds,
            outer_fold,
            seed,
            max_iterations,
        )
        scores = {
            candidate: candidate_cv_rmse(
                potential[candidate], inner_residual, folds, outer_fold
            )
            for candidate in candidates
        }
        selected_scale, selected_beta = min(
            candidates,
            key=lambda candidate: (
                scores[candidate],
                abs(candidate[1] - 1.5),
                candidate[0],
            ),
        )
        selected = potential[(selected_scale, selected_beta)]
        mean = float(np.mean(selected[outer_train]))
        standard_deviation = float(np.std(selected[outer_train], ddof=0))
        if standard_deviation <= 1e-10:
            raise RuntimeError("degenerate selected gravity potential")
        selected_z = (selected - mean) / standard_deviation
        exposure[outer_test] = selected_z[outer_test]
        inner_design = np.column_stack(
            (np.ones(outer_train.sum()), selected_z[outer_train])
        )
        coefficient = (
            np.linalg.pinv(inner_design, rcond=1e-10)
            @ inner_residual[outer_train]
        )
        projection_prediction[outer_test] = (
            context_oof_prediction[outer_test]
            + coefficient[0]
            + coefficient[1] * selected_z[outer_test]
        )
        for shift in SHIFT_GRID:
            shifted_raw = gravity_potential(
                n_lsoa,
                edges,
                selected_scale,
                selected_beta,
                float(shift),
            )
            shifted[float(shift)][outer_test] = (
                shifted_raw[outer_test] - mean
            ) / standard_deviation
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "scale": selected_scale,
                "beta": selected_beta,
                "alpha": ALPHA,
                "inner_rmse": scores[(selected_scale, selected_beta)],
                "gravity_coefficient": float(coefficient[1]),
                "training_potential_mean": mean,
                "training_potential_sd": standard_deviation,
            }
        )
    if not np.isfinite(exposure).all() or not np.isfinite(projection_prediction).all():
        raise RuntimeError("incomplete cross-fitted gravity output")
    if any(not np.isfinite(values).all() for values in shifted.values()):
        raise RuntimeError("incomplete shifted gravity output")
    return exposure, shifted, projection_prediction, selection_rows


def gravity_design(
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


def contrast(
    beta: np.ndarray,
    covariance: np.ndarray,
    vector: np.ndarray,
    clusters: int,
) -> dict[str, float]:
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    critical = float(stats.t.ppf(0.975, df=clusters - 1))
    p_value = (
        float(2.0 * stats.t.sf(abs(estimate / standard_error), df=clusters - 1))
        if standard_error > 0
        else np.nan
    )
    return {
        "estimate": estimate,
        "se": standard_error,
        "lower": estimate - critical * standard_error,
        "upper": estimate + critical * standard_error,
        "p": p_value,
    }


def gravity_interaction(
    study: str,
    relation: str,
    outcome: str,
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
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    design = gravity_design(exposure, moderators_z, moderator_index, censored)
    inference = cluster_robust_inference(design, residual, spatial_block)
    beta = inference["beta"]
    covariance = inference["covariance"]
    clusters = len(np.unique(spatial_block))
    interaction_vector = np.zeros(len(beta))
    interaction_vector[5] = 1.0
    interaction = contrast(beta, covariance, interaction_vector, clusters)

    raw_moderator = moderators_raw[:, moderator_index]
    raw_q25, raw_q75 = np.quantile(raw_moderator, [0.25, 0.75])
    raw_mean = float(np.mean(raw_moderator))
    raw_sd = float(np.std(raw_moderator, ddof=0))
    z25 = float((raw_q25 - raw_mean) / raw_sd)
    z75 = float((raw_q75 - raw_mean) / raw_sd)
    connected = censored == 0

    penalty_rows: list[dict[str, object]] = []
    delta_five = shifted[5.0] - exposure
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
        penalty_rows.append(
            {
                "study": study,
                "outcome": outcome,
                "relation": relation,
                "moderator": moderator,
                "band": band_label,
                "band_index": band_index,
                "n_band": int(selected.sum()),
                "clusters": clusters,
                "mean_gravity_change_for_5min_delay": mean_delta,
                "low_penalty": low_effect["estimate"],
                "low_penalty_lower": low_effect["lower"],
                "low_penalty_upper": low_effect["upper"],
                "high_penalty": high_effect["estimate"],
                "high_penalty_lower": high_effect["lower"],
                "high_penalty_upper": high_effect["upper"],
                "penalty_difference": difference["estimate"],
                "penalty_difference_se": difference["se"],
                "penalty_difference_lower": difference["lower"],
                "penalty_difference_upper": difference["upper"],
                "penalty_difference_p": difference["p"],
            }
        )

    curve_rows: list[dict[str, object]] = []
    for shift, shifted_exposure in shifted.items():
        mean_delta = float(np.mean((shifted_exposure - exposure)[connected]))
        for stratum, z_value in (("lower vulnerability", z25), ("higher vulnerability", z75)):
            vector = np.zeros(len(beta))
            vector[1] = mean_delta
            vector[5] = mean_delta * z_value
            effect = contrast(beta, covariance, vector, clusters)
            curve_rows.append(
                {
                    "study": study,
                    "outcome": outcome,
                    "relation": relation,
                    "moderator": moderator,
                    "stratum": stratum,
                    "additional_minutes": shift,
                    "effect": effect["estimate"],
                    "effect_lower": effect["lower"],
                    "effect_upper": effect["upper"],
                }
            )
    summary = {
        "study": study,
        "outcome": outcome,
        "relation": relation,
        "relation_label": RELATION_LABELS[relation],
        "moderator": moderator,
        "moderator_label": MODERATOR_LABELS[moderator],
        "n": len(residual),
        "clusters": clusters,
        "moderator_q25": float(raw_q25),
        "moderator_q75": float(raw_q75),
        "gravity_main_effect": float(beta[1]),
        "gravity_interaction": interaction["estimate"],
        "gravity_interaction_se": interaction["se"],
        "gravity_interaction_lower": interaction["lower"],
        "gravity_interaction_upper": interaction["upper"],
        "gravity_interaction_p": interaction["p"],
    }
    return summary, penalty_rows, curve_rows


def gravity_threshold_tests(
    study: str,
    relation: str,
    residual: np.ndarray,
    exposure: np.ndarray,
    nearest: np.ndarray,
    censored: np.ndarray,
    spatial_block: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clusters = len(np.unique(spatial_block))
    for threshold in FIXED_THRESHOLDS:
        design = np.column_stack(
            (
                np.ones(len(residual)),
                exposure,
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
                "post_threshold_deviation_per_5min": hinge["estimate"],
                "post_threshold_deviation_se": hinge["se"],
                "post_threshold_deviation_lower": hinge["lower"],
                "post_threshold_deviation_upper": hinge["upper"],
                "post_threshold_deviation_p": hinge["p"],
            }
        )
    return rows


def apply_fdr(
    summaries: pd.DataFrame, penalties: pd.DataFrame, thresholds: pd.DataFrame
) -> None:
    summaries["gravity_interaction_q"] = np.nan
    penalties["penalty_difference_q"] = np.nan
    for outcome in summaries["outcome"].unique():
        selected = summaries["outcome"] == outcome
        summaries.loc[selected, "gravity_interaction_q"] = bh_fdr(
            summaries.loc[selected, "gravity_interaction_p"].to_numpy(float)
        )
        selected_penalty = penalties["outcome"] == outcome
        penalties.loc[selected_penalty, "penalty_difference_q"] = bh_fdr(
            penalties.loc[selected_penalty, "penalty_difference_p"].to_numpy(float)
        )
    thresholds["step_q"] = bh_fdr(thresholds["step_p"].to_numpy(float))
    thresholds["post_threshold_deviation_q"] = bh_fdr(
        thresholds["post_threshold_deviation_p"].to_numpy(float)
    )


def independent_city_difference(
    london_estimate: float,
    london_se: float,
    london_clusters: int,
    manchester_estimate: float,
    manchester_se: float,
    manchester_clusters: int,
) -> dict[str, float]:
    difference = manchester_estimate - london_estimate
    london_variance = london_se**2
    manchester_variance = manchester_se**2
    variance = london_variance + manchester_variance
    standard_error = float(np.sqrt(variance))
    degrees_of_freedom = float(
        variance**2
        / (
            london_variance**2 / max(london_clusters - 1, 1)
            + manchester_variance**2 / max(manchester_clusters - 1, 1)
        )
    )
    critical = float(stats.t.ppf(0.975, df=degrees_of_freedom))
    p_value = (
        float(2.0 * stats.t.sf(abs(difference / standard_error), df=degrees_of_freedom))
        if standard_error > 0
        else np.nan
    )
    return {
        "manchester_minus_london": float(difference),
        "difference_se": standard_error,
        "difference_lower": float(difference - critical * standard_error),
        "difference_upper": float(difference + critical * standard_error),
        "difference_df": degrees_of_freedom,
        "difference_p": p_value,
    }


def compare_cities(
    summaries: pd.DataFrame, penalties: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {"greater_london", "greater_manchester"}
    if set(summaries["study"].unique()) != expected:
        raise ValueError("city comparison requires Greater London and Greater Manchester")

    interaction_rows: list[dict[str, object]] = []
    for keys, frame in summaries.groupby(
        ["outcome", "relation", "moderator"], sort=True
    ):
        indexed = frame.set_index("study")
        london = indexed.loc["greater_london"]
        manchester = indexed.loc["greater_manchester"]
        comparison = independent_city_difference(
            float(london["gravity_interaction"]),
            float(london["gravity_interaction_se"]),
            int(london["clusters"]),
            float(manchester["gravity_interaction"]),
            float(manchester["gravity_interaction_se"]),
            int(manchester["clusters"]),
        )
        interaction_rows.append(
            {
                "outcome": keys[0],
                "relation": keys[1],
                "moderator": keys[2],
                "london_interaction": float(london["gravity_interaction"]),
                "manchester_interaction": float(manchester["gravity_interaction"]),
                **comparison,
            }
        )

    penalty_rows: list[dict[str, object]] = []
    for keys, frame in penalties.groupby(
        ["outcome", "relation", "moderator", "band", "band_index"], sort=True
    ):
        indexed = frame.set_index("study")
        london = indexed.loc["greater_london"]
        manchester = indexed.loc["greater_manchester"]
        comparison = independent_city_difference(
            float(london["penalty_difference"]),
            float(london["penalty_difference_se"]),
            int(london["clusters"]),
            float(manchester["penalty_difference"]),
            float(manchester["penalty_difference_se"]),
            int(manchester["clusters"]),
        )
        penalty_rows.append(
            {
                "outcome": keys[0],
                "relation": keys[1],
                "moderator": keys[2],
                "band": keys[3],
                "band_index": keys[4],
                "london_penalty_difference": float(london["penalty_difference"]),
                "manchester_penalty_difference": float(
                    manchester["penalty_difference"]
                ),
                **comparison,
            }
        )

    interaction_comparisons = pd.DataFrame(interaction_rows)
    penalty_comparisons = pd.DataFrame(penalty_rows)
    interaction_comparisons["difference_q"] = np.nan
    penalty_comparisons["difference_q"] = np.nan
    for outcome in OUTCOMES:
        selected = interaction_comparisons["outcome"] == outcome
        interaction_comparisons.loc[selected, "difference_q"] = bh_fdr(
            interaction_comparisons.loc[selected, "difference_p"].to_numpy(float)
        )
        selected = penalty_comparisons["outcome"] == outcome
        penalty_comparisons.loc[selected, "difference_q"] = bh_fdr(
            penalty_comparisons.loc[selected, "difference_p"].to_numpy(float)
        )
    return interaction_comparisons, penalty_comparisons


def block_bootstrap_primary(
    adjusted: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    estimate_draws: dict[tuple[str, str], np.ndarray] = {}
    targets = adjusted[
        (adjusted["outcome"] == "bad_health_pct")
        & (adjusted["moderator"] == "no_car_pct")
        & (adjusted["relation"].isin(("nhs_gp", "osm_pharmacy")))
    ]
    for target_index, ((study, relation), frame) in enumerate(
        targets.groupby(["study", "relation"], sort=True)
    ):
        frame = frame.reset_index(drop=True)
        x = frame[[f"design_{index}" for index in range(7)]].to_numpy(float)
        y = frame["adjusted_residual"].to_numpy(float)
        block = frame["spatial_block"].to_numpy(int)
        nearest = frame["nearest_time"].to_numpy(float)
        censored = frame["censored"].to_numpy(float)
        delta = frame["gravity_delta_5min"].to_numpy(float)
        raw = frame["no_car_pct"].to_numpy(float)
        z_delta = float(
            (np.quantile(raw, 0.75) - np.quantile(raw, 0.25))
            / np.std(raw, ddof=0)
        )
        unique_blocks = np.unique(block)
        rng = np.random.default_rng(seed + target_index * 100)
        estimates = np.empty(replicates)
        band = (censored == 0) & (nearest > 10.0) & (nearest <= 20.0)
        for replicate in range(replicates):
            sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
            block_counts = {value: int(np.sum(sampled == value)) for value in unique_blocks}
            weights = np.asarray([block_counts[value] for value in block], dtype=float)
            beta = np.linalg.pinv(x.T @ (weights[:, None] * x), rcond=1e-10) @ (
                x.T @ (weights * y)
            )
            band_weights = weights[band]
            mean_delta = float(np.average(delta[band], weights=band_weights))
            estimates[replicate] = beta[5] * z_delta * mean_delta
        estimate_draws[(study, relation)] = estimates
        rows.append(
            {
                "study": study,
                "relation": relation,
                "moderator": "no_car_pct",
                "band": "10--20 min",
                "replicates": replicates,
                "bootstrap_median": float(np.median(estimates)),
                "bootstrap_lower": float(np.quantile(estimates, 0.025)),
                "bootstrap_upper": float(np.quantile(estimates, 0.975)),
                "bootstrap_probability_positive": float(np.mean(estimates > 0)),
                "note": "spatial-block bootstrap conditional on cross-fitted kernels and nuisance predictions",
            }
        )
    difference_rows: list[dict[str, object]] = []
    for relation in ("nhs_gp", "osm_pharmacy"):
        difference = (
            estimate_draws[("greater_manchester", relation)]
            - estimate_draws[("greater_london", relation)]
        )
        difference_rows.append(
            {
                "relation": relation,
                "moderator": "no_car_pct",
                "band": "10--20 min",
                "comparison": "greater_manchester_minus_greater_london",
                "replicates": replicates,
                "bootstrap_median": float(np.median(difference)),
                "bootstrap_lower": float(np.quantile(difference, 0.025)),
                "bootstrap_upper": float(np.quantile(difference, 0.975)),
                "bootstrap_probability_positive": float(np.mean(difference > 0)),
                "note": "independent spatial-block resampling within each city; conditional on cross-fitted kernels and nuisance predictions",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(difference_rows)


def save_primary_figure(curves: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    selected = curves[
        (curves["outcome"] == "bad_health_pct")
        & (curves["moderator"] == "no_car_pct")
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
                line = panel[panel["stratum"] == stratum]
                axis.fill_between(
                    line["additional_minutes"],
                    line["effect_lower"],
                    line["effect_upper"],
                    color=colors[stratum],
                    alpha=0.13,
                    linewidth=0,
                )
                axis.plot(
                    line["additional_minutes"],
                    line["effect"],
                    color=colors[stratum],
                    linewidth=2.0,
                    label=stratum,
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.grid(color="#eeeeee", linewidth=0.7)
            axis.set_title(
                f"{STUDY_LABELS[study]} | {RELATION_LABELS[relation]}",
                fontsize=11,
            )
            if row_index == 1:
                axis.set_xlabel("Uniform increase in routed walking time (min)")
            if column_index == 0:
                axis.set_ylabel("Adjusted poor-health difference\n(percentage points)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Gravity-derived primary-care penalties by neighbourhood car availability",
        fontsize=14,
        fontweight="bold",
    )
    pdf = output_dir / "figure_gravity_derived_no_car_primary_care.pdf"
    png = output_dir / "figure_gravity_derived_no_car_primary_care.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    census = load_census_features(args.census_root)
    kernel_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    penalty_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    adjusted_rows: list[pd.DataFrame] = []

    for study_index, study in enumerate(args.studies):
        bundle = load_case(args.case_root, args.oof_dir, study)
        frame = bundle["frame"]
        joined = pd.DataFrame(
            {"lsoa21cd": frame["lsoa21cd"].astype(str).to_numpy()}
        ).merge(census, on="lsoa21cd", how="left", validate="one_to_one")
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
        n_lsoa, edge_bundle = relation_edges(
            args.case_root / study / "facility_lsoa_graph.npz"
        )
        ahah = np.asarray(bundle["y"], dtype=float)
        health = joined["bad_health_pct"].to_numpy(float)

        for relation_index, relation in enumerate(args.relations):
            context, _ = relation_context(bundle, relation)
            ahah_residual, ahah_context_prediction = fit_nuisance(
                context,
                ahah,
                folds,
                args.seed + study_index * 10000 + relation_index * 100,
                args.max_iterations,
            )
            exposure, shifted, projection_prediction, selections = (
                select_cross_fitted_gravity(
                    n_lsoa,
                    edge_bundle[relation],
                    context,
                    ahah,
                    folds,
                    ahah_context_prediction,
                    args.seed + study_index * 10000 + relation_index * 100,
                    args.max_iterations,
                )
            )
            for selection in selections:
                kernel_rows.append(
                    {"study": study, "relation": relation, **selection}
                )
            for fold in sorted(int(value) for value in np.unique(folds)):
                test = folds == fold
                performance_rows.append(
                    {
                        "study": study,
                        "relation": relation,
                        "fold": fold,
                        **{
                            f"context_{name}": value
                            for name, value in metrics(
                                ahah[test], ahah_context_prediction[test]
                            ).items()
                        },
                        **{
                            f"gravity_{name}": value
                            for name, value in metrics(
                                ahah[test], projection_prediction[test]
                            ).items()
                        },
                    }
                )
            nearest = edge_bundle[relation]["nearest"]
            censored = edge_bundle[relation]["censored"]
            threshold_rows.extend(
                gravity_threshold_tests(
                    study,
                    relation,
                    ahah_residual,
                    exposure,
                    nearest,
                    censored,
                    spatial_block,
                )
            )
            if relation not in PRIMARY_RELATIONS:
                print(
                    json.dumps(
                        {
                            "event": "relation_complete",
                            "study": study,
                            "relation": relation,
                            "vulnerability": False,
                        }
                    ),
                    flush=True,
                )
                continue
            health_residual, health_context_prediction = fit_nuisance(
                context,
                health,
                folds,
                args.seed + study_index * 10000 + relation_index * 100 + 50,
                args.max_iterations,
            )
            outcomes = {
                "bad_health_pct": (health, health_residual, health_context_prediction),
                "ahah": (ahah, ahah_residual, ahah_context_prediction),
            }
            for outcome, (observed, residual, nuisance_prediction) in outcomes.items():
                for moderator_index, moderator in enumerate(MODERATORS):
                    summary, penalties, curves = gravity_interaction(
                        study,
                        relation,
                        outcome,
                        moderator,
                        residual,
                        exposure,
                        shifted,
                        nearest,
                        censored,
                        moderators_raw,
                        moderators_z,
                        moderator_index,
                        spatial_block,
                    )
                    summary_rows.append(summary)
                    penalty_rows.extend(penalties)
                    curve_rows.extend(curves)
                    design = gravity_design(
                        exposure, moderators_z, moderator_index, censored
                    )
                    adjusted_rows.append(
                        pd.DataFrame(
                            {
                                "study": study,
                                "lsoa21cd": joined["lsoa21cd"],
                                "relation": relation,
                                "outcome": outcome,
                                "moderator": moderator,
                                "observed": observed,
                                "nuisance_prediction": nuisance_prediction,
                                "adjusted_residual": residual,
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
                    {
                        "event": "relation_complete",
                        "study": study,
                        "relation": relation,
                        "vulnerability": True,
                    }
                ),
                flush=True,
            )

    kernels = pd.DataFrame(kernel_rows)
    performance = pd.DataFrame(performance_rows)
    summaries = pd.DataFrame(summary_rows)
    penalties = pd.DataFrame(penalty_rows)
    curves = pd.DataFrame(curve_rows)
    thresholds = pd.DataFrame(threshold_rows)
    adjusted = pd.concat(adjusted_rows, ignore_index=True)
    apply_fdr(summaries, penalties, thresholds)
    city_interactions, city_penalties = compare_cities(summaries, penalties)
    bootstrap, city_bootstrap = block_bootstrap_primary(
        adjusted, args.bootstrap_replicates, args.seed
    )

    kernels.to_csv(args.output_dir / "cross_fitted_kernel_selection.csv", index=False)
    performance.to_csv(
        args.output_dir / "gravity_projection_performance_by_fold.csv", index=False
    )
    summaries.to_csv(args.output_dir / "gravity_interaction_summary.csv", index=False)
    penalties.to_csv(args.output_dir / "gravity_delay_penalties.csv", index=False)
    curves.to_csv(args.output_dir / "gravity_delay_curves.csv", index=False)
    thresholds.to_csv(
        args.output_dir / "gravity_adjusted_threshold_tests.csv", index=False
    )
    adjusted.to_parquet(args.output_dir / "adjusted_gravity_oof.parquet", index=False)
    bootstrap.to_csv(
        args.output_dir / "primary_no_car_block_bootstrap.csv", index=False
    )
    city_interactions.to_csv(
        args.output_dir / "between_city_gravity_interactions.csv", index=False
    )
    city_penalties.to_csv(
        args.output_dir / "between_city_gravity_penalties.csv", index=False
    )
    city_bootstrap.to_csv(
        args.output_dir / "primary_no_car_between_city_bootstrap.csv", index=False
    )
    figure = save_primary_figure(curves, args.output_dir)

    payload = {
        "design": {
            "gravity_potential": "log1p(sum min(1, q^alpha exp(-(t/scale)^beta)))",
            "alpha": ALPHA,
            "candidate_scales": list(SCALES),
            "candidate_betas": list(BETAS),
            "kernel_selection": (
                "within each outer spatial fold, select scale and beta on the four "
                "remaining folds using incremental prediction of cross-fitted AHAH "
                "context residuals"
            ),
            "health_independence": (
                "kernel selection uses AHAH only; Census poor-health values do not "
                "enter kernel selection"
            ),
            "gradient": (
                "outcome change induced by adding walking minutes to every retained "
                "facility edge and recomputing the cross-fitted gravity potential"
            ),
            "inference": (
                "25-block spatial-cluster covariance and BH-FDR within outcome families"
            ),
            "city_comparison": (
                "direct Greater Manchester-minus-Greater London contrasts with "
                "independent-city standard errors and outcome-specific BH-FDR"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "counts": {
            "kernel_fits": int(len(kernels)),
            "gravity_interactions": int(len(summaries)),
            "primary_interactions_fdr": int(
                (
                    (summaries["outcome"] == "bad_health_pct")
                    & (summaries["gravity_interaction_q"] <= 0.05)
                ).sum()
            ),
            "secondary_interactions_fdr": int(
                (
                    (summaries["outcome"] == "ahah")
                    & (summaries["gravity_interaction_q"] <= 0.05)
                ).sum()
            ),
            "threshold_steps_fdr": int((thresholds["step_q"] <= 0.05).sum()),
            "between_city_interactions_fdr": int(
                (city_interactions["difference_q"] <= 0.05).sum()
            ),
            "between_city_penalties_fdr": int(
                (city_penalties["difference_q"] <= 0.05).sum()
            ),
        },
        "outputs": {
            "figure": figure,
            "directory": str(args.output_dir.resolve()),
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **payload["counts"]}), flush=True)


if __name__ == "__main__":
    main()
