from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDIES = ("greater_london", "greater_manchester")
STUDY_LABELS = {
    "greater_london": "Greater London",
    "greater_manchester": "Greater Manchester",
}
RELATION_LABELS = {
    "nhs_gp": "GP practices",
    "nhs_dentist": "Dentists",
    "osm_hospital": "Hospitals",
    "osm_pharmacy": "Pharmacies",
    "osm_leisure": "Leisure facilities",
    "os_green": "General green space",
    "os_active_green": "Active green space",
    "osm_fast_food": "Fast-food outlets",
    "osm_gambling": "Gambling premises",
    "osm_pub_bar": "Pubs and bars",
    "osm_tobacco": "Tobacco retail",
    "osm_food_retail": "Food retail",
}
PRIMARY_RELATIONS = ("nhs_gp", "osm_pharmacy")
THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
MODEL_LABELS = {
    "local_hgb": "Local HGB",
    "pooled_context_hgb": "Pooled Context-HGB",
    "continuous_gravity_hgb": "Continuous Gravity-HGB",
    "nested_gravity_hgb_tabpfn": "Nested Gravity HGB-TabPFN",
    "hfge": "HFGE",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Multi-model threshold robustness and breakpoint injection study"
    )
    result.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--bootstrap-replicates", type=int, default=2000)
    result.add_argument("--simulation-replicates", type=int, default=1000)
    result.add_argument("--spatial-blocks", type=int, default=25)
    # Match the frozen threshold-analysis block plan and resampling seeds.
    result.add_argument("--seed", type=int, default=20260827)
    return result


def bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    selected = values[finite]
    order = np.argsort(selected)
    ranked = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    adjusted[finite] = restored
    return adjusted


def bh_fdr_rows(values: np.ndarray) -> np.ndarray:
    return np.vstack([bh_fdr(row) for row in np.asarray(values, dtype=float)])


def make_spatial_blocks(
    coordinates: np.ndarray, n_blocks: int, seed: int
) -> np.ndarray:
    scaled = (coordinates - coordinates.mean(axis=0)) / coordinates.std(axis=0)
    return KMeans(
        n_clusters=n_blocks,
        n_init=20,
        random_state=seed,
    ).fit_predict(scaled)


def load_relation_distances(
    graph_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload = np.load(graph_path, allow_pickle=False)
    codes = payload["lsoa_codes"].astype(str)
    relations = payload["relation_names"].astype(str).tolist()[:-1]
    n_lsoa = len(codes)
    source, destination = payload["edge_index"]
    edge_type = payload["edge_type"]
    travel_time = payload["edge_impedance"].astype(float)
    nearest: dict[str, np.ndarray] = {}
    censored: dict[str, np.ndarray] = {}
    for relation_index, relation in enumerate(relations):
        selected = (edge_type == relation_index) & (destination < n_lsoa)
        relation_time = np.full(n_lsoa, np.inf, dtype=float)
        np.minimum.at(relation_time, destination[selected], travel_time[selected])
        relation_censored = ~np.isfinite(relation_time)
        relation_time[relation_censored] = 40.0
        nearest[relation] = relation_time
        censored[relation] = relation_censored.astype(float)
    return codes, nearest, censored


def aligned_values(
    frame: pd.DataFrame,
    expected_codes: np.ndarray,
    expected_folds: np.ndarray,
    value_column: str,
) -> np.ndarray:
    if not np.array_equal(
        frame["lsoa21cd"].astype(str).to_numpy(), expected_codes.astype(str)
    ):
        raise ValueError("OOF row order differs from the facility graph")
    if "fold" in frame and not np.array_equal(
        frame["fold"].to_numpy(dtype=int), expected_folds
    ):
        raise ValueError("OOF fold plan differs between model outputs")
    return frame[value_column].to_numpy(dtype=float)


def load_study(project_root: Path, study: str, n_blocks: int, seed: int) -> dict:
    oof_path = project_root / "results" / "oof_predictions" / f"{study}.parquet"
    oof = gpd.read_parquet(oof_path)
    graph_path = (
        project_root
        / "data"
        / "processed"
        / "case_studies"
        / study
        / "facility_lsoa_graph.npz"
    )
    codes, nearest, censored = load_relation_distances(graph_path)
    if not np.array_equal(oof["lsoa21cd"].astype(str).to_numpy(), codes):
        raise ValueError(f"OOF/graph row order mismatch for {study}")
    folds = oof["fold"].to_numpy(dtype=int)
    points = oof.geometry.representative_point()
    coordinates = np.column_stack((points.x.to_numpy(), points.y.to_numpy()))
    blocks = make_spatial_blocks(coordinates, n_blocks, seed)

    nested = pd.read_csv(
        project_root
        / "results"
        / "tabpfn_gravity"
        / f"nested_blend_{study}.csv"
    )
    ensemble = pd.read_parquet(
        project_root
        / "results"
        / "heterogeneous_graph_ensemble_relation_access_seeds5_e5"
        / f"{study}.parquet"
    )
    residuals = {
        "local_hgb": oof["residual_local_hgb"].to_numpy(dtype=float),
        "pooled_context_hgb": oof[
            "residual_proposed_pooled_hgb"
        ].to_numpy(dtype=float),
        "continuous_gravity_hgb": oof[
            "residual_facility_gravity_scale15"
        ].to_numpy(dtype=float),
        "nested_gravity_hgb_tabpfn": aligned_values(
            nested, codes, folds, "residual_nested_blend"
        ),
        "hfge": aligned_values(
            ensemble, codes, folds, "residual_heterogeneous_graph_ensemble"
        ),
    }
    if any(not np.isfinite(values).all() for values in residuals.values()):
        raise ValueError(f"non-finite OOF residuals for {study}")
    return {
        "codes": codes,
        "folds": folds,
        "observed": oof["observed"].to_numpy(dtype=float),
        "nearest": nearest,
        "censored": censored,
        "blocks": blocks,
        "residuals": residuals,
    }


def step_design(
    nearest: np.ndarray, censored: np.ndarray, threshold: float
) -> np.ndarray:
    return np.column_stack(
        (
            np.ones(len(nearest)),
            nearest / 5.0,
            (nearest > threshold).astype(float),
            np.maximum(nearest - threshold, 0.0) / 5.0,
            censored,
        )
    )


def cluster_step_summary(
    design: np.ndarray, outcome: np.ndarray, blocks: np.ndarray
) -> dict[str, float]:
    xtx_inverse = np.linalg.pinv(design.T @ design, rcond=1e-10)
    beta = xtx_inverse @ design.T @ outcome
    residual = outcome - design @ beta
    unique_blocks = np.unique(blocks)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    for block in unique_blocks:
        selected = blocks == block
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    n, k = design.shape
    g = len(unique_blocks)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    se = float(np.sqrt(max(covariance[2, 2], 0.0)))
    statistic = float(beta[2] / se) if se > 0 else float("nan")
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=g - 1))
    critical = float(stats.t.ppf(0.975, df=g - 1))
    return {
        "step": float(beta[2]),
        "cluster_se": se,
        "cluster_lower": float(beta[2] - critical * se),
        "cluster_upper": float(beta[2] + critical * se),
        "cluster_p": p_value,
    }


def paired_block_bootstrap_steps(
    design: np.ndarray,
    outcomes: np.ndarray,
    blocks: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_blocks = np.unique(blocks)
    block_xtx = np.stack(
        [design[blocks == block].T @ design[blocks == block] for block in unique_blocks]
    )
    block_xty = np.stack(
        [
            design[blocks == block].T @ outcomes[blocks == block]
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
    xty = np.einsum("bg,gim->bim", counts, block_xty, optimize=True)
    beta = np.einsum(
        "bij,bjm->bim", np.linalg.pinv(xtx, rcond=1e-10), xty, optimize=True
    )
    full_beta = np.linalg.pinv(design.T @ design, rcond=1e-10) @ design.T @ outcomes
    return full_beta[2], beta[:, 2, :]


def bootstrap_probability(draws: np.ndarray, estimate: float) -> float:
    centered = draws - estimate
    return float(
        min(
            (1.0 + np.sum(np.abs(centered) >= abs(estimate)))
            / (len(draws) + 1.0),
            1.0,
        )
    )


def run_multimodel_thresholds(
    studies: dict[str, dict], replicates: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_names = list(MODEL_LABELS)
    for study_index, study in enumerate(STUDIES):
        bundle = studies[study]
        outcomes = np.column_stack(
            [bundle["residuals"][model] for model in model_names]
        )
        for relation_index, relation in enumerate(RELATION_LABELS):
            nearest = bundle["nearest"][relation]
            censored = bundle["censored"][relation]
            for threshold_index, threshold in enumerate(THRESHOLDS):
                design = step_design(nearest, censored, threshold)
                estimates, draws = paired_block_bootstrap_steps(
                    design,
                    outcomes,
                    bundle["blocks"],
                    replicates,
                    seed
                    + study_index * 100_000
                    + relation_index * 1_000
                    + threshold_index,
                )
                for model_index, model in enumerate(model_names):
                    cluster = cluster_step_summary(
                        design, outcomes[:, model_index], bundle["blocks"]
                    )
                    model_draws = draws[:, model_index]
                    rows.append(
                        {
                            "study": study,
                            "relation": relation,
                            "threshold": threshold,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "step": float(estimates[model_index]),
                            "block_bootstrap_se": float(
                                np.std(model_draws, ddof=1)
                            ),
                            "block_bootstrap_lower": float(
                                np.quantile(model_draws, 0.025)
                            ),
                            "block_bootstrap_upper": float(
                                np.quantile(model_draws, 0.975)
                            ),
                            "block_bootstrap_p": bootstrap_probability(
                                model_draws, float(estimates[model_index])
                            ),
                            **cluster,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["model_family_q"] = frame.groupby("model", sort=False)[
        "block_bootstrap_p"
    ].transform(lambda values: bh_fdr(values.to_numpy(float)))
    frame["all_model_q"] = bh_fdr(frame["block_bootstrap_p"].to_numpy(float))
    frame["primary_care"] = frame["relation"].isin(PRIMARY_RELATIONS)
    return frame


def add_policy_bounds(
    threshold_results: pd.DataFrame, studies: dict[str, dict]
) -> pd.DataFrame:
    primary = threshold_results[threshold_results["primary_care"]].copy()
    city_sd = {
        study: float(np.std(bundle["observed"], ddof=1))
        for study, bundle in studies.items()
    }
    rows: list[pd.DataFrame] = []
    for fraction in (0.10, 0.20, 0.30):
        current = primary.copy()
        current["sesoi_sd"] = fraction
        current["city_ahah_sd"] = current["study"].map(city_sd)
        current["sesoi_ahah_points"] = current["sesoi_sd"] * current["city_ahah_sd"]
        current["largest_compatible_abs_effect"] = np.maximum(
            current["block_bootstrap_lower"].abs(),
            current["block_bootstrap_upper"].abs(),
        )
        current["largest_compatible_abs_effect_sd"] = (
            current["largest_compatible_abs_effect"] / current["city_ahah_sd"]
        )
        current["policy_relevant_effect_excluded"] = (
            (current["block_bootstrap_lower"] > -current["sesoi_ahah_points"])
            & (current["block_bootstrap_upper"] < current["sesoi_ahah_points"])
        )
        current["fdr_detected"] = current["model_family_q"] <= 0.05
        current["evidential_status"] = np.select(
            [
                current["fdr_detected"],
                current["policy_relevant_effect_excluded"],
            ],
            ["detected", "effect_at_or_above_bound_excluded"],
            default="inconclusive",
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def vectorized_cluster_tests(
    design: np.ndarray, outcomes: np.ndarray, blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return step estimates, standard errors, and p-values for outcome columns."""

    xtx_inverse = np.linalg.pinv(design.T @ design, rcond=1e-10)
    beta = xtx_inverse @ design.T @ outcomes
    residual = outcomes - design @ beta
    projection = design @ xtx_inverse[:, 2]
    unique_blocks = np.unique(blocks)
    cluster_scores = np.stack(
        [
            np.sum(projection[blocks == block, None] * residual[blocks == block], axis=0)
            for block in unique_blocks
        ]
    )
    n, k = design.shape
    g = len(unique_blocks)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.maximum(correction * np.sum(cluster_scores**2, axis=0), 0.0))
    statistic = np.divide(
        beta[2], se, out=np.full(se.shape, np.nan), where=se > 0
    )
    p_value = 2.0 * stats.t.sf(np.abs(statistic), df=g - 1)
    return beta[2], se, p_value


def wild_cluster_base(
    residual: np.ndarray,
    nearest: np.ndarray,
    censored: np.ndarray,
    blocks: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    null_design = np.column_stack(
        (np.ones(len(nearest)), nearest / 5.0, censored)
    )
    fitted = null_design @ (
        np.linalg.pinv(null_design.T @ null_design, rcond=1e-10)
        @ null_design.T
        @ residual
    )
    null_residual = residual - fitted
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(blocks)
    rademacher = rng.choice((-1.0, 1.0), size=(len(unique_blocks), replicates))
    block_lookup = {block: index for index, block in enumerate(unique_blocks)}
    row_weights = np.vstack([rademacher[block_lookup[block]] for block in blocks])
    return fitted[:, None] + null_residual[:, None] * row_weights, null_residual


def run_injection_simulation(
    studies: dict[str, dict], replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels: list[dict[str, object]] = []
    panel_index = 0
    for study_index, study in enumerate(STUDIES):
        bundle = studies[study]
        for relation_index, relation in enumerate(PRIMARY_RELATIONS):
            nearest = bundle["nearest"][relation]
            censored = bundle["censored"][relation]
            pseudo_null, null_residual = wild_cluster_base(
                bundle["residuals"]["hfge"],
                nearest,
                censored,
                bundle["blocks"],
                replicates,
                seed + study_index * 1000 + relation_index * 100,
            )
            panels.append(
                {
                    "panel_index": panel_index,
                    "study": study,
                    "relation": relation,
                    "nearest": nearest,
                    "censored": censored,
                    "blocks": bundle["blocks"],
                    "pseudo_null": pseudo_null,
                    "null_residual_sd": float(np.std(null_residual, ddof=1)),
                }
            )
            panel_index += 1

    scenario_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []

    def evaluate_scenario(
        true_threshold: float | None, injected_step: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_estimates = np.empty((replicates, len(panels), len(THRESHOLDS)))
        all_se = np.empty_like(all_estimates)
        all_p = np.empty_like(all_estimates)
        for panel_position, panel in enumerate(panels):
            outcomes = np.asarray(panel["pseudo_null"]).copy()
            if true_threshold is not None and injected_step != 0.0:
                outcomes += injected_step * (
                    np.asarray(panel["nearest"]) > true_threshold
                )[:, None]
            for threshold_index, candidate_threshold in enumerate(THRESHOLDS):
                design = step_design(
                    np.asarray(panel["nearest"]),
                    np.asarray(panel["censored"]),
                    candidate_threshold,
                )
                estimate, se, p_value = vectorized_cluster_tests(
                    design, outcomes, np.asarray(panel["blocks"])
                )
                all_estimates[:, panel_position, threshold_index] = estimate
                all_se[:, panel_position, threshold_index] = se
                all_p[:, panel_position, threshold_index] = p_value
        return all_estimates, all_se, all_p

    null_estimates, null_se, null_p = evaluate_scenario(None, 0.0)
    null_q = bh_fdr_rows(null_p.reshape(replicates, -1))
    null_selected = null_q <= 0.05
    scenario_rows.append(
        {
            "true_threshold": np.nan,
            "injected_step": 0.0,
            "absolute_step": 0.0,
            "direction": "null",
            "true_cell_power": np.nan,
            "any_true_cell_power": np.nan,
            "majority_true_cell_power": np.nan,
            "all_true_cell_power": np.nan,
            "any_off_target_selection": float(np.mean(null_selected.any(axis=1))),
            "mean_off_target_selections": float(np.mean(null_selected.sum(axis=1))),
            "mean_bias": float(np.mean(null_estimates)),
            "coverage_95": float(
                np.mean(
                    np.abs(null_estimates)
                    <= stats.t.ppf(0.975, 24) * null_se
                )
            ),
        }
    )

    for absolute_step in (0.5, 1.0, 1.5, 2.0):
        for direction in (-1.0, 1.0):
            injected_step = direction * absolute_step
            for true_threshold in THRESHOLDS:
                estimates, standard_errors, p_values = evaluate_scenario(
                    true_threshold, injected_step
                )
                q_values = bh_fdr_rows(p_values.reshape(replicates, -1)).reshape(
                    replicates, len(panels), len(THRESHOLDS)
                )
                selected = q_values <= 0.05
                true_index = THRESHOLDS.index(true_threshold)
                true_selected = selected[:, :, true_index]
                off_target = selected.copy()
                off_target[:, :, true_index] = False
                true_estimate = estimates[:, :, true_index]
                true_se = standard_errors[:, :, true_index]
                critical = stats.t.ppf(0.975, 24)
                coverage = (
                    (true_estimate - critical * true_se <= injected_step)
                    & (true_estimate + critical * true_se >= injected_step)
                )
                scenario_rows.append(
                    {
                        "true_threshold": true_threshold,
                        "injected_step": injected_step,
                        "absolute_step": absolute_step,
                        "direction": "negative" if direction < 0 else "positive",
                        "true_cell_power": float(np.mean(true_selected)),
                        "any_true_cell_power": float(
                            np.mean(true_selected.any(axis=1))
                        ),
                        "majority_true_cell_power": float(
                            np.mean(true_selected.sum(axis=1) >= 3)
                        ),
                        "all_true_cell_power": float(
                            np.mean(true_selected.all(axis=1))
                        ),
                        "any_off_target_selection": float(
                            np.mean(off_target.reshape(replicates, -1).any(axis=1))
                        ),
                        "mean_off_target_selections": float(
                            np.mean(off_target.reshape(replicates, -1).sum(axis=1))
                        ),
                        "mean_bias": float(np.mean(true_estimate - injected_step)),
                        "coverage_95": float(np.mean(coverage)),
                    }
                )
                for panel_position, panel in enumerate(panels):
                    panel_rows.append(
                        {
                            "study": panel["study"],
                            "relation": panel["relation"],
                            "true_threshold": true_threshold,
                            "injected_step": injected_step,
                            "absolute_step": absolute_step,
                            "direction": (
                                "negative" if direction < 0 else "positive"
                            ),
                            "power": float(
                                np.mean(true_selected[:, panel_position])
                            ),
                            "bias": float(
                                np.mean(
                                    true_estimate[:, panel_position] - injected_step
                                )
                            ),
                            "coverage_95": float(
                                np.mean(coverage[:, panel_position])
                            ),
                            "null_residual_sd": panel["null_residual_sd"],
                        }
                    )
    return pd.DataFrame(scenario_rows), pd.DataFrame(panel_rows)


def save_model_robustness_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    primary = frame[frame["primary_care"]].copy()
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(MODEL_LABELS)))
    offsets = np.linspace(-0.28, 0.28, len(MODEL_LABELS))
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex=True, layout="constrained")
    for row_index, study in enumerate(STUDIES):
        for column_index, relation in enumerate(PRIMARY_RELATIONS):
            axis = axes[row_index, column_index]
            panel = primary[
                (primary["study"] == study) & (primary["relation"] == relation)
            ]
            for model_index, model in enumerate(MODEL_LABELS):
                selected = panel[panel["model"] == model].sort_values("threshold")
                x = np.arange(len(THRESHOLDS)) + offsets[model_index]
                lower_error = selected["step"] - selected["block_bootstrap_lower"]
                upper_error = selected["block_bootstrap_upper"] - selected["step"]
                axis.errorbar(
                    x,
                    selected["step"],
                    yerr=np.vstack((lower_error, upper_error)),
                    color=colors[model_index],
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    capsize=2,
                    label=MODEL_LABELS[model],
                )
            axis.axhline(0.0, color="#555555", linewidth=0.8)
            axis.grid(axis="y", color="#e5e5e5", linewidth=0.7)
            axis.set_title(
                f"{STUDY_LABELS[study]} | {RELATION_LABELS[relation]}", fontsize=10.5
            )
            axis.set_xticks(np.arange(len(THRESHOLDS)), [f"{int(x)}" for x in THRESHOLDS])
            if row_index == 1:
                axis.set_xlabel("Candidate walking-time boundary (minutes)")
            if column_index == 0:
                axis.set_ylabel("Immediate residual step (AHAH points)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Primary-care threshold estimates across strict out-of-fold models",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_dir / "figure_threshold_model_robustness.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "figure_threshold_model_robustness.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_injection_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    nonnull = frame[frame["absolute_step"] > 0].copy()
    averaged = (
        nonnull.groupby(["true_threshold", "absolute_step"], as_index=False)
        .agg(
            true_cell_power=("true_cell_power", "mean"),
            any_true_cell_power=("any_true_cell_power", "mean"),
            any_off_target_selection=("any_off_target_selection", "mean"),
        )
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(THRESHOLDS)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    for threshold, color in zip(THRESHOLDS, colors, strict=True):
        selected = averaged[averaged["true_threshold"] == threshold]
        axes[0].plot(
            selected["absolute_step"],
            selected["true_cell_power"],
            color=color,
            marker="o",
            label=f"{int(threshold)} min",
        )
        axes[1].plot(
            selected["absolute_step"],
            selected["any_off_target_selection"],
            color=color,
            marker="o",
            label=f"{int(threshold)} min",
        )
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Injected immediate step (absolute AHAH points)")
        axis.grid(color="#e8e8e8", linewidth=0.7)
    axes[0].set_ylabel("True-boundary detection rate")
    axes[0].set_title("Recovery of the injected boundary")
    axes[1].set_ylabel("Probability of any off-target selection")
    axes[1].set_title("Selection at another candidate boundary")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(
        "Spatial breakpoint-injection calibration",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_dir / "figure_breakpoint_injection.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "figure_breakpoint_injection.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def result_summary(
    thresholds: pd.DataFrame,
    policy: pd.DataFrame,
    injection: pd.DataFrame,
    studies: dict[str, dict],
    bootstrap_replicates: int,
    simulation_replicates: int,
) -> dict[str, object]:
    discovery_counts = (
        thresholds.groupby("model", as_index=False)
        .agg(
            all_fdr_discoveries=("model_family_q", lambda value: int((value <= 0.05).sum())),
            primary_fdr_discoveries=(
                "model_family_q",
                lambda value: 0,
            ),
        )
        .drop(columns="primary_fdr_discoveries")
    )
    primary_counts = (
        thresholds[thresholds["primary_care"]]
        .groupby("model")["model_family_q"]
        .apply(lambda value: int((value <= 0.05).sum()))
    )
    discovery_counts["primary_fdr_discoveries"] = discovery_counts["model"].map(
        primary_counts
    )

    primary = thresholds[thresholds["primary_care"]]
    pivot = primary.pivot_table(
        index=["study", "relation", "threshold"], columns="model", values="step"
    )
    correlations = pivot.corr()
    signs = np.sign(pivot.to_numpy(float))
    sign_agreement = np.mean(np.all(signs == signs[:, [0]], axis=1))

    hfge_policy = policy[policy["model"] == "hfge"]
    policy_counts = (
        hfge_policy.groupby("sesoi_sd")["evidential_status"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    null = injection[injection["absolute_step"] == 0].iloc[0]
    nonnull = injection[injection["absolute_step"] > 0]
    power_by_size = (
        nonnull.groupby("absolute_step", as_index=False)
        .agg(
            mean_true_cell_power=("true_cell_power", "mean"),
            minimum_true_cell_power=("true_cell_power", "min"),
            maximum_true_cell_power=("true_cell_power", "max"),
            mean_any_true_cell_power=("any_true_cell_power", "mean"),
            mean_any_off_target_selection=("any_off_target_selection", "mean"),
            mean_coverage_95=("coverage_95", "mean"),
            maximum_absolute_bias=("mean_bias", lambda value: float(np.max(np.abs(value)))),
        )
    )
    return {
        "design": {
            "models": MODEL_LABELS,
            "threshold_family_per_model": 96,
            "bootstrap_replicates": bootstrap_replicates,
            "injection_simulation_replicates": simulation_replicates,
            "spatial_blocks": 25,
            "injection_scope": (
                "inference-stage semi-synthetic calibration using the empirical "
                "HFGE OOF residual surface, 25 spatial blocks, the same step-hinge "
                "specification, and BH-FDR across the 16 primary-care tests"
            ),
            "policy_bounds": (
                "95% spatial-block bootstrap intervals compared with 0.10, 0.20, "
                "and 0.30 within-city AHAH standard deviations; these are design "
                "benchmarks rather than externally validated policy effects"
            ),
        },
        "city_ahah_sd": {
            study: float(np.std(bundle["observed"], ddof=1))
            for study, bundle in studies.items()
        },
        "multimodel": {
            "discovery_counts": discovery_counts.to_dict(orient="records"),
            "primary_step_correlations": correlations.round(6).to_dict(),
            "complete_sign_agreement_fraction": float(sign_agreement),
        },
        "policy_bound_counts_hfge": policy_counts.to_dict(orient="records"),
        "injection": {
            "null_probability_any_fdr_selection": float(
                null["any_off_target_selection"]
            ),
            "null_mean_number_fdr_selections": float(
                null["mean_off_target_selections"]
            ),
            "power_by_absolute_step": power_by_size.to_dict(orient="records"),
        },
    }


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    studies = {
        study: load_study(
            args.project_root,
            study,
            args.spatial_blocks,
            args.seed + study_index,
        )
        for study_index, study in enumerate(STUDIES)
    }
    thresholds = run_multimodel_thresholds(
        studies, args.bootstrap_replicates, args.seed
    )
    policy = add_policy_bounds(thresholds, studies)
    injection, injection_panels = run_injection_simulation(
        studies, args.simulation_replicates, args.seed + 9_000_000
    )

    thresholds.to_csv(args.output_dir / "multimodel_threshold_tests.csv", index=False)
    policy.to_csv(args.output_dir / "policy_effect_bounds.csv", index=False)
    injection.to_csv(args.output_dir / "breakpoint_injection_summary.csv", index=False)
    injection_panels.to_csv(
        args.output_dir / "breakpoint_injection_by_city_relation.csv", index=False
    )
    save_model_robustness_figure(thresholds, args.output_dir)
    save_injection_figure(injection, args.output_dir)
    summary = result_summary(
        thresholds,
        policy,
        injection,
        studies,
        args.bootstrap_replicates,
        args.simulation_replicates,
    )
    (args.output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
