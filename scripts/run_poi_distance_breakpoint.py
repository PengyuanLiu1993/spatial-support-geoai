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
from sklearn.ensemble import HistGradientBoostingRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


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
RELATION_ROLE = {
    "nhs_gp": "health-supporting",
    "nhs_dentist": "health-supporting",
    "osm_hospital": "health-supporting",
    "osm_pharmacy": "health-supporting",
    "osm_leisure": "health-supporting",
    "os_green": "health-supporting",
    "os_active_green": "health-supporting",
    "osm_fast_food": "health-constraining",
    "osm_gambling": "health-constraining",
    "osm_pub_bar": "health-constraining",
    "osm_tobacco": "health-constraining",
    "osm_food_retail": "mixed",
}
EXPECTED_SIGN = {
    relation: (1 if role == "health-supporting" else -1 if role == "health-constraining" else 0)
    for relation, role in RELATION_ROLE.items()
}
RELATION_EXCLUDE_TOKENS = {
    "nhs_gp": ("nhs_gp", "osm_primary_care"),
    "nhs_dentist": ("nhs_dentist", "osm_dentist"),
    "osm_hospital": ("osm_hospital",),
    "osm_pharmacy": ("osm_pharmacy",),
    "osm_leisure": ("osm_leisure",),
    "os_green": ("os_green_",),
    "os_active_green": ("active_green",),
    "osm_fast_food": ("osm_fast_food",),
    "osm_gambling": ("osm_gambling",),
    "osm_pub_bar": ("osm_pub_bar",),
    "osm_tobacco": ("osm_tobacco",),
    "osm_food_retail": ("osm_food_retail",),
}
CORE_RELATIONS = ("nhs_gp", "osm_hospital", "os_green", "osm_fast_food")
FIXED_THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
SEARCH_THRESHOLDS = tuple(float(value) for value in range(5, 26))
CURVE_GRID = np.linspace(0.0, 35.0, 141)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relation-specific walking-time breakpoint analysis of the AHAH index"
        )
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/poi_distance_breakpoint")
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--spatial-blocks", type=int, default=25)
    return parser


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    selected = values[finite]
    order = np.argsort(selected)
    ranked = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    output[finite] = adjusted
    return output


def load_case(case_root: Path, oof_dir: Path, study: str) -> dict[str, object]:
    graph_path = case_root / study / "facility_lsoa_graph.npz"
    payload = np.load(graph_path, allow_pickle=False)
    lsoa_codes = payload["lsoa_codes"].astype(str)
    n_lsoa = len(lsoa_codes)
    relations = payload["relation_names"].astype(str).tolist()[:-1]
    frame = gpd.read_parquet(oof_dir / f"{study}.parquet")
    if not np.array_equal(frame["lsoa21cd"].astype(str).to_numpy(), lsoa_codes):
        raise ValueError(f"OOF/graph row order mismatch for {study}")
    source, destination = payload["edge_index"]
    edge_type = payload["edge_type"]
    travel_time = payload["edge_impedance"].astype(float)
    nearest: dict[str, np.ndarray] = {}
    censored: dict[str, np.ndarray] = {}
    for relation_index, relation in enumerate(relations):
        selected = (edge_type == relation_index) & (destination < n_lsoa)
        relation_time = np.full(n_lsoa, np.inf, dtype=float)
        np.minimum.at(
            relation_time,
            destination[selected],
            travel_time[selected],
        )
        relation_censored = ~np.isfinite(relation_time)
        relation_time[relation_censored] = 40.0
        nearest[relation] = relation_time
        censored[relation] = relation_censored.astype(float)
    points = frame.geometry.representative_point()
    coordinates = np.column_stack((points.x.to_numpy(), points.y.to_numpy()))
    return {
        "frame": frame,
        "x": payload["x"][:n_lsoa].astype(float),
        "feature_names": payload["feature_names"].astype(str).tolist(),
        "y": frame["observed"].to_numpy(dtype=float),
        "fold": frame["fold"].to_numpy(dtype=int),
        "coordinates": coordinates,
        "nearest": nearest,
        "censored": censored,
        "relations": relations,
    }


def relation_context(bundle: dict[str, object], relation: str) -> tuple[np.ndarray, list[str]]:
    feature_names = list(bundle["feature_names"])
    tokens = RELATION_EXCLUDE_TOKENS[relation]
    keep = np.asarray(
        [not any(token in name for token in tokens) for name in feature_names],
        dtype=bool,
    )
    return np.asarray(bundle["x"])[:, keep], [
        name for name, selected in zip(feature_names, keep, strict=True) if selected
    ]


def fit_nuisance_residuals(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(int(value) for value in np.unique(folds)):
        train = folds != fold
        test = folds == fold
        model = HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=400,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=3.0,
            random_state=seed + fold,
        )
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("incomplete nuisance predictions")
    return y - prediction, prediction


def make_spatial_blocks(coordinates: np.ndarray, n_blocks: int, seed: int) -> np.ndarray:
    scaled = (coordinates - coordinates.mean(axis=0)) / coordinates.std(axis=0)
    return KMeans(
        n_clusters=n_blocks,
        n_init=20,
        random_state=seed,
    ).fit_predict(scaled)


def design_matrix(
    travel_time: np.ndarray,
    censored: np.ndarray,
    model: str,
    threshold: float | None = None,
) -> np.ndarray:
    columns = [np.ones(len(travel_time)), travel_time]
    if model == "linear":
        pass
    elif model == "hinge":
        if threshold is None:
            raise ValueError("hinge model requires a threshold")
        columns.append(np.maximum(travel_time - threshold, 0.0))
    elif model == "step_hinge":
        if threshold is None:
            raise ValueError("step-hinge model requires a threshold")
        columns.append((travel_time > threshold).astype(float))
        columns.append(np.maximum(travel_time - threshold, 0.0))
    else:
        raise ValueError(f"unknown model: {model}")
    # Keep this column even when predicting a fully observed theoretical curve so
    # the fitted and prediction matrices always have the same schema.
    columns.append(censored)
    return np.column_stack(columns)


def weighted_ols(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    positive = weights > 0
    x_selected = x[positive]
    y_selected = y[positive]
    weight_selected = weights[positive]
    xtwx = x_selected.T @ (weight_selected[:, None] * x_selected)
    xtwy = x_selected.T @ (weight_selected * y_selected)
    beta = np.linalg.pinv(xtwx, rcond=1e-10) @ xtwy
    residual = y_selected - x_selected @ beta
    rss = float(np.dot(weight_selected, residual * residual))
    return beta, residual, rss


def bic_value(rss: float, effective_n: float, parameters: int) -> float:
    return float(
        effective_n * np.log(max(rss / effective_n, np.finfo(float).tiny))
        + parameters * np.log(effective_n)
    )


def valid_threshold(
    travel_time: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
) -> bool:
    if weights is None:
        weights = np.ones(len(travel_time), dtype=float)
    below = float(weights[travel_time <= threshold].sum())
    above = float(weights[travel_time > threshold].sum())
    minimum = max(50.0, 0.04 * float(weights.sum()))
    return below >= minimum and above >= minimum


def select_threshold(
    y: np.ndarray,
    travel_time: np.ndarray,
    censored: np.ndarray,
    model: str,
    thresholds: tuple[float, ...],
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    best: tuple[float, np.ndarray, float] | None = None
    effective_n = float(weights.sum())
    for threshold in thresholds:
        if not valid_threshold(travel_time, threshold, weights):
            continue
        x = design_matrix(travel_time, censored, model, threshold)
        beta, _, rss = weighted_ols(x, y, weights)
        bic = bic_value(rss, effective_n, x.shape[1])
        if best is None or bic < best[2]:
            best = (threshold, beta, bic)
    if best is None:
        raise ValueError("no threshold has adequate observations on both sides")
    return best


def predict(
    beta: np.ndarray,
    travel_time: np.ndarray,
    censored: np.ndarray,
    model: str,
    threshold: float | None = None,
) -> np.ndarray:
    return design_matrix(travel_time, censored, model, threshold) @ beta


def nested_spatial_cv(
    y: np.ndarray,
    travel_time: np.ndarray,
    censored: np.ndarray,
    folds: np.ndarray,
    model: str,
) -> dict[str, object]:
    prediction = np.full(len(y), np.nan, dtype=float)
    selected_thresholds: list[float] = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if model == "linear":
            train_x = design_matrix(travel_time[train], censored[train], model)
            beta, _, _ = weighted_ols(train_x, y[train])
            prediction[test] = predict(
                beta, travel_time[test], censored[test], model
            )
        else:
            threshold, beta, _ = select_threshold(
                y[train],
                travel_time[train],
                censored[train],
                model,
                SEARCH_THRESHOLDS,
            )
            selected_thresholds.append(float(threshold))
            prediction[test] = predict(
                beta,
                travel_time[test],
                censored[test],
                model,
                threshold,
            )
    rmse = float(np.sqrt(np.mean((y - prediction) ** 2)))
    return {
        "rmse": rmse,
        "prediction": prediction,
        "selected_thresholds": selected_thresholds,
    }


def cluster_robust_inference(
    x: np.ndarray,
    y: np.ndarray,
    cluster: np.ndarray,
) -> dict[str, np.ndarray]:
    beta, _, _ = weighted_ols(x, y)
    residual = y - x @ beta
    xtx_inverse = np.linalg.pinv(x.T @ x, rcond=1e-10)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    unique_clusters = np.unique(cluster)
    for cluster_id in unique_clusters:
        selected = cluster == cluster_id
        score = x[selected].T @ residual[selected]
        meat += np.outer(score, score)
    n = len(y)
    k = x.shape[1]
    g = len(unique_clusters)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistic = np.divide(
        beta,
        standard_error,
        out=np.full(beta.shape, np.nan),
        where=standard_error > 0,
    )
    p_value = 2.0 * stats.t.sf(np.abs(statistic), df=g - 1)
    return {
        "beta": beta,
        "se": standard_error,
        "p": p_value,
        "covariance": covariance,
    }


def coefficient_summary(beta: np.ndarray, model: str) -> dict[str, float]:
    pre_slope = float(beta[1])
    if model == "hinge":
        return {
            "pre_slope": pre_slope,
            "step": 0.0,
            "slope_change": float(beta[2]),
            "post_slope": float(beta[1] + beta[2]),
        }
    if model == "step_hinge":
        return {
            "pre_slope": pre_slope,
            "step": float(beta[2]),
            "slope_change": float(beta[3]),
            "post_slope": float(beta[1] + beta[3]),
        }
    return {
        "pre_slope": pre_slope,
        "step": 0.0,
        "slope_change": 0.0,
        "post_slope": pre_slope,
    }


def bootstrap_breakpoint(
    y: np.ndarray,
    travel_time: np.ndarray,
    censored: np.ndarray,
    spatial_block: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(spatial_block)
    candidate_statistics: dict[float, dict[str, np.ndarray]] = {}
    for threshold in SEARCH_THRESHOLDS:
        if not valid_threshold(travel_time, threshold):
            continue
        x = design_matrix(travel_time, censored, "step_hinge", threshold)
        xtx = np.zeros((len(unique_blocks), x.shape[1], x.shape[1]), dtype=float)
        xty = np.zeros((len(unique_blocks), x.shape[1]), dtype=float)
        yty = np.zeros(len(unique_blocks), dtype=float)
        below = np.zeros(len(unique_blocks), dtype=float)
        above = np.zeros(len(unique_blocks), dtype=float)
        for block_index, block_id in enumerate(unique_blocks):
            selected = spatial_block == block_id
            block_x = x[selected]
            block_y = y[selected]
            xtx[block_index] = block_x.T @ block_x
            xty[block_index] = block_x.T @ block_y
            yty[block_index] = float(block_y @ block_y)
            below[block_index] = float(np.sum(travel_time[selected] <= threshold))
            above[block_index] = float(np.sum(travel_time[selected] > threshold))
        candidate_statistics[threshold] = {
            "xtx": xtx,
            "xty": xty,
            "yty": yty,
            "below": below,
            "above": above,
        }
    rows: list[dict[str, float]] = []
    curve_rows: list[np.ndarray] = []
    grid_censored = np.zeros(len(CURVE_GRID), dtype=float)
    reference_censored = np.zeros(1, dtype=float)
    for _ in range(replicates):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        counts = np.asarray(
            [np.sum(sampled == block_id) for block_id in unique_blocks], dtype=float
        )
        effective_n = float(
            np.dot(
                counts,
                [np.sum(spatial_block == block_id) for block_id in unique_blocks],
            )
        )
        minimum_side = max(50.0, 0.04 * effective_n)
        best: tuple[float, np.ndarray, float] | None = None
        for threshold, sufficient in candidate_statistics.items():
            if (
                float(counts @ sufficient["below"]) < minimum_side
                or float(counts @ sufficient["above"]) < minimum_side
            ):
                continue
            xtwx = np.tensordot(counts, sufficient["xtx"], axes=(0, 0))
            xtwy = counts @ sufficient["xty"]
            yty = float(counts @ sufficient["yty"])
            beta = np.linalg.pinv(xtwx, rcond=1e-10) @ xtwy
            rss = float(yty - 2.0 * beta @ xtwy + beta @ xtwx @ beta)
            bic = bic_value(rss, effective_n, len(beta))
            if best is None or bic < best[2]:
                best = (threshold, beta, bic)
        if best is None:
            continue
        threshold, beta, _ = best
        summary = coefficient_summary(beta, "step_hinge")
        rows.append({"threshold": threshold, **summary})
        curve = predict(
            beta,
            CURVE_GRID,
            grid_censored,
            "step_hinge",
            threshold,
        )
        reference = predict(
            beta,
            np.asarray([5.0]),
            reference_censored,
            "step_hinge",
            threshold,
        )[0]
        curve_rows.append(curve - reference)
    bootstrap = pd.DataFrame(rows)
    curves = np.vstack(curve_rows)
    intervals = {}
    for name in ("threshold", "step", "pre_slope", "post_slope", "slope_change"):
        values = bootstrap[name].to_numpy(dtype=float)
        intervals[name] = {
            "median": float(np.median(values)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
    curve_frame = pd.DataFrame(
        {
            "travel_time": CURVE_GRID,
            "effect_median": np.median(curves, axis=0),
            "effect_lower": np.quantile(curves, 0.025, axis=0),
            "effect_upper": np.quantile(curves, 0.975, axis=0),
        }
    )
    return {
        "successful_replicates": int(len(bootstrap)),
        "intervals": intervals,
    }, curve_frame


def binned_residuals(y: np.ndarray, travel_time: np.ndarray) -> pd.DataFrame:
    boundaries = np.arange(0.0, 42.0, 2.0)
    bin_id = np.digitize(travel_time, boundaries, right=False) - 1
    rows = []
    for index in range(len(boundaries) - 1):
        selected = bin_id == index
        if int(selected.sum()) < 20:
            continue
        rows.append(
            {
                "bin_mid": float((boundaries[index] + boundaries[index + 1]) / 2),
                "mean": float(np.mean(y[selected])),
                "se": float(np.std(y[selected], ddof=1) / np.sqrt(selected.sum())),
                "n": int(selected.sum()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    reference_index = int(np.argmin(np.abs(frame["bin_mid"].to_numpy() - 5.0)))
    reference = float(frame.iloc[reference_index]["mean"])
    frame["mean"] -= reference
    return frame


def fixed_threshold_rows(
    study: str,
    relation: str,
    residual: np.ndarray,
    travel_time: np.ndarray,
    censored: np.ndarray,
    spatial_block: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in FIXED_THRESHOLDS:
        if not valid_threshold(travel_time, threshold):
            rows.append(
                {
                    "study": study,
                    "relation": relation,
                    "relation_label": RELATION_LABELS[relation],
                    "role": RELATION_ROLE[relation],
                    "threshold": threshold,
                    "valid": False,
                }
            )
            continue
        x = design_matrix(travel_time, censored, "step_hinge", threshold)
        inference = cluster_robust_inference(x, residual, spatial_block)
        beta = inference["beta"]
        covariance = inference["covariance"]
        post_contrast = np.zeros(len(beta), dtype=float)
        post_contrast[1] = 1.0
        post_contrast[3] = 1.0
        post_slope = float(post_contrast @ beta)
        post_se = float(np.sqrt(post_contrast @ covariance @ post_contrast))
        post_p = float(
            2.0
            * stats.t.sf(
                abs(post_slope / post_se),
                df=len(np.unique(spatial_block)) - 1,
            )
        )
        rows.append(
            {
                "study": study,
                "relation": relation,
                "relation_label": RELATION_LABELS[relation],
                "role": RELATION_ROLE[relation],
                "threshold": threshold,
                "valid": True,
                "n_below_or_equal": int(np.sum(travel_time <= threshold)),
                "n_above": int(np.sum(travel_time > threshold)),
                "step": float(beta[2]),
                "step_se": float(inference["se"][2]),
                "step_p": float(inference["p"][2]),
                "slope_change": float(beta[3]),
                "slope_change_se": float(inference["se"][3]),
                "slope_change_p": float(inference["p"][3]),
                "pre_slope": float(beta[1]),
                "pre_slope_p": float(inference["p"][1]),
                "post_slope": post_slope,
                "post_slope_se": post_se,
                "post_slope_p": post_p,
            }
        )
    return rows


def evidence_label(row: pd.Series) -> str:
    step_interval = row["bootstrap_step_lower"], row["bootstrap_step_upper"]
    slope_interval = (
        row["bootstrap_slope_change_lower"],
        row["bootstrap_slope_change_upper"],
    )
    abrupt = step_interval[0] > 0 or step_interval[1] < 0
    slope_change = slope_interval[0] > 0 or slope_interval[1] < 0
    if abrupt and slope_change:
        return "step and slope change"
    if abrupt:
        return "abrupt step"
    if slope_change:
        return "slope change only"
    return "no robust breakpoint"


def save_figures(
    summary: pd.DataFrame,
    fixed: pd.DataFrame,
    curves: pd.DataFrame,
    bins: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    colors = {
        "health-supporting": "#087e8b",
        "health-constraining": "#c44536",
        "mixed": "#6a5acd",
    }
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.8), layout="constrained")
    for row_index, study in enumerate(STUDIES):
        for column_index, relation in enumerate(CORE_RELATIONS):
            axis = axes[row_index, column_index]
            selected_curve = curves[
                (curves["study"] == study) & (curves["relation"] == relation)
            ]
            selected_bins = bins[
                (bins["study"] == study) & (bins["relation"] == relation)
            ]
            selected_summary = summary[
                (summary["study"] == study) & (summary["relation"] == relation)
            ].iloc[0]
            color = colors[RELATION_ROLE[relation]]
            axis.axhline(0, color="#888888", linewidth=0.8)
            for threshold in FIXED_THRESHOLDS:
                axis.axvline(threshold, color="#e5e5e5", linewidth=0.7, zorder=0)
            axis.fill_between(
                selected_curve["travel_time"],
                selected_curve["effect_lower"],
                selected_curve["effect_upper"],
                color=color,
                alpha=0.17,
                linewidth=0,
            )
            axis.plot(
                selected_curve["travel_time"],
                selected_curve["effect_median"],
                color=color,
                linewidth=2.0,
            )
            if not selected_bins.empty:
                axis.errorbar(
                    selected_bins["bin_mid"],
                    selected_bins["mean"],
                    yerr=1.96 * selected_bins["se"],
                    fmt="o",
                    color="#333333",
                    markerfacecolor="white",
                    markersize=3.5,
                    linewidth=0.7,
                    alpha=0.75,
                )
            threshold = float(selected_summary["selected_step_threshold"])
            axis.axvline(threshold, color=color, linestyle="--", linewidth=1.4)
            axis.text(
                0.97,
                0.94,
                f"Exploratory breakpoint: {threshold:.0f} min\n{selected_summary['evidence']}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            )
            if row_index == 0:
                axis.set_title(RELATION_LABELS[relation], fontsize=11)
            if column_index == 0:
                axis.set_ylabel(
                    f"{STUDY_LABELS[study]}\nAdjusted AHAH difference\n(relative to 5 min)"
                )
            axis.set_xlim(0, 35)
            axis.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
            axis.set_xlabel("Nearest routed walking time (min)")
            axis.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    fig.suptitle(
        "Facility-specific walking-time breakpoints in neighbourhood environmental health",
        fontsize=14,
    )
    curve_png = output_dir / "figure_poi_distance_breakpoint_curves.png"
    curve_pdf = output_dir / "figure_poi_distance_breakpoint_curves.pdf"
    fig.savefig(curve_png, dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(curve_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    city_curve_outputs: dict[str, dict[str, str]] = {}
    for study in STUDIES:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), layout="constrained")
        for panel_index, relation in enumerate(CORE_RELATIONS):
            axis = axes.flat[panel_index]
            selected_curve = curves[
                (curves["study"] == study) & (curves["relation"] == relation)
            ]
            selected_bins = bins[
                (bins["study"] == study) & (bins["relation"] == relation)
            ]
            selected_summary = summary[
                (summary["study"] == study) & (summary["relation"] == relation)
            ].iloc[0]
            color = colors[RELATION_ROLE[relation]]
            axis.axhline(0, color="#888888", linewidth=0.8)
            for fixed_threshold in FIXED_THRESHOLDS:
                axis.axvline(
                    fixed_threshold,
                    color="#e5e5e5",
                    linewidth=0.7,
                    zorder=0,
                )
            axis.fill_between(
                selected_curve["travel_time"],
                selected_curve["effect_lower"],
                selected_curve["effect_upper"],
                color=color,
                alpha=0.17,
                linewidth=0,
            )
            axis.plot(
                selected_curve["travel_time"],
                selected_curve["effect_median"],
                color=color,
                linewidth=2.2,
            )
            if not selected_bins.empty:
                axis.errorbar(
                    selected_bins["bin_mid"],
                    selected_bins["mean"],
                    yerr=1.96 * selected_bins["se"],
                    fmt="o",
                    color="#333333",
                    markerfacecolor="white",
                    markersize=4.0,
                    linewidth=0.8,
                    alpha=0.78,
                )
            threshold = float(selected_summary["selected_step_threshold"])
            axis.axvline(threshold, color=color, linestyle="--", linewidth=1.5)
            axis.text(
                0.97,
                0.94,
                f"Exploratory breakpoint: {threshold:.0f} min\n{selected_summary['evidence']}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            )
            axis.set_title(RELATION_LABELS[relation], fontsize=11.5)
            axis.set_xlim(0, 35)
            axis.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
            axis.set_xlabel("Nearest routed walking time (min)")
            axis.set_ylabel("Adjusted AHAH difference\n(relative to 5 min)")
            axis.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        fig.suptitle(
            f"{STUDY_LABELS[study]}: facility-specific walking-time breakpoints",
            fontsize=14,
        )
        city_png = output_dir / f"figure_poi_distance_breakpoint_{study}.png"
        city_pdf = output_dir / f"figure_poi_distance_breakpoint_{study}.pdf"
        fig.savefig(city_png, dpi=600, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(city_pdf, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        city_curve_outputs[study] = {
            "png": str(city_png),
            "pdf": str(city_pdf),
        }

    valid_fixed = fixed[fixed["valid"]].copy()
    max_abs = float(np.nanquantile(np.abs(valid_fixed["step"]), 0.98))
    max_abs = max(max_abs, 0.1)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.4), layout="constrained")
    ordered_relations = list(RELATION_LABELS)
    image = None
    for axis, study in zip(axes, STUDIES, strict=True):
        matrix = np.full((len(ordered_relations), len(FIXED_THRESHOLDS)), np.nan)
        q_matrix = np.full_like(matrix, np.nan)
        city = fixed[(fixed["study"] == study) & fixed["valid"]]
        for row_index, relation in enumerate(ordered_relations):
            for column_index, threshold in enumerate(FIXED_THRESHOLDS):
                selected = city[
                    (city["relation"] == relation)
                    & (city["threshold"] == threshold)
                ]
                if not selected.empty:
                    matrix[row_index, column_index] = selected.iloc[0]["step"]
                    q_matrix[row_index, column_index] = selected.iloc[0]["step_q_bh"]
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    label = "NA"
                else:
                    label = f"{value:+.2f}"
                    if q_matrix[row_index, column_index] <= 0.05:
                        label += "*"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=("white" if np.isfinite(value) and abs(value) > 0.58 * max_abs else "#222222"),
                )
        axis.set_xticks(range(len(FIXED_THRESHOLDS)), [f"{int(value)}" for value in FIXED_THRESHOLDS])
        axis.set_yticks(
            range(len(ordered_relations)),
            [RELATION_LABELS[value] for value in ordered_relations],
        )
        axis.set_xlabel("Pre-specified walking-time threshold (min)")
        axis.set_title(STUDY_LABELS[study])
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Immediate AHAH step after threshold")
    fig.suptitle(
        "Tests of immediate changes at 5, 10, 15 and 20 minutes\n* BH-FDR q <= 0.05 across all city-relation-threshold tests",
        fontsize=13,
    )
    heatmap_png = output_dir / "figure_poi_fixed_threshold_heatmap.png"
    heatmap_pdf = output_dir / "figure_poi_fixed_threshold_heatmap.pdf"
    fig.savefig(heatmap_png, dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(heatmap_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return {
        "breakpoint_curves": {"png": str(curve_png), "pdf": str(curve_pdf)},
        "breakpoint_curves_by_city": city_curve_outputs,
        "fixed_threshold_heatmap": {
            "png": str(heatmap_png),
            "pdf": str(heatmap_pdf),
        },
    }


def interval_text(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:.2f} [{lower:.2f}, {upper:.2f}]"


def write_core_latex_table(summary: pd.DataFrame, output: Path) -> None:
    core = summary[summary["relation"].isin(CORE_RELATIONS)].copy()
    lines = [
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "City & Facility & Breakpoint & Step & Pre-slope & Post-slope & Evidence \\\\",
        "\\midrule",
    ]
    for study in STUDIES:
        city = core[core["study"] == study]
        for relation in CORE_RELATIONS:
            row = city[city["relation"] == relation].iloc[0]
            lines.append(
                " & ".join(
                    (
                        STUDY_LABELS[study].replace("Greater ", ""),
                        RELATION_LABELS[relation],
                        f"{row['selected_step_threshold']:.0f}",
                        interval_text(
                            row["step"],
                            row["bootstrap_step_lower"],
                            row["bootstrap_step_upper"],
                        ),
                        interval_text(
                            row["pre_slope"],
                            row["bootstrap_pre_slope_lower"],
                            row["bootstrap_pre_slope_upper"],
                        ),
                        interval_text(
                            row["post_slope"],
                            row["bootstrap_post_slope_lower"],
                            row["bootstrap_post_slope_upper"],
                        ),
                        str(row["evidence"]),
                    )
                )
                + " \\\\"
            )
        if study != STUDIES[-1]:
            lines.append("\\addlinespace")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        study: load_case(args.case_root, args.oof_dir, study) for study in STUDIES
    }
    summary_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    nuisance_rows: list[pd.DataFrame] = []
    settings = {
        "outcome": (
            "Access to Healthy Assets and Hazards v5.1 composite; higher values "
            "denote less healthy neighbourhood environments"
        ),
        "exposure": (
            "nearest routed walking time by facility relation; no facility within "
            "40 minutes is right-censored at 40 minutes with a censor indicator"
        ),
        "fixed_thresholds_minutes": FIXED_THRESHOLDS,
        "exploratory_search_minutes": SEARCH_THRESHOLDS,
        "adjustment": (
            "relation-specific five-fold cross-fitted HGB residual after removing "
            "context features derived from the tested facility relation"
        ),
        "breakpoint_model": (
            "step-hinge regression with pre-threshold slope, immediate level step, "
            "post-threshold slope change, and censor indicator"
        ),
        "fixed_threshold_inference": (
            "two-sided cluster-robust t tests over spatial k-means blocks; "
            "Benjamini-Hochberg FDR across both cities, all relations, and four thresholds"
        ),
        "exploratory_uncertainty": (
            "spatial block bootstrap with breakpoint re-selection in each replicate"
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "spatial_blocks": args.spatial_blocks,
        "causal_claim": False,
    }
    for study_index, study in enumerate(STUDIES):
        bundle = bundles[study]
        spatial_block = make_spatial_blocks(
            np.asarray(bundle["coordinates"]),
            args.spatial_blocks,
            args.seed + study_index,
        )
        for relation_index, relation in enumerate(bundle["relations"]):
            x_context, retained_features = relation_context(bundle, relation)
            residual, nuisance_prediction = fit_nuisance_residuals(
                x_context,
                np.asarray(bundle["y"]),
                np.asarray(bundle["fold"]),
                args.seed + study_index * 10000 + relation_index * 100,
            )
            travel_time = np.asarray(bundle["nearest"][relation], dtype=float)
            censored = np.asarray(bundle["censored"][relation], dtype=float)
            full_threshold, full_beta, full_bic = select_threshold(
                residual,
                travel_time,
                censored,
                "step_hinge",
                SEARCH_THRESHOLDS,
            )
            hinge_threshold, _, hinge_bic = select_threshold(
                residual,
                travel_time,
                censored,
                "hinge",
                SEARCH_THRESHOLDS,
            )
            coefficient = coefficient_summary(full_beta, "step_hinge")
            linear_cv = nested_spatial_cv(
                residual,
                travel_time,
                censored,
                np.asarray(bundle["fold"]),
                "linear",
            )
            hinge_cv = nested_spatial_cv(
                residual,
                travel_time,
                censored,
                np.asarray(bundle["fold"]),
                "hinge",
            )
            step_cv = nested_spatial_cv(
                residual,
                travel_time,
                censored,
                np.asarray(bundle["fold"]),
                "step_hinge",
            )
            bootstrap, curve = bootstrap_breakpoint(
                residual,
                travel_time,
                censored,
                spatial_block,
                args.bootstrap_replicates,
                args.seed + study_index * 10000 + relation_index * 100,
            )
            intervals = bootstrap["intervals"]
            row = {
                "study": study,
                "relation": relation,
                "relation_label": RELATION_LABELS[relation],
                "role": RELATION_ROLE[relation],
                "expected_sign": EXPECTED_SIGN[relation],
                "n": len(residual),
                "coverage_within_40": float(1.0 - censored.mean()),
                "median_nearest_time": float(np.median(travel_time[censored == 0])),
                "selected_step_threshold": float(full_threshold),
                "selected_hinge_threshold": float(hinge_threshold),
                "step_model_bic": float(full_bic),
                "hinge_model_bic": float(hinge_bic),
                "linear_cv_rmse": linear_cv["rmse"],
                "hinge_cv_rmse": hinge_cv["rmse"],
                "step_hinge_cv_rmse": step_cv["rmse"],
                "fold_selected_step_thresholds": json.dumps(step_cv["selected_thresholds"]),
                "fold_selected_hinge_thresholds": json.dumps(hinge_cv["selected_thresholds"]),
                **coefficient,
                "bootstrap_successful": bootstrap["successful_replicates"],
            }
            for name in ("threshold", "step", "pre_slope", "post_slope", "slope_change"):
                row[f"bootstrap_{name}_median"] = intervals[name]["median"]
                row[f"bootstrap_{name}_lower"] = intervals[name]["lower"]
                row[f"bootstrap_{name}_upper"] = intervals[name]["upper"]
            summary_rows.append(row)
            fixed_rows.extend(
                fixed_threshold_rows(
                    study,
                    relation,
                    residual,
                    travel_time,
                    censored,
                    spatial_block,
                )
            )
            curve.insert(0, "relation", relation)
            curve.insert(0, "study", study)
            curve_frames.append(curve)
            binned = binned_residuals(residual, travel_time)
            binned.insert(0, "relation", relation)
            binned.insert(0, "study", study)
            bin_frames.append(binned)
            nuisance_rows.append(
                pd.DataFrame(
                    {
                        "study": study,
                        "relation": relation,
                        "lsoa21cd": bundle["frame"]["lsoa21cd"].astype(str),
                        "fold": bundle["fold"],
                        "observed": bundle["y"],
                        "nuisance_prediction": nuisance_prediction,
                        "adjusted_residual": residual,
                        "nearest_walking_time": travel_time,
                        "right_censored_40": censored.astype(int),
                        "spatial_block": spatial_block,
                    }
                )
            )
            print(
                json.dumps(
                    {
                        "event": "relation_complete",
                        "study": study,
                        "relation": relation,
                        "breakpoint": full_threshold,
                        "step": coefficient["step"],
                        "post_slope": coefficient["post_slope"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    fixed = pd.DataFrame(fixed_rows)
    valid = fixed["valid"].fillna(False)
    fixed.loc[valid, "step_q_bh"] = bh_fdr(
        fixed.loc[valid, "step_p"].to_numpy(dtype=float)
    )
    fixed.loc[valid, "slope_change_q_bh"] = bh_fdr(
        fixed.loc[valid, "slope_change_p"].to_numpy(dtype=float)
    )
    fixed.loc[valid, "post_slope_q_bh"] = bh_fdr(
        fixed.loc[valid, "post_slope_p"].to_numpy(dtype=float)
    )
    summary["evidence"] = summary.apply(evidence_label, axis=1)
    curves = pd.concat(curve_frames, ignore_index=True)
    bins = pd.concat(bin_frames, ignore_index=True)
    nuisance = pd.concat(nuisance_rows, ignore_index=True)

    summary_path = args.output_dir / "breakpoint_summary.csv"
    fixed_path = args.output_dir / "fixed_threshold_tests.csv"
    curve_path = args.output_dir / "breakpoint_curves.csv"
    bins_path = args.output_dir / "binned_adjusted_means.csv"
    nuisance_path = args.output_dir / "relation_adjusted_oof.parquet"
    table_path = args.output_dir / "table_poi_breakpoint_core.tex"
    summary.to_csv(summary_path, index=False)
    fixed.to_csv(fixed_path, index=False)
    curves.to_csv(curve_path, index=False)
    bins.to_csv(bins_path, index=False)
    nuisance.to_parquet(nuisance_path, index=False)
    write_core_latex_table(summary, table_path)
    figures = save_figures(summary, fixed, curves, bins, args.output_dir)
    result = {
        "settings": settings,
        "outputs": {
            "summary": str(summary_path.resolve()),
            "fixed_threshold_tests": str(fixed_path.resolve()),
            "curves": str(curve_path.resolve()),
            "binned_adjusted_means": str(bins_path.resolve()),
            "relation_adjusted_oof": str(nuisance_path.resolve()),
            "core_latex_table": str(table_path.resolve()),
            "figures": figures,
        },
        "headline": summary[
            summary["relation"].isin(CORE_RELATIONS)
        ].to_dict(orient="records"),
    }
    result_path = args.output_dir / "poi_distance_breakpoint_results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "output": str(result_path)}), flush=True)


if __name__ == "__main__":
    main()
