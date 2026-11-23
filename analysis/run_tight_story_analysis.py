from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_origin_threshold_robustness as origin
import run_threshold_robustness as base


OUTCOME_SPECS = {
    "raw_ahah": "Observed AHAH",
    "context_adjusted": "Context-adjusted residual",
    "hfge_adjusted": "HFGE-adjusted residual",
}

INJECTION_SPECS = {
    "aligned_area_point": "Break aligned to area point",
    "postcode_aggregated": "Resident-origin break aggregated by LSOA",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Focused boundary-family analysis and resident-origin injection calibration"
        )
    )
    result.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    result.add_argument("--origin-audit", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--bootstrap-replicates", type=int, default=2000)
    result.add_argument("--simulation-replicates", type=int, default=1000)
    result.add_argument("--spatial-blocks", type=int, default=25)
    result.add_argument("--seed", type=int, default=20260903)
    return result


def block_counts(blocks: np.ndarray, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique), np.full(len(unique), 1.0 / len(unique)), size=replicates
    )
    return unique, counts


def bootstrap_steps_with_counts(
    design: np.ndarray,
    outcomes: np.ndarray,
    blocks: np.ndarray,
    unique_blocks: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_xtx = np.stack(
        [design[blocks == block].T @ design[blocks == block] for block in unique_blocks]
    )
    block_xty = np.stack(
        [design[blocks == block].T @ outcomes[blocks == block] for block in unique_blocks]
    )
    xtx = np.einsum("bg,gij->bij", counts, block_xtx, optimize=True)
    xty = np.einsum("bg,gim->bim", counts, block_xty, optimize=True)
    beta = np.einsum(
        "bij,bjm->bim", np.linalg.pinv(xtx, rcond=1e-10), xty, optimize=True
    )
    full_beta = np.linalg.pinv(design.T @ design, rcond=1e-10) @ design.T @ outcomes
    return full_beta[2], beta[:, 2, :]


def focused_outcomes(bundle: dict) -> tuple[list[str], np.ndarray]:
    names = list(OUTCOME_SPECS)
    values = np.column_stack(
        (
            bundle["observed"],
            bundle["residuals"]["pooled_context_hgb"],
            bundle["residuals"]["hfge"],
        )
    )
    return names, values


def run_boundary_family_scan(
    studies: dict[str, dict],
    exposures: dict,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    for study_index, study in enumerate(base.STUDIES):
        bundle = studies[study]
        model_names, full_outcomes = focused_outcomes(bundle)
        for relation_index, relation in enumerate(base.PRIMARY_RELATIONS):
            for origin_index, origin_spec in enumerate(origin.ORIGIN_SPECS):
                full_nearest, full_censored, mask = exposures[study][relation][origin_spec]
                nearest = np.asarray(full_nearest)[mask]
                censored = np.asarray(full_censored)[mask]
                outcomes = full_outcomes[mask]
                blocks = np.asarray(bundle["blocks"])[mask]
                unique, counts = block_counts(
                    blocks,
                    replicates,
                    seed
                    + study_index * 100_000
                    + relation_index * 10_000
                    + origin_index * 100,
                )
                valid_thresholds: list[float] = []
                estimates_by_threshold: list[np.ndarray] = []
                draws_by_threshold: list[np.ndarray] = []
                supports: list[tuple[int, int]] = []
                for threshold in base.THRESHOLDS:
                    below = int(np.sum(nearest <= threshold))
                    above = int(np.sum(nearest > threshold))
                    minimum_side = max(50.0, 0.04 * len(nearest))
                    if below < minimum_side or above < minimum_side:
                        continue
                    design = base.step_design(nearest, censored, threshold)
                    estimates, draws = bootstrap_steps_with_counts(
                        design, outcomes, blocks, unique, counts
                    )
                    valid_thresholds.append(float(threshold))
                    estimates_by_threshold.append(estimates)
                    draws_by_threshold.append(draws)
                    supports.append((below, above))
                if not valid_thresholds:
                    continue
                estimates_matrix = np.stack(estimates_by_threshold)
                draws_array = np.stack(draws_by_threshold, axis=1)
                standard_errors = np.std(draws_array, axis=0, ddof=1)
                for model_index, model in enumerate(model_names):
                    selected = standard_errors[:, model_index] > 0
                    observed_t = np.divide(
                        estimates_matrix[:, model_index],
                        standard_errors[:, model_index],
                        out=np.full(len(valid_thresholds), np.nan),
                        where=selected,
                    )
                    centred_t = np.divide(
                        draws_array[:, :, model_index]
                        - estimates_matrix[None, :, model_index],
                        standard_errors[None, :, model_index],
                        out=np.zeros((replicates, len(valid_thresholds))),
                        where=standard_errors[None, :, model_index] > 0,
                    )
                    null_max = np.max(np.abs(centred_t[:, selected]), axis=1)
                    observed_max = float(np.nanmax(np.abs(observed_t[selected])))
                    omnibus_p = float(
                        (1.0 + np.sum(null_max >= observed_max)) / (replicates + 1.0)
                    )
                    critical = float(np.quantile(null_max, 0.95))
                    peak_index = int(np.nanargmax(np.abs(observed_t)))
                    family_rows.append(
                        {
                            "study": study,
                            "relation": relation,
                            "origin_spec": origin_spec,
                            "origin_label": origin.ORIGIN_LABELS[origin_spec],
                            "outcome_spec": model,
                            "outcome_label": OUTCOME_SPECS[model],
                            "n": int(len(nearest)),
                            "valid_candidate_count": int(len(valid_thresholds)),
                            "candidate_thresholds": ";".join(
                                f"{value:g}" for value in valid_thresholds
                            ),
                            "peak_threshold": valid_thresholds[peak_index],
                            "peak_abs_t": observed_max,
                            "omnibus_max_t_p": omnibus_p,
                        }
                    )
                    for threshold_index, threshold in enumerate(valid_thresholds):
                        threshold_p = float(
                            (
                                1.0
                                + np.sum(null_max >= abs(observed_t[threshold_index]))
                            )
                            / (replicates + 1.0)
                        )
                        estimate = float(estimates_matrix[threshold_index, model_index])
                        se = float(standard_errors[threshold_index, model_index])
                        threshold_rows.append(
                            {
                                "study": study,
                                "relation": relation,
                                "origin_spec": origin_spec,
                                "origin_label": origin.ORIGIN_LABELS[origin_spec],
                                "outcome_spec": model,
                                "outcome_label": OUTCOME_SPECS[model],
                                "threshold": threshold,
                                "n": int(len(nearest)),
                                "n_at_or_below_threshold": supports[threshold_index][0],
                                "n_above_threshold": supports[threshold_index][1],
                                "step": estimate,
                                "bootstrap_se": se,
                                "pointwise_lower": float(
                                    np.quantile(
                                        draws_array[:, threshold_index, model_index], 0.025
                                    )
                                ),
                                "pointwise_upper": float(
                                    np.quantile(
                                        draws_array[:, threshold_index, model_index], 0.975
                                    )
                                ),
                                "max_t_adjusted_p": threshold_p,
                                "simultaneous_lower": estimate - critical * se,
                                "simultaneous_upper": estimate + critical * se,
                                "family_omnibus_p": omnibus_p,
                            }
                        )
    families = pd.DataFrame(family_rows)
    families["omnibus_q_across_primary_care_panels"] = families.groupby(
        ["origin_spec", "outcome_spec"], sort=False
    )["omnibus_max_t_p"].transform(lambda values: base.bh_fdr(values.to_numpy(float)))
    thresholds = pd.DataFrame(threshold_rows)
    thresholds = thresholds.merge(
        families[
            [
                "study",
                "relation",
                "origin_spec",
                "outcome_spec",
                "omnibus_q_across_primary_care_panels",
            ]
        ],
        on=["study", "relation", "origin_spec", "outcome_spec"],
        how="left",
        validate="many_to_one",
    )
    return thresholds, families


def empirical_tail_p(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    indices = np.searchsorted(ordered, np.asarray(values, dtype=float), side="left")
    return (1.0 + len(ordered) - indices) / (len(ordered) + 1.0)


def run_misaligned_injection(
    studies: dict[str, dict],
    audit_path: Path,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(audit_path)
    audit = audit[
        audit["origin_type"].eq("postcode_centres")
        & audit["relation"].isin(base.PRIMARY_RELATIONS)
    ].copy()
    panels: list[dict[str, object]] = []
    attenuation_rows: list[dict[str, object]] = []
    for study_index, study in enumerate(base.STUDIES):
        bundle = studies[study]
        codes = np.asarray(bundle["codes"]).astype(str)
        for relation_index, relation in enumerate(base.PRIMARY_RELATIONS):
            nearest = np.asarray(bundle["nearest"][relation], dtype=float)
            censored = np.asarray(bundle["censored"][relation], dtype=float)
            blocks = np.asarray(bundle["blocks"])
            pseudo_null, _ = base.wild_cluster_base(
                bundle["residuals"]["pooled_context_hgb"],
                nearest,
                censored,
                blocks,
                replicates,
                seed + study_index * 1000 + relation_index * 100,
            )
            relation_audit = (
                audit[audit["study"].eq(study) & audit["relation"].eq(relation)]
                .set_index("lsoa21cd")
                .reindex(codes)
            )
            if relation_audit["share_within_15"].isna().any():
                raise ValueError(f"missing postcode shares for {study}/{relation}")
            signals = {
                "aligned_area_point": (nearest > 15.0).astype(float),
                "postcode_aggregated": 1.0
                - relation_audit["share_within_15"].to_numpy(dtype=float),
            }
            designs = [
                base.step_design(nearest, censored, threshold)
                for threshold in base.THRESHOLDS
            ]
            null_estimates: list[np.ndarray] = []
            null_standard_errors: list[np.ndarray] = []
            for design in designs:
                estimate, standard_error, _ = base.vectorized_cluster_tests(
                    design, pseudo_null, blocks
                )
                null_estimates.append(estimate)
                null_standard_errors.append(standard_error)
            null_estimate_matrix = np.column_stack(null_estimates)
            null_se_matrix = np.column_stack(null_standard_errors)
            null_t = np.divide(
                null_estimate_matrix,
                null_se_matrix,
                out=np.zeros_like(null_estimate_matrix),
                where=null_se_matrix > 0,
            )
            panel = {
                "study": study,
                "relation": relation,
                "nearest": nearest,
                "censored": censored,
                "blocks": blocks,
                "pseudo_null": pseudo_null,
                "designs": designs,
                "signals": signals,
                "null_max": np.max(np.abs(null_t), axis=1),
            }
            panels.append(panel)
            for injection_spec, signal in signals.items():
                coefficient = np.linalg.pinv(designs[2].T @ designs[2], rcond=1e-10) @ (
                    designs[2].T @ signal
                )
                attenuation_rows.append(
                    {
                        "study": study,
                        "relation": relation,
                        "injection_spec": injection_spec,
                        "injection_label": INJECTION_SPECS[injection_spec],
                        "retained_step_fraction_at_15": float(coefficient[2]),
                        "signal_sd": float(np.std(signal, ddof=1)),
                    }
                )

    scenario_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for absolute_step in (0.5, 1.0, 1.5, 2.0):
        for injection_spec in INJECTION_SPECS:
            for direction in (-1.0, 1.0):
                injected_step = direction * absolute_step
                panel_p_values: list[np.ndarray] = []
                panel_peak_15: list[np.ndarray] = []
                panel_step15: list[np.ndarray] = []
                for panel in panels:
                    outcomes = np.asarray(panel["pseudo_null"]) + injected_step * np.asarray(
                        panel["signals"][injection_spec]
                    )[:, None]
                    estimates: list[np.ndarray] = []
                    standard_errors: list[np.ndarray] = []
                    for design in panel["designs"]:
                        estimate, standard_error, _ = base.vectorized_cluster_tests(
                            np.asarray(design), outcomes, np.asarray(panel["blocks"])
                        )
                        estimates.append(estimate)
                        standard_errors.append(standard_error)
                    estimate_matrix = np.column_stack(estimates)
                    se_matrix = np.column_stack(standard_errors)
                    t_matrix = np.divide(
                        estimate_matrix,
                        se_matrix,
                        out=np.zeros_like(estimate_matrix),
                        where=se_matrix > 0,
                    )
                    max_t = np.max(np.abs(t_matrix), axis=1)
                    p_values = empirical_tail_p(np.asarray(panel["null_max"]), max_t)
                    peak_15 = np.argmax(np.abs(t_matrix), axis=1) == 2
                    panel_p_values.append(p_values)
                    panel_peak_15.append(peak_15)
                    panel_step15.append(estimate_matrix[:, 2])
                p_matrix = np.column_stack(panel_p_values)
                q_matrix = base.bh_fdr_rows(p_matrix)
                selected = q_matrix <= 0.05
                peak_15_matrix = np.column_stack(panel_peak_15)
                step15_matrix = np.column_stack(panel_step15)
                correct_and_selected = selected & peak_15_matrix
                selection_count = int(selected.sum())
                scenario_rows.append(
                    {
                        "injection_spec": injection_spec,
                        "injection_label": INJECTION_SPECS[injection_spec],
                        "absolute_resident_level_step": absolute_step,
                        "direction": "negative" if direction < 0 else "positive",
                        "mean_panel_detection_rate": float(np.mean(selected)),
                        "probability_any_panel_detected": float(
                            np.mean(selected.any(axis=1))
                        ),
                        "peak_at_15_rate": float(np.mean(peak_15_matrix)),
                        "peak_at_15_given_detection": (
                            float(correct_and_selected.sum() / selection_count)
                            if selection_count
                            else np.nan
                        ),
                        "mean_estimated_step15_ratio": float(
                            np.mean(step15_matrix / injected_step)
                        ),
                    }
                )
                for panel_index, panel in enumerate(panels):
                    panel_selected = selected[:, panel_index]
                    panel_rows.append(
                        {
                            "study": panel["study"],
                            "relation": panel["relation"],
                            "injection_spec": injection_spec,
                            "injection_label": INJECTION_SPECS[injection_spec],
                            "absolute_resident_level_step": absolute_step,
                            "direction": "negative" if direction < 0 else "positive",
                            "detection_rate": float(np.mean(panel_selected)),
                            "peak_at_15_rate": float(
                                np.mean(peak_15_matrix[:, panel_index])
                            ),
                            "mean_estimated_step15_ratio": float(
                                np.mean(step15_matrix[:, panel_index] / injected_step)
                            ),
                        }
                    )

    null_p = np.column_stack(
        [empirical_tail_p(panel["null_max"], panel["null_max"]) for panel in panels]
    )
    null_q = base.bh_fdr_rows(null_p)
    null_summary = pd.DataFrame(
        [
            {
                "probability_any_panel_detected": float(
                    np.mean((null_q <= 0.05).any(axis=1))
                ),
                "mean_panel_false_selection_rate": float(np.mean(null_q <= 0.05)),
                "replicates": replicates,
            }
        ]
    )
    return (
        pd.DataFrame(scenario_rows),
        pd.DataFrame(panel_rows),
        pd.DataFrame(attenuation_rows),
        null_summary,
    )


def save_injection_figure(
    scenarios: pd.DataFrame,
    attenuation: pd.DataFrame,
    output_dir: Path,
) -> None:
    averaged = (
        scenarios.groupby(
            ["injection_spec", "injection_label", "absolute_resident_level_step"],
            as_index=False,
        )
        .agg(
            detection_rate=("mean_panel_detection_rate", "mean"),
            retained_ratio=("mean_estimated_step15_ratio", "mean"),
        )
        .sort_values("absolute_resident_level_step")
    )
    colours = {
        "aligned_area_point": "#2f5597",
        "postcode_aggregated": "#c55a11",
    }
    markers = {"aligned_area_point": "o", "postcode_aggregated": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    for injection_spec, label in INJECTION_SPECS.items():
        selected = averaged[averaged["injection_spec"].eq(injection_spec)]
        axes[0].plot(
            selected["absolute_resident_level_step"],
            selected["detection_rate"],
            color=colours[injection_spec],
            marker=markers[injection_spec],
            linewidth=2,
            label=label,
        )
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xticks((0.5, 1.0, 1.5, 2.0))
    axes[0].set_xlabel("Injected resident-level step (AHAH points)")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_title("Boundary-family detection")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    attenuation = attenuation.copy()
    attenuation["panel"] = attenuation.apply(
        lambda row: (
            f"{base.STUDY_LABELS[row['study']].replace('Greater ', '')}\n"
            f"{base.RELATION_LABELS[row['relation']]}"
        ),
        axis=1,
    )
    panels = list(dict.fromkeys(attenuation["panel"]))
    x = np.arange(len(panels), dtype=float)
    width = 0.36
    for offset_index, (injection_spec, label) in enumerate(INJECTION_SPECS.items()):
        selected = attenuation[attenuation["injection_spec"].eq(injection_spec)].set_index(
            "panel"
        )
        axes[1].bar(
            x + (offset_index - 0.5) * width,
            selected.reindex(panels)["retained_step_fraction_at_15"],
            width=width,
            color=colours[injection_spec],
            label=label,
        )
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, panels, fontsize=8)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("Fraction retained by area-point step estimate")
    axes[1].set_title("Attenuation at the 15-minute boundary")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Resident-origin aggregation weakens area-point boundary evidence",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure_misaligned_injection.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "figure_misaligned_injection.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    studies = {
        study: base.load_study(
            args.project_root,
            study,
            args.spatial_blocks,
            args.seed + index,
        )
        for index, study in enumerate(base.STUDIES)
    }
    exposures = origin.load_origin_exposures(args.origin_audit, studies)
    thresholds, families = run_boundary_family_scan(
        studies, exposures, args.bootstrap_replicates, args.seed
    )
    scenarios, panels, attenuation, null_summary = run_misaligned_injection(
        studies, args.origin_audit, args.simulation_replicates, args.seed + 1_000_000
    )
    thresholds.to_csv(args.output_dir / "focused_boundary_estimates.csv", index=False)
    families.to_csv(args.output_dir / "focused_boundary_families.csv", index=False)
    scenarios.to_csv(args.output_dir / "misaligned_injection_summary.csv", index=False)
    panels.to_csv(args.output_dir / "misaligned_injection_by_panel.csv", index=False)
    attenuation.to_csv(args.output_dir / "misaligned_injection_attenuation.csv", index=False)
    null_summary.to_csv(args.output_dir / "boundary_family_null_calibration.csv", index=False)
    save_injection_figure(scenarios, attenuation, args.output_dir)

    family_minima = (
        families.groupby(["origin_spec", "outcome_spec"], as_index=False)
        .agg(
            min_omnibus_p=("omnibus_max_t_p", "min"),
            min_omnibus_q=("omnibus_q_across_primary_care_panels", "min"),
            discoveries=(
                "omnibus_q_across_primary_care_panels",
                lambda values: int(np.sum(values <= 0.05)),
            ),
        )
        .to_dict(orient="records")
    )
    scenario_average = (
        scenarios.groupby(["injection_spec", "absolute_resident_level_step"])
        .agg(
            detection_rate=("mean_panel_detection_rate", "mean"),
            retained_ratio=("mean_estimated_step15_ratio", "mean"),
            peak_at_15_given_detection=("peak_at_15_given_detection", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "scope": {
            "observed_outcomes": OUTCOME_SPECS,
            "origin_definitions": origin.ORIGIN_LABELS,
            "candidate_boundaries_minutes": base.THRESHOLDS,
            "boundary_inference": (
                "paired spatial-block bootstrap max-t over candidate boundaries, "
                "then BH across four primary-care city-relation panels"
            ),
            "injection": (
                "15-minute resident-level step represented either by the area-point "
                "indicator or by the within-LSOA postcode share beyond 15 minutes"
            ),
        },
        "family_minima": family_minima,
        "injection_average": scenario_average,
        "null_calibration": null_summary.iloc[0].to_dict(),
    }
    (args.output_dir / "tight_story_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
