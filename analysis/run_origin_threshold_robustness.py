from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_threshold_robustness as base


ORIGIN_SPECS = (
    "representative_point",
    "postcode_median",
    "postcode_p90",
    "all_building_median",
    "residential_building_median",
)
ORIGIN_LABELS = {
    "representative_point": "LSOA representative point",
    "postcode_median": "Postcode-origin median",
    "postcode_p90": "Postcode-origin P90",
    "all_building_median": "All-building-origin median",
    "residential_building_median": "Residential-building-origin median",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Primary-care threshold robustness to within-LSOA origin placement"
    )
    result.add_argument("--project-root", type=Path, required=True)
    result.add_argument("--origin-audit", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--bootstrap-replicates", type=int, default=2000)
    result.add_argument("--spatial-blocks", type=int, default=25)
    result.add_argument("--seed", type=int, default=20260827)
    return result


def load_origin_exposures(
    audit_path: Path,
    studies: dict[str, dict],
) -> dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    audit = pd.read_csv(audit_path)
    audit = audit[
        audit["relation"].isin(base.PRIMARY_RELATIONS)
    ].copy()
    result: dict[
        str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ] = {}
    for study in base.STUDIES:
        result[study] = {}
        bundle = studies[study]
        expected_codes = np.asarray(bundle["codes"]).astype(str)
        for relation in base.PRIMARY_RELATIONS:
            relation_audit = audit[
                audit["study"].eq(study) & audit["relation"].eq(relation)
            ].copy()
            postcode = relation_audit[
                relation_audit["origin_type"].eq("postcode_centres")
            ].set_index("lsoa21cd").reindex(expected_codes)
            if postcode["n_origins"].isna().any():
                raise ValueError(f"missing postcode origins for {study}/{relation}")
            all_buildings = relation_audit[
                relation_audit["origin_type"].eq("all_building_centres_sample")
            ].set_index("lsoa21cd").reindex(expected_codes)
            residential = relation_audit[
                relation_audit["origin_type"].eq(
                    "explicit_residential_centres_sample"
                )
            ].set_index("lsoa21cd").reindex(expected_codes)
            full_mask = np.ones(len(expected_codes), dtype=bool)
            exposures = {
                "representative_point": (
                    np.asarray(bundle["nearest"][relation], dtype=float),
                    np.asarray(bundle["censored"][relation], dtype=float),
                    full_mask,
                ),
                "postcode_median": (
                    postcode["median_capped40_minutes"].to_numpy(dtype=float),
                    postcode["median_right_censored_40"].to_numpy(dtype=float),
                    full_mask,
                ),
                "postcode_p90": (
                    postcode["p90_capped40_minutes"].to_numpy(dtype=float),
                    postcode["p90_right_censored_40"].to_numpy(dtype=float),
                    full_mask,
                ),
                "all_building_median": (
                    all_buildings["median_capped40_minutes"].to_numpy(dtype=float),
                    all_buildings["median_right_censored_40"].to_numpy(dtype=float),
                    all_buildings["n_origins"].notna().to_numpy(),
                ),
                "residential_building_median": (
                    residential["median_capped40_minutes"].to_numpy(dtype=float),
                    residential["median_right_censored_40"].to_numpy(dtype=float),
                    residential["n_origins"].notna().to_numpy(),
                ),
            }
            for origin_spec, (nearest, censored, mask) in exposures.items():
                if not np.isfinite(nearest[mask]).all() or not np.isfinite(
                    censored[mask]
                ).all():
                    raise ValueError(
                        f"non-finite exposure for {study}/{relation}/{origin_spec}"
                    )
                if np.any(nearest[mask] < 0.0) or np.any(
                    nearest[mask] > 40.0 + 1e-8
                ):
                    raise ValueError(
                        f"exposure outside [0, 40] for {study}/{relation}/{origin_spec}"
                    )
            result[study][relation] = exposures
    return result


def run_thresholds(
    studies: dict[str, dict],
    exposures: dict[
        str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_names = list(base.MODEL_LABELS)
    for study_index, study in enumerate(base.STUDIES):
        bundle = studies[study]
        full_outcomes = np.column_stack(
            [bundle["residuals"][model] for model in model_names]
        )
        for relation_index, relation in enumerate(base.PRIMARY_RELATIONS):
            for origin_spec in ORIGIN_SPECS:
                full_nearest, full_censored, mask = exposures[study][relation][
                    origin_spec
                ]
                nearest = full_nearest[mask]
                censored = full_censored[mask]
                outcomes = full_outcomes[mask]
                selected_blocks = np.asarray(bundle["blocks"])[mask]
                for threshold_index, threshold in enumerate(base.THRESHOLDS):
                    below = int(np.sum(nearest <= threshold))
                    above = int(np.sum(nearest > threshold))
                    minimum_side = max(50.0, 0.04 * len(nearest))
                    valid = below >= minimum_side and above >= minimum_side
                    if not valid:
                        for model in model_names:
                            rows.append(
                                {
                                    "study": study,
                                    "relation": relation,
                                    "origin_spec": origin_spec,
                                    "origin_label": ORIGIN_LABELS[origin_spec],
                                    "threshold": threshold,
                                    "model": model,
                                    "model_label": base.MODEL_LABELS[model],
                                    "n": int(len(nearest)),
                                    "n_at_or_below_threshold": below,
                                    "n_above_threshold": above,
                                    "minimum_side_required": minimum_side,
                                    "valid": False,
                                    "right_censored_count": int(np.sum(censored > 0)),
                                    "right_censored_percent": float(
                                        100.0 * np.mean(censored > 0)
                                    ),
                                    "median_exposure_minutes": float(
                                        np.median(nearest[censored == 0])
                                    ),
                                }
                            )
                        continue
                    design = base.step_design(nearest, censored, threshold)
                    # The same seed is used across origin specifications so the
                    # block-resampling comparison is paired.
                    replicate_seed = (
                        seed
                        + study_index * 100_000
                        + relation_index * 1_000
                        + threshold_index
                    )
                    estimates, draws = base.paired_block_bootstrap_steps(
                        design,
                        outcomes,
                        selected_blocks,
                        replicates,
                        replicate_seed,
                    )
                    for model_index, model in enumerate(model_names):
                        cluster = base.cluster_step_summary(
                            design, outcomes[:, model_index], selected_blocks
                        )
                        model_draws = draws[:, model_index]
                        rows.append(
                            {
                                "study": study,
                                "relation": relation,
                                "origin_spec": origin_spec,
                                "origin_label": ORIGIN_LABELS[origin_spec],
                                "threshold": threshold,
                                "model": model,
                                "model_label": base.MODEL_LABELS[model],
                                "n": int(len(nearest)),
                                "n_at_or_below_threshold": below,
                                "n_above_threshold": above,
                                "minimum_side_required": minimum_side,
                                "valid": True,
                                "right_censored_count": int(np.sum(censored > 0)),
                                "right_censored_percent": float(
                                    100.0 * np.mean(censored > 0)
                                ),
                                "median_exposure_minutes": float(
                                    np.median(nearest[censored == 0])
                                ),
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
                                "block_bootstrap_p": base.bootstrap_probability(
                                    model_draws, float(estimates[model_index])
                                ),
                                **cluster,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame["q_within_origin_family"] = frame.groupby(
        ["model", "origin_spec"], sort=False
    )["block_bootstrap_p"].transform(
        lambda values: base.bh_fdr(values.to_numpy(float))
    )
    frame["q_joint_origin_family"] = frame.groupby("model", sort=False)[
        "block_bootstrap_p"
    ].transform(lambda values: base.bh_fdr(values.to_numpy(float)))
    return frame


def add_effect_bounds(
    thresholds: pd.DataFrame, studies: dict[str, dict]
) -> pd.DataFrame:
    city_sd = {
        study: float(np.std(bundle["observed"], ddof=1))
        for study, bundle in studies.items()
    }
    rows: list[pd.DataFrame] = []
    for fraction in (0.10, 0.20, 0.30):
        current = thresholds.copy()
        current["sesoi_sd"] = fraction
        current["city_ahah_sd"] = current["study"].map(city_sd)
        current["sesoi_ahah_points"] = fraction * current["city_ahah_sd"]
        current["largest_compatible_abs_effect"] = np.maximum(
            current["block_bootstrap_lower"].abs(),
            current["block_bootstrap_upper"].abs(),
        )
        current["effect_at_or_above_bound_excluded"] = (
            current["block_bootstrap_lower"].gt(-current["sesoi_ahah_points"])
            & current["block_bootstrap_upper"].lt(current["sesoi_ahah_points"])
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def save_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    hfge = frame[frame["model"].eq("hfge")].copy()
    colors = {
        "representative_point": "#5f6368",
        "postcode_median": "#1f77b4",
        "postcode_p90": "#d95f02",
        "all_building_median": "#2ca02c",
        "residential_building_median": "#9467bd",
    }
    offsets = {
        "representative_point": -0.32,
        "postcode_median": -0.16,
        "postcode_p90": 0.0,
        "all_building_median": 0.16,
        "residential_building_median": 0.32,
    }
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 7.8), sharex=True, sharey=True, layout="constrained"
    )
    for row_index, study in enumerate(base.STUDIES):
        for column_index, relation in enumerate(base.PRIMARY_RELATIONS):
            axis = axes[row_index, column_index]
            panel = hfge[
                hfge["study"].eq(study) & hfge["relation"].eq(relation)
            ]
            for origin_spec in ORIGIN_SPECS:
                selected = panel[panel["origin_spec"].eq(origin_spec)].sort_values(
                    "threshold"
                )
                x = np.arange(len(base.THRESHOLDS)) + offsets[origin_spec]
                axis.errorbar(
                    x,
                    selected["step"],
                    yerr=np.vstack(
                        (
                            selected["step"] - selected["block_bootstrap_lower"],
                            selected["block_bootstrap_upper"] - selected["step"],
                        )
                    ),
                    color=colors[origin_spec],
                    marker="o",
                    linewidth=1.5,
                    capsize=2.5,
                    label=ORIGIN_LABELS[origin_spec],
                )
            axis.axhline(0.0, color="#555555", linewidth=0.8)
            axis.grid(axis="y", color="#e5e5e5", linewidth=0.7)
            axis.set_title(
                f"{base.STUDY_LABELS[study]} — {base.RELATION_LABELS[relation]}"
            )
            axis.set_xticks(
                np.arange(len(base.THRESHOLDS)),
                [f"{int(value)}" for value in base.THRESHOLDS],
            )
            if row_index == 1:
                axis.set_xlabel("Candidate walking-time boundary (minutes)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.supylabel("Immediate residual step (AHAH points)")
    fig.suptitle(
        "HFGE threshold estimates under alternative within-LSOA origins",
        fontweight="bold",
    )
    fig.savefig(output_dir / "figure_origin_threshold_hfge.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "figure_origin_threshold_hfge.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_summary(
    thresholds: pd.DataFrame,
    effects: pd.DataFrame,
    replicates: int,
) -> dict[str, object]:
    discoveries = (
        thresholds.groupby(["origin_spec", "model"], as_index=False)
        .agg(
            valid_tests=("valid", "sum"),
            discoveries_within_origin=(
                "q_within_origin_family",
                lambda x: int((x <= 0.05).sum()),
            ),
            discoveries_joint_origins=(
                "q_joint_origin_family",
                lambda x: int((x <= 0.05).sum()),
            ),
            minimum_q_within_origin=("q_within_origin_family", "min"),
            minimum_q_joint_origins=("q_joint_origin_family", "min"),
        )
    )
    hfge_15 = thresholds[
        thresholds["model"].eq("hfge") & thresholds["threshold"].eq(15.0)
    ][
        [
            "study",
            "relation",
            "origin_spec",
            "step",
            "block_bootstrap_lower",
            "block_bootstrap_upper",
            "block_bootstrap_p",
            "valid",
            "q_within_origin_family",
            "q_joint_origin_family",
        ]
    ]
    bound_counts = (
        effects[effects["model"].eq("hfge")]
        .groupby(["origin_spec", "sesoi_sd"], as_index=False)[
            "effect_at_or_above_bound_excluded"
        ]
        .sum()
    )
    return {
        "design": {
            "relations": list(base.PRIMARY_RELATIONS),
            "cities": list(base.STUDIES),
            "thresholds_minutes": list(base.THRESHOLDS),
            "models": base.MODEL_LABELS,
            "origin_specifications": ORIGIN_LABELS,
            "bootstrap_replicates": replicates,
            "spatial_blocks": 25,
            "fdr_families": {
                "q_within_origin_family": "valid primary-care cells within each model and origin specification",
                "q_joint_origin_family": "all valid primary-care cells across the five origin specifications within each model",
            },
        },
        "discoveries": discoveries.to_dict(orient="records"),
        "hfge_15_minute": hfge_15.to_dict(orient="records"),
        "hfge_effect_bound_counts": bound_counts.to_dict(orient="records"),
    }


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    studies = {
        study: base.load_study(
            args.project_root,
            study,
            args.spatial_blocks,
            args.seed + study_index,
        )
        for study_index, study in enumerate(base.STUDIES)
    }
    exposures = load_origin_exposures(args.origin_audit, studies)
    thresholds = run_thresholds(
        studies, exposures, args.bootstrap_replicates, args.seed
    )
    effects = add_effect_bounds(thresholds, studies)
    thresholds.to_csv(
        args.output_dir / "multimodel_origin_threshold_tests.csv", index=False
    )
    effects.to_csv(args.output_dir / "origin_threshold_effect_bounds.csv", index=False)
    save_figure(thresholds, args.output_dir)
    summary = make_summary(thresholds, effects, args.bootstrap_replicates)
    (args.output_dir / "origin_threshold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
