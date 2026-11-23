from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from run_spatial_residual_diagnostics import (
    LISA_COLORS,
    LISA_LABELS,
    global_moran,
    local_moran,
    semivariogram,
    weights_for,
)


STUDIES = ("greater_london", "greater_manchester")
STUDY_LABELS = {
    "greater_london": "Greater London",
    "greater_manchester": "Greater Manchester",
}
MODEL_LABELS = {
    "pooled_hgb": "Two-city pooled HGB",
    "gravity_hgb": "Continuous Gravity–HGB",
    "nested_blend": "Nested Gravity HGB–TabPFN",
}
MODEL_COLORS = {
    "pooled_hgb": "#7570b3",
    "gravity_hgb": "#d95f02",
    "nested_blend": "#1b9e77",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create publication-ready baseline, residual, and distance figures"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/publication_outputs")
    )
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--semivariogram-pairs", type=int, default=500_000)
    parser.add_argument("--semivariogram-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(macro: dict, name: str) -> float:
    return float(macro[name]["mean"])


def baseline_rows() -> list[dict]:
    tabular = read_json("results/final_baselines.json")
    graph = read_json("results/gnn_baselines.json")
    gravity_graph = read_json("results/gravity_gnn_real.json")
    facility_gnn = read_json("results/facility_gravity_gnn_screen.json")
    facility_hybrid = read_json("results/facility_graph_hybrid.json")
    distance = read_json("results/distance_effect/distance_effect_results.json")
    tabpfn = read_json("results/tabpfn_gravity/screen_v2_e1.json")
    nested = read_json("results/tabpfn_gravity/nested_blend_v2_e1.json")

    specifications = [
        (
            "Conventional baseline",
            "Mean",
            "Training-fold mean",
            lambda city: tabular["within_city"][city]["mean"]["macro"],
        ),
        (
            "Conventional baseline",
            "Ridge",
            "Linear baseline",
            lambda city: tabular["within_city"][city]["ridge"]["macro"],
        ),
        (
            "Conventional baseline",
            "Random Forest",
            "Nonlinear tabular baseline",
            lambda city: tabular["within_city"][city]["random_forest"]["macro"],
        ),
        (
            "Conventional baseline",
            "Local HGB",
            "Strong single-city baseline",
            lambda city: tabular["within_city"][city]["local_hgb"]["macro"],
        ),
        (
            "Graph baseline",
            "Vanilla GCN",
            "Queen graph without distance",
            lambda city: graph["within_city"][city]["models"]["gcn"][
                "ensemble_macro"
            ],
        ),
        (
            "Graph baseline",
            "GAT",
            "Queen graph attention",
            lambda city: graph["within_city"][city]["models"]["gat"][
                "ensemble_macro"
            ],
        ),
        (
            "Gravity graph ablation",
            "Fixed Gravity-GNN",
            "Fixed Queen-edge gravity kernel",
            lambda city: gravity_graph["within_city"][city]["models"][
                "fixed_gravity"
            ]["ensemble_macro"],
        ),
        (
            "Gravity graph ablation",
            "Learned GEL-GNN",
            "Learned Queen-edge gravity kernel",
            lambda city: gravity_graph["within_city"][city]["models"]["gel_gnn"][
                "ensemble_macro"
            ],
        ),
        (
            "Strong baseline",
            "Two-city pooled HGB",
            "Cross-city tabular baseline",
            lambda city: tabular["within_city"][city]["proposed_pooled_hgb"][
                "macro"
            ],
        ),
        (
            "Facility graph ablation",
            "Facility multi-relation Gravity-GNN",
            "One-seed facility graph screen",
            lambda city: facility_gnn["within_city"][city]["ensemble_macro"],
        ),
        (
            "Alternative hybrid",
            "Facility gravity nested hybrid",
            "Nested facility-feature HGB",
            lambda city: facility_hybrid["within_city"][city]["macro"],
        ),
        (
            "Proposed primary",
            "Continuous Gravity–HGB",
            "Scale-15 continuous facility gravity",
            lambda city: distance["studies"][city]["variants"]["gravity_scale_15"][
                "macro"
            ],
        ),
        (
            "TabPFN ablation",
            "Continuous Gravity–TabPFN v2",
            "One-member TabPFN screen",
            lambda city: tabpfn["studies"][city]["variants"]["gravity_scale_15"][
                "macro"
            ],
        ),
        (
            "Extended candidate",
            "Nested Gravity HGB–TabPFN",
            "Inner-fold selected blend",
            lambda city: nested["studies"][city]["macro"],
        ),
    ]

    rows = []
    for category, model, role, getter in specifications:
        london = getter("greater_london")
        manchester = getter("greater_manchester")
        rows.append(
            {
                "Category": category,
                "Model": model,
                "London R2": metric(london, "r2"),
                "London RMSE": metric(london, "rmse"),
                "Manchester R2": metric(manchester, "r2"),
                "Manchester RMSE": metric(manchester, "rmse"),
                "Role": role,
            }
        )
    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [
        "Model",
        "London R²",
        "London RMSE",
        "Manchester R²",
        "Manchester RMSE",
        "Role",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        name = row.Model
        if row.Category in {"Proposed primary", "Extended candidate"}:
            name = f"**{name}**"
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{row._2:.3f}",
                    f"{row._3:.3f}",
                    f"{row._4:.3f}",
                    f"{row._5:.3f}",
                    row.Role,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def save_baseline_table(output_dir: Path) -> dict[str, str]:
    frame = pd.DataFrame(baseline_rows())
    csv_path = output_dir / "table1_baseline_comparison.csv"
    md_path = output_dir / "table1_baseline_comparison.md"
    tex_path = output_dir / "table1_baseline_comparison.tex"
    frame.to_csv(csv_path, index=False, float_format="%.6f")
    md_path.write_text(markdown_table(frame), encoding="utf-8")
    latex = frame[
        [
            "Model",
            "London R2",
            "London RMSE",
            "Manchester R2",
            "Manchester RMSE",
            "Role",
        ]
    ].to_latex(index=False, float_format="%.3f", escape=True)
    tex_path.write_text(latex, encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path)}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 320,
        }
    )


def distance_effect_figure(output_dir: Path) -> dict[str, str]:
    frame = pd.read_csv("results/distance_effect/performance.csv")
    scales = np.asarray([5, 10, 15, 20, 30, 40], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), constrained_layout=True)
    for column, study in enumerate(STUDIES):
        city = frame[frame["study"] == study].set_index("variant")
        for row, metric_name in enumerate(("r2", "rmse")):
            axis = axes[row, column]
            continuous = np.asarray(
                [city.loc[f"gravity_scale_{int(scale)}", metric_name] for scale in scales]
            )
            axis.axvspan(15, 20, color="#f1f1f1", zorder=0)
            axis.plot(
                scales,
                continuous,
                color="#1b9e77",
                marker="o",
                linewidth=2.0,
                markersize=5,
                label="Continuous gravity",
                zorder=3,
            )
            for threshold, marker, color in (
                (15, "D", "#d95f02"),
                (20, "s", "#7570b3"),
            ):
                axis.scatter(
                    threshold,
                    city.loc[f"threshold_{threshold}", metric_name],
                    marker=marker,
                    s=58,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                    label=f"Hard {threshold}-min threshold",
                )
            axis.scatter(
                17.5,
                city.loc["segmented_15_20", metric_name],
                marker="^",
                s=64,
                color="#e6ab02",
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
                label="Segmented 15/20-min",
            )
            axis.axhline(
                city.loc["context_only", metric_name],
                color="#666666",
                linestyle="--",
                linewidth=1.2,
                label="Context only",
            )
            axis.axhline(
                city.loc["no_distance_40", metric_name],
                color="#999999",
                linestyle=":",
                linewidth=1.4,
                label="No-distance facility counts",
            )
            if row == 0:
                best_index = int(np.argmax(continuous))
                better = "max"
            else:
                best_index = int(np.argmin(continuous))
                better = "min"
            axis.annotate(
                f"Best continuous: {int(scales[best_index])} min",
                xy=(scales[best_index], continuous[best_index]),
                xytext=(6, -16 if better == "max" else 10),
                textcoords="offset points",
                fontsize=8.5,
                color="#1b9e77",
            )
            axis.set_xlim(3, 42)
            axis.set_xticks(scales)
            axis.grid(axis="y", color="#dddddd", linewidth=0.7)
            axis.set_xlabel("Gravity decay scale (minutes)")
            axis.set_ylabel("Macro spatial-CV R²" if row == 0 else "Macro spatial-CV RMSE")
            if row == 0:
                axis.set_title(STUDY_LABELS[study])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Hard travel-time thresholds versus continuous gravity decay", fontsize=13)
    png = output_dir / "figure2_distance_representation.png"
    pdf = output_dir / "figure2_distance_representation.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def merge_nested(frame: gpd.GeoDataFrame, study: str) -> gpd.GeoDataFrame:
    nested = pd.read_csv(f"results/tabpfn_gravity/nested_blend_{study}.csv")
    columns = ["lsoa21cd", "prediction_nested_blend", "residual_nested_blend"]
    output = frame.merge(nested[columns], on="lsoa21cd", validate="one_to_one")
    if len(output) != len(frame):
        raise ValueError(f"nested blend row mismatch for {study}")
    return output


def plot_moran(moran_frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    order = ("pooled_hgb", "gravity_hgb", "nested_blend")
    y = np.arange(len(order))
    for axis, study in zip(axes, STUDIES, strict=True):
        city = moran_frame[moran_frame["study"] == study].set_index("model")
        values = np.asarray([city.loc[model, "I"] for model in order])
        for position, (model, value) in enumerate(zip(order, values, strict=True)):
            axis.hlines(position, 0, value, color="#cfcfcf", linewidth=2)
            axis.scatter(
                value,
                position,
                color=MODEL_COLORS[model],
                s=62,
                zorder=3,
            )
            axis.text(value + 0.008, position, f"{value:.3f}", va="center", fontsize=9)
        axis.set_yticks(y, [MODEL_LABELS[model] for model in order])
        axis.invert_yaxis()
        axis.set_xlim(0, max(0.62, values.max() + 0.08))
        axis.set_xlabel("Global Moran's I of OOF residuals")
        axis.set_title(STUDY_LABELS[study])
        axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    fig.suptitle("Residual spatial autocorrelation under Queen contiguity", fontsize=13)
    png = output_dir / "figure3_residual_global_moran.png"
    pdf = output_dir / "figure3_residual_global_moran.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def plot_lisa(frames: dict[str, gpd.GeoDataFrame], output_dir: Path) -> dict[str, str]:
    models = ("pooled_hgb", "gravity_hgb", "nested_blend")
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.1), constrained_layout=True)
    for row, study in enumerate(STUDIES):
        frame = frames[study]
        for column, model in enumerate(models):
            axis = axes[row, column]
            labels = frame[f"{model}_lisa_cluster"]
            for label in ("NS", "HH", "LL", "HL", "LH"):
                selected = labels == label
                if selected.any():
                    frame.loc[selected].plot(
                        ax=axis,
                        color=LISA_COLORS[label],
                        linewidth=0.035,
                        edgecolor="#777777",
                    )
            if row == 0:
                axis.set_title(MODEL_LABELS[model])
            if column == 0:
                axis.text(
                    -0.04,
                    0.5,
                    STUDY_LABELS[study],
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=11,
                )
            axis.set_axis_off()
    handles = [
        Patch(facecolor=LISA_COLORS[label], label=label)
        for label in ("HH", "LL", "HL", "LH", "NS")
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Local Moran residual clusters (Queen weights; BH-FDR q≤0.05)", fontsize=13)
    png = output_dir / "figure4_residual_lisa.png"
    pdf = output_dir / "figure4_residual_lisa.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def plot_semivariogram(rows: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for axis, study in zip(axes, STUDIES, strict=True):
        city = rows[rows["study"] == study]
        for model in ("pooled_hgb", "gravity_hgb", "nested_blend"):
            selected = city[city["model"] == model]
            axis.plot(
                selected["distance_mean_m"] / 1000.0,
                selected["semivariance"],
                color=MODEL_COLORS[model],
                marker="o",
                markersize=3,
                linewidth=1.7,
                label=MODEL_LABELS[model],
            )
        axis.set_title(STUDY_LABELS[study])
        axis.set_xlabel("LSOA centroid separation (km)")
        axis.set_ylabel("Residual semivariance (AHAH²)")
        axis.grid(color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.suptitle("Empirical semivariograms of out-of-fold residuals", fontsize=13)
    png = output_dir / "figure5_residual_semivariogram.png"
    pdf = output_dir / "figure5_residual_semivariogram.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def residual_outputs(args: argparse.Namespace) -> dict:
    old_moran = pd.read_csv("results/spatial_diagnostics/global_moran.csv")
    frames: dict[str, gpd.GeoDataFrame] = {}
    moran_rows = []
    semivariogram_rows = []
    lisa_counts = []
    old_names = {
        "pooled_hgb": "pooled_hgb",
        "gravity_hgb": "facility_gravity_scale15",
    }
    for study_index, study in enumerate(STUDIES):
        frame = gpd.read_parquet(
            f"results/spatial_diagnostics/{study}_lisa.parquet"
        )
        frame = merge_nested(frame, study)
        bundle = weights_for(frame, 5000.0)
        for model, old_name in old_names.items():
            row = old_moran[
                (old_moran["study"] == study)
                & (old_moran["weights"] == "queen")
                & (old_moran["model"] == old_name)
            ].iloc[0].to_dict()
            row["model"] = model
            moran_rows.append(row)
        nested_moran = global_moran(
            frame["residual_nested_blend"].to_numpy(dtype=float),
            bundle["queen"],
            args.permutations,
            args.seed + study_index * 1000 + 900,
        )
        moran_rows.append(
            {
                "study": study,
                "weights": "queen",
                "model": "nested_blend",
                **nested_moran,
            }
        )

        existing_columns = {
            "pooled_hgb": "pooled_hgb_lisa_cluster",
            "gravity_hgb": "facility_gravity_scale15_lisa_cluster",
        }
        for model, column in existing_columns.items():
            frame[f"{model}_lisa_cluster"] = frame[column]
        nested_lisa = local_moran(
            frame["residual_nested_blend"].to_numpy(dtype=float),
            bundle["queen"],
            args.permutations,
            args.seed + study_index * 1000 + 950,
        )
        frame["nested_blend_lisa_cluster"] = [
            LISA_LABELS[int(value)] for value in nested_lisa["cluster_code"]
        ]
        frame["nested_blend_lisa_I"] = nested_lisa["local_I"]
        frame["nested_blend_lisa_p"] = nested_lisa["p_permutation"]
        frame["nested_blend_lisa_q_bh"] = nested_lisa["q_bh"]
        for model in ("pooled_hgb", "gravity_hgb", "nested_blend"):
            counts = frame[f"{model}_lisa_cluster"].value_counts()
            lisa_counts.append(
                {
                    "study": study,
                    "model": model,
                    **{
                        label: int(counts.get(label, 0))
                        for label in ("HH", "LL", "HL", "LH", "NS")
                    },
                }
            )
        frames[study] = frame
        frame.to_parquet(args.output_dir / f"residual_lisa_{study}.parquet", index=False)

        residual_map = {
            "pooled_hgb": frame["residual_proposed_pooled_hgb"].to_numpy(dtype=float),
            "gravity_hgb": frame["residual_facility_gravity_scale15"].to_numpy(
                dtype=float
            ),
            "nested_blend": frame["residual_nested_blend"].to_numpy(dtype=float),
        }
        semivariogram_rows.extend(
            {"study": study, **row}
            for row in semivariogram(
                bundle["coordinates"],
                residual_map,
                args.semivariogram_pairs,
                args.semivariogram_bins,
                args.seed + study_index,
            )
        )

    moran_frame = pd.DataFrame(moran_rows)
    moran_path = args.output_dir / "residual_global_moran.csv"
    moran_frame.to_csv(moran_path, index=False)
    lisa_count_frame = pd.DataFrame(lisa_counts)
    lisa_count_path = args.output_dir / "residual_lisa_counts.csv"
    lisa_count_frame.to_csv(lisa_count_path, index=False)
    semivariogram_frame = pd.DataFrame(semivariogram_rows)
    semivariogram_path = args.output_dir / "residual_semivariogram.csv"
    semivariogram_frame.to_csv(semivariogram_path, index=False)
    return {
        "moran_data": str(moran_path),
        "moran_figure": plot_moran(moran_frame, args.output_dir),
        "lisa_counts": str(lisa_count_path),
        "lisa_figure": plot_lisa(frames, args.output_dir),
        "semivariogram_data": str(semivariogram_path),
        "semivariogram_figure": plot_semivariogram(
            semivariogram_frame, args.output_dir
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    outputs = {
        "baseline_table": save_baseline_table(args.output_dir),
        "distance_representation": distance_effect_figure(args.output_dir),
        "residual_diagnostics": residual_outputs(args),
        "settings": {
            "permutations": args.permutations,
            "semivariogram_pairs": args.semivariogram_pairs,
            "semivariogram_bins": args.semivariogram_bins,
            "residuals": "strict out-of-fold observed minus prediction",
            "primary_metric": "macro mean over five frozen contiguous spatial folds",
        },
    }
    manifest = args.output_dir / "publication_outputs_manifest.json"
    manifest.write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "manifest": str(manifest)}), flush=True)


if __name__ == "__main__":
    main()
