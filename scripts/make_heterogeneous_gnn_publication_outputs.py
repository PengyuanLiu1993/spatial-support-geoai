from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

STUDIES = ("greater_london", "greater_manchester")
STUDY_LABELS = {
    "greater_london": "Greater London",
    "greater_manchester": "Greater Manchester",
}
RESIDUAL_MODELS = (
    "context_hgb",
    "aggregated_gravity_hgb",
    "gated_gravity_gnn",
    "proposed_ensemble",
)
MODEL_LABELS = {
    "context_hgb": "Context-HGB",
    "aggregated_gravity_hgb": "Continuous Gravity-HGB",
    "gated_gravity_gnn": "Gated-prior facility-GNN hybrid",
    "proposed_ensemble": "HFGE (50:50)",
}
MODEL_COLORS = {
    "context_hgb": "#7570b3",
    "aggregated_gravity_hgb": "#d95f02",
    "gated_gravity_gnn": "#1b9e77",
    "proposed_ensemble": "#1f78b4",
}
LISA_LABELS = {0: "NS", 1: "HH", 2: "LH", 3: "LL", 4: "HL"}
LISA_COLORS = {
    "NS": "#d9d9d9",
    "HH": "#b2182b",
    "LL": "#2166ac",
    "HL": "#ef8a62",
    "LH": "#67a9cf",
}

# Keep the dense LSOA tessellation visible as geographic context without
# competing with the statistically significant LISA cluster fills.
LSOA_BOUNDARY_COLOR = (0.47, 0.47, 0.47, 0.22)
LSOA_BOUNDARY_WIDTH = 0.025


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create updated heterogeneous-GNN publication outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication_outputs_heterogeneous_gnn"),
    )
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--semivariogram-pairs", type=int, default=500_000)
    parser.add_argument("--semivariogram-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--reuse-residual-data",
        action="store_true",
        help="Redraw figures from existing Moran, LISA, and semivariogram outputs",
    )
    return parser


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def metric(macro: dict, name: str) -> float:
    return float(macro[name]["mean"])


def baseline_rows() -> list[dict[str, object]]:
    tabular = read_json("results/final_baselines.json")
    graph = read_json("results/gnn_baselines.json")
    updated = read_json("results/heterogeneous_graph_ensemble.json")
    nested = read_json("results/tabpfn_gravity/nested_blend_v2_e5.json")

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
            "Linear tabular baseline",
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
            "Single-city HGB",
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
    ]
    updated_models = (
        ("Strong baseline", "Context-HGB", "context_hgb", "No facility inputs"),
        (
            "Aggregated alternative",
            "Continuous Gravity-HGB",
            "aggregated_gravity_hgb_scale15",
            "Scale-15 gravity table features",
        ),
        (
            "TabPFN baseline",
            "Context-only TabPFN",
            "context_tabpfn",
            "One-estimator context model",
        ),
        (
            "Graph ablation",
            "Hard Gravity facility-GNN hybrid",
            "heterogeneous_facility_hard_gravity_gnn_hybrid",
            "Full monotone kernel",
        ),
        (
            "Graph ablation",
            "No-Gravity facility-GNN hybrid",
            "heterogeneous_facility_no_gravity_gnn_hybrid",
            "Edge-conditioned time model",
        ),
        (
            "Graph candidate",
            "Gated Gravity facility-GNN hybrid",
            "heterogeneous_facility_gated_gravity_gnn_hybrid",
            "Relation-vector messages and soft gravity prior",
        ),
    )
    for category, label, key, role in updated_models:
        specifications.append(
            (
                category,
                label,
                role,
                lambda city, model_key=key: updated["studies"][city]["models"][
                    model_key
                ]["macro"],
            )
        )
    specifications.extend(
        [
            (
                "Previous extended candidate",
                "Nested Gravity HGB-TabPFN",
                "Uses aggregated gravity inputs",
                lambda city: nested["studies"][city]["macro"],
            ),
            (
                "Proposed development candidate",
                "Context TabPFN + gated facility Gravity-GNN ensemble",
                "Fixed 50:50; no aggregated gravity inputs",
                lambda city: updated["studies"][city]["models"][
                    "proposed_fixed_equal_ensemble"
                ]["macro"],
            ),
        ]
    )

    rows: list[dict[str, object]] = []
    for category, model, role, getter in specifications:
        london = getter("greater_london")
        manchester = getter("greater_manchester")
        rows.append(
            {
                "Category": category,
                "Model": model,
                "London R2": metric(london, "r2"),
                "London RMSE": metric(london, "rmse"),
                "London MAE": metric(london, "mae"),
                "Manchester R2": metric(manchester, "r2"),
                "Manchester RMSE": metric(manchester, "rmse"),
                "Manchester MAE": metric(manchester, "mae"),
                "Role": role,
            }
        )
    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Model | London R² | London RMSE | Manchester R² | Manchester RMSE | Role |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        model = row.Model
        if row.Category == "Proposed development candidate":
            model = f"**{model}**"
        lines.append(
            f"| {model} | {row._2:.3f} | {row._3:.3f} | "
            f"{row._5:.3f} | {row._6:.3f} | {row.Role} |"
        )
    return "\n".join(lines) + "\n"


def save_baseline_table(output_dir: Path) -> dict[str, str]:
    frame = pd.DataFrame(baseline_rows())
    csv_path = output_dir / "table1_updated_baseline_comparison.csv"
    md_path = output_dir / "table1_updated_baseline_comparison.md"
    tex_path = output_dir / "table1_updated_baseline_comparison.tex"
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


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def distance_performance_figure(output_dir: Path) -> dict[str, str]:
    frame = pd.read_csv("results/distance_effect/performance.csv")
    scales = np.asarray([5, 10, 15, 20, 30, 40], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), layout="constrained")
    for column, study in enumerate(STUDIES):
        city = frame[frame["study"] == study].set_index("variant")
        for row, metric_name in enumerate(("r2", "rmse")):
            axis = axes[row, column]
            continuous = np.asarray(
                [city.loc[f"gravity_scale_{int(scale)}", metric_name] for scale in scales]
            )
            axis.axvspan(15, 20, color="#eeeeee", zorder=0)
            axis.plot(
                scales,
                continuous,
                color="#1b9e77",
                marker="o",
                linewidth=2,
                markersize=5,
                label="Continuous gravity",
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
            best = int(np.argmax(continuous) if metric_name == "r2" else np.argmin(continuous))
            axis.margins(y=0.18)
            vertical_offset = 12 if metric_name == "r2" else -18
            axis.annotate(
                f"Best continuous: {int(scales[best])} min",
                (scales[best], continuous[best]),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                fontsize=9.5,
                color="#1b9e77",
                ha="center",
                va="bottom" if metric_name == "r2" else "top",
                bbox={
                    "boxstyle": "square,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
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
        columnspacing=1.8,
        handletextpad=0.6,
    )
    fig.suptitle("Travel-time representation and spatial-CV performance", fontsize=14)
    return save_figure(fig, output_dir / "figure2a_distance_representation_performance")


def distance_shape_figure(output_dir: Path) -> dict[str, str]:
    time = np.linspace(0.0, 40.0, 401)
    continuous = np.exp(-np.power(time / 15.0, 1.5))
    hard_15 = (time <= 15.0).astype(float)
    hard_20 = (time <= 20.0).astype(float)
    no_distance = np.ones_like(time)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), layout="constrained")
    axis = axes[0]
    axis.plot(time, continuous, color="#1b9e77", linewidth=2.2, label="Continuous gravity (scale 15)")
    axis.step(time, hard_15, where="post", color="#d95f02", linewidth=1.8, label="Hard 15-min")
    axis.step(time, hard_20, where="post", color="#7570b3", linewidth=1.8, label="Hard 20-min")
    axis.plot(time, no_distance, color="#777777", linestyle=":", linewidth=1.5, label="No distance within 40 min")
    axis.axvspan(15, 20, color="#eeeeee", zorder=0)
    axis.set_xlabel("Walking travel time (minutes)")
    axis.set_ylabel("Opportunity weight")
    axis.set_xlim(0, 40)
    axis.set_ylim(-0.03, 1.05)
    axis.grid(color="#dddddd", linewidth=0.7)
    axis.set_title("Alternative distance-response functions")
    handles, labels = axis.get_legend_handles_labels()

    axis = axes[1]
    intervals = ((0, 15, "≤15", "#1b9e77"), (15, 20, "15–20", "#e6ab02"), (20, 40, ">20–40", "#d95f02"))
    for row, (lower, upper, label, color) in enumerate(intervals):
        axis.barh(row, upper - lower, left=lower, height=0.58, color=color, alpha=0.85)
        axis.text((lower + upper) / 2, row, label, ha="center", va="center", color="white", fontsize=9)
    axis.set_yticks(range(3), ["Near basis", "Intermediate basis", "Distant basis"])
    axis.set_xlabel("Walking travel time (minutes)")
    axis.set_xlim(0, 40)
    axis.set_title("Segmented 15/20-minute representation")
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.6,
    )
    fig.suptitle("How facility distance is encoded", fontsize=14)
    return save_figure(fig, output_dir / "figure2b_distance_encoding_shapes")


def load_residual_frame(study: str) -> gpd.GeoDataFrame:
    reference = gpd.read_parquet(f"results/distance_effect/oof/{study}.parquet")
    proposed = gpd.read_parquet(f"results/heterogeneous_graph_ensemble/{study}.parquet")
    columns = [
        "lsoa21cd",
        "prediction_heterogeneous_facility_gated_gravity_gnn_hybrid",
        "prediction_heterogeneous_graph_ensemble",
    ]
    frame = reference.merge(
        proposed[columns].drop(columns=[], errors="ignore"),
        on="lsoa21cd",
        validate="one_to_one",
    )
    observed = frame["observed"].to_numpy(dtype=float)
    frame["residual_context_hgb"] = frame["residual_context_only"]
    frame["residual_aggregated_gravity_hgb"] = frame["residual_gravity_scale_15"]
    frame["residual_gated_gravity_gnn"] = observed - frame[
        "prediction_heterogeneous_facility_gated_gravity_gnn_hybrid"
    ].to_numpy(dtype=float)
    frame["residual_proposed_ensemble"] = observed - frame[
        "prediction_heterogeneous_graph_ensemble"
    ].to_numpy(dtype=float)
    return frame


def plot_moran(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), layout="constrained")
    y = np.arange(len(RESIDUAL_MODELS))
    for axis, study in zip(axes, STUDIES, strict=True):
        city = frame[frame["study"] == study].set_index("model")
        values = np.asarray([city.loc[model, "I"] for model in RESIDUAL_MODELS])
        lower = min(-0.02, float(values.min()) - 0.04)
        upper = max(0.1, float(values.max()) + 0.08)
        axis.axvline(0, color="#999999", linewidth=0.8)
        for position, (model, value) in enumerate(zip(RESIDUAL_MODELS, values, strict=True)):
            axis.hlines(position, 0, value, color="#cfcfcf", linewidth=2)
            axis.scatter(value, position, color=MODEL_COLORS[model], s=62, zorder=3)
            axis.annotate(
                f"{value:.3f}",
                xy=(value, position),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
            )
        axis.set_yticks(y, [MODEL_LABELS[model] for model in RESIDUAL_MODELS])
        axis.invert_yaxis()
        axis.set_xlim(lower, upper)
        axis.set_xlabel("Global Moran's I of OOF residuals")
        axis.set_title(STUDY_LABELS[study])
        axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    fig.suptitle("Residual spatial autocorrelation under Queen contiguity", fontsize=14)
    return save_figure(fig, output_dir / "figure3_residual_global_moran")


def plot_lisa(frames: dict[str, gpd.GeoDataFrame], output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.8), layout="constrained")
    for row, study in enumerate(STUDIES):
        frame = frames[study]
        for column, model in enumerate(RESIDUAL_MODELS):
            axis = axes[row, column]
            labels = frame[f"{model}_lisa_cluster"]
            for label in ("NS", "HH", "LL", "HL", "LH"):
                selected = labels == label
                if selected.any():
                    frame.loc[selected].plot(
                        ax=axis,
                        color=LISA_COLORS[label],
                        linewidth=LSOA_BOUNDARY_WIDTH,
                        edgecolor=LSOA_BOUNDARY_COLOR,
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
                    fontsize=12,
                )
            axis.set_axis_off()
    handles = [Patch(facecolor=LISA_COLORS[label], label=label) for label in ("HH", "LL", "HL", "LH", "NS")]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Local Moran residual clusters (Queen weights; BH-FDR q≤0.05)",
        fontsize=14,
    )
    return save_figure(fig, output_dir / "figure4_residual_lisa")


def plot_semivariogram(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), layout="constrained")
    for axis, study in zip(axes, STUDIES, strict=True):
        city = frame[frame["study"] == study]
        for model in RESIDUAL_MODELS:
            selected = city[city["model"] == model]
            axis.plot(
                selected["distance_mean_m"] / 1000,
                selected["semivariance"],
                color=MODEL_COLORS[model],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=MODEL_LABELS[model],
            )
        axis.set_title(STUDY_LABELS[study])
        axis.set_xlabel("LSOA centroid separation (km)")
        axis.set_ylabel("Residual semivariance (AHAH²)")
        axis.grid(color="#dddddd", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.6,
    )
    fig.suptitle("Empirical semivariograms of out-of-fold residuals", fontsize=14)
    return save_figure(fig, output_dir / "figure5_residual_semivariogram")


def residual_outputs(args: argparse.Namespace) -> dict[str, object]:
    from run_spatial_residual_diagnostics import (
        global_moran,
        local_moran,
        semivariogram,
        weights_for,
    )

    frames: dict[str, gpd.GeoDataFrame] = {}
    moran_rows: list[dict[str, object]] = []
    lisa_counts: list[dict[str, object]] = []
    semivariogram_rows: list[dict[str, object]] = []
    for study_index, study in enumerate(STUDIES):
        frame = load_residual_frame(study)
        bundle = weights_for(frame, 5000.0)
        residual_map = {
            model: frame[f"residual_{model}"].to_numpy(dtype=float)
            for model in RESIDUAL_MODELS
        }
        for model_index, model in enumerate(RESIDUAL_MODELS):
            moran_rows.append(
                {
                    "study": study,
                    "weights": "queen",
                    "model": model,
                    **global_moran(
                        residual_map[model],
                        bundle["queen"],
                        args.permutations,
                        args.seed + study_index * 1000 + model_index,
                    ),
                }
            )
            lisa = local_moran(
                residual_map[model],
                bundle["queen"],
                args.permutations,
                args.seed + study_index * 1000 + 100 + model_index,
            )
            frame[f"{model}_lisa_cluster"] = [
                LISA_LABELS[int(value)] for value in lisa["cluster_code"]
            ]
            frame[f"{model}_lisa_I"] = lisa["local_I"]
            frame[f"{model}_lisa_p"] = lisa["p_permutation"]
            frame[f"{model}_lisa_q_bh"] = lisa["q_bh"]
            counts = frame[f"{model}_lisa_cluster"].value_counts()
            lisa_counts.append(
                {
                    "study": study,
                    "model": model,
                    **{label: int(counts.get(label, 0)) for label in ("HH", "LL", "HL", "LH", "NS")},
                }
            )
        frames[study] = frame
        frame.to_parquet(args.output_dir / f"residual_lisa_{study}.parquet", index=False)
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
    lisa_frame = pd.DataFrame(lisa_counts)
    semivariogram_frame = pd.DataFrame(semivariogram_rows)
    moran_path = args.output_dir / "residual_global_moran.csv"
    lisa_path = args.output_dir / "residual_lisa_counts.csv"
    semivariogram_path = args.output_dir / "residual_semivariogram.csv"
    moran_frame.to_csv(moran_path, index=False)
    lisa_frame.to_csv(lisa_path, index=False)
    semivariogram_frame.to_csv(semivariogram_path, index=False)
    return {
        "moran_data": str(moran_path),
        "moran_figure": plot_moran(moran_frame, args.output_dir),
        "lisa_counts": str(lisa_path),
        "lisa_figure": plot_lisa(frames, args.output_dir),
        "semivariogram_data": str(semivariogram_path),
        "semivariogram_figure": plot_semivariogram(semivariogram_frame, args.output_dir),
    }


def residual_outputs_from_existing(args: argparse.Namespace) -> dict[str, object]:
    moran_path = args.output_dir / "residual_global_moran.csv"
    lisa_path = args.output_dir / "residual_lisa_counts.csv"
    semivariogram_path = args.output_dir / "residual_semivariogram.csv"
    required = [moran_path, lisa_path, semivariogram_path]
    required.extend(
        args.output_dir / f"residual_lisa_{study}.parquet" for study in STUDIES
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot reuse residual data; missing: " + ", ".join(missing)
        )
    moran_frame = pd.read_csv(moran_path)
    semivariogram_frame = pd.read_csv(semivariogram_path)
    frames = {
        study: gpd.read_parquet(
            args.output_dir / f"residual_lisa_{study}.parquet"
        )
        for study in STUDIES
    }
    return {
        "moran_data": str(moran_path),
        "moran_figure": plot_moran(moran_frame, args.output_dir),
        "lisa_counts": str(lisa_path),
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
        "distance_performance": distance_performance_figure(args.output_dir),
        "distance_encoding_shapes": distance_shape_figure(args.output_dir),
        "residual_diagnostics": (
            residual_outputs_from_existing(args)
            if args.reuse_residual_data
            else residual_outputs(args)
        ),
        "settings": {
            "permutations": args.permutations,
            "semivariogram_pairs": args.semivariogram_pairs,
            "semivariogram_bins": args.semivariogram_bins,
            "residuals": "strict out-of-fold observed minus prediction",
            "primary_metric": "macro mean over five frozen contiguous spatial folds",
            "candidate_status": "exploratory development candidate",
        },
    }
    manifest = args.output_dir / "publication_outputs_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        path.name: sha256(path)
        for path in args.output_dir.iterdir()
        if path.is_file() and path != manifest
    }
    outputs["sha256"] = checksums
    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "manifest": str(manifest.resolve())}), flush=True)


if __name__ == "__main__":
    main()
