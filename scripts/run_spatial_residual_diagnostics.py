from __future__ import annotations

import argparse
import json
from pathlib import Path

import esda
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from libpysal.weights import DistanceBand, KNN, Queen

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MODEL_COLUMNS = {
    "mean": "residual_mean",
    "ridge": "residual_ridge",
    "random_forest": "residual_random_forest",
    "local_hgb": "residual_local_hgb",
    "pooled_hgb": "residual_proposed_pooled_hgb",
    "gcn": "residual_gcn",
    "gat": "residual_gat",
    "fixed_gravity": "residual_fixed_gravity",
    "gel_gnn": "residual_gel_gnn",
    "facility_gravity_hybrid": "residual_facility_gravity_hybrid",
    "facility_gravity_scale15": "residual_facility_gravity_scale15",
    "facility_gravity_gnn": "residual_facility_gravity_gnn",
}
LISA_MODELS = (
    "pooled_hgb",
    "facility_gravity_hybrid",
    "facility_gravity_scale15",
)
LISA_LABELS = {0: "NS", 1: "HH", 2: "LH", 3: "LL", 4: "HL"}
LISA_COLORS = {
    "NS": "#d9d9d9",
    "HH": "#b2182b",
    "LL": "#2166ac",
    "HL": "#ef8a62",
    "LH": "#67a9cf",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OOF residual Moran, FDR-LISA, and semivariogram diagnostics"
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--distance-band-metres", type=float, default=5000.0)
    parser.add_argument("--semivariogram-pairs", type=int, default=500_000)
    parser.add_argument("--semivariogram-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/spatial_diagnostics")
    )
    return parser


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def weights_for(frame: gpd.GeoDataFrame, distance_band: float) -> dict:
    projected = frame.to_crs(27700)
    centroids = projected.geometry.centroid
    coordinates = np.column_stack((centroids.x.to_numpy(), centroids.y.to_numpy()))
    ids = list(range(len(frame)))
    queen = Queen.from_dataframe(projected, ids=ids, silence_warnings=True)
    knn = KNN.from_array(coordinates, k=8, ids=ids)
    band = DistanceBand.from_array(
        coordinates,
        threshold=distance_band,
        binary=True,
        ids=ids,
        silence_warnings=True,
    )
    for weights in (queen, knn, band):
        weights.transform = "R"
    return {
        "projected": projected,
        "coordinates": coordinates,
        "queen": queen,
        "knn8": knn,
        "distance_band_5km": band,
    }


def weight_audit(weights: object) -> dict:
    cardinalities = np.asarray(list(weights.cardinalities.values()), dtype=float)
    return {
        "n": int(weights.n),
        "islands": [int(value) for value in weights.islands],
        "components": int(weights.n_components),
        "neighbours_mean": float(cardinalities.mean()),
        "neighbours_min": int(cardinalities.min()),
        "neighbours_max": int(cardinalities.max()),
    }


def global_moran(
    residual: np.ndarray, weights: object, permutations: int, seed: int
) -> dict:
    np.random.seed(seed)
    fitted = esda.Moran(
        residual,
        weights,
        transformation="R",
        permutations=permutations,
        two_tailed=True,
    )
    return {
        "I": float(fitted.I),
        "expected_I": float(fitted.EI),
        "p_permutation": float(fitted.p_sim),
        "z_permutation": float(fitted.z_sim),
        "p_normal": float(fitted.p_norm),
    }


def local_moran(
    residual: np.ndarray, weights: object, permutations: int, seed: int
) -> dict[str, np.ndarray]:
    fitted = esda.Moran_Local(
        residual,
        weights,
        transformation="R",
        permutations=permutations,
        seed=seed,
        n_jobs=1,
        keep_simulations=False,
    )
    p_value = np.asarray(fitted.p_sim, dtype=float)
    q_value = bh_fdr(p_value)
    significant = q_value <= 0.05
    cluster_code = np.where(significant, np.asarray(fitted.q, dtype=int), 0)
    return {
        "local_I": np.asarray(fitted.Is, dtype=float),
        "quadrant": np.asarray(fitted.q, dtype=int),
        "p_permutation": p_value,
        "q_bh": q_value,
        "cluster_code": cluster_code,
    }


def semivariogram(
    coordinates: np.ndarray,
    residuals: dict[str, np.ndarray],
    pairs: int,
    bins: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(coordinates), size=pairs, dtype=np.int64)
    second = rng.integers(0, len(coordinates), size=pairs, dtype=np.int64)
    same = first == second
    while same.any():
        second[same] = rng.integers(0, len(coordinates), size=int(same.sum()))
        same = first == second
    distance = np.sqrt(np.sum((coordinates[first] - coordinates[second]) ** 2, axis=1))
    maximum = float(np.quantile(distance, 0.9))
    edges = np.linspace(0.0, maximum, bins + 1)
    bin_index = np.digitize(distance, edges, right=False) - 1
    output: list[dict] = []
    for model, residual in residuals.items():
        semivariance = 0.5 * (residual[first] - residual[second]) ** 2
        for index in range(bins):
            selected = bin_index == index
            if not selected.any():
                continue
            output.append(
                {
                    "model": model,
                    "bin": index,
                    "distance_lower_m": float(edges[index]),
                    "distance_upper_m": float(edges[index + 1]),
                    "distance_mean_m": float(distance[selected].mean()),
                    "semivariance": float(semivariance[selected].mean()),
                    "pairs": int(selected.sum()),
                }
            )
    return output


def plot_lisa(frame: gpd.GeoDataFrame, study: str, output: Path) -> None:
    fig, axes = plt.subplots(1, len(LISA_MODELS), figsize=(18, 7), constrained_layout=True)
    for axis, model in zip(np.atleast_1d(axes), LISA_MODELS, strict=True):
        labels = frame[f"{model}_lisa_cluster"]
        for label in LISA_LABELS.values():
            selected = labels == label
            if selected.any():
                frame.loc[selected].plot(
                    ax=axis,
                    color=LISA_COLORS[label],
                    linewidth=0.05,
                    edgecolor="#777777",
                )
        axis.set_title(model.replace("_", " ").title())
        axis.set_axis_off()
    handles = [
        Patch(facecolor=LISA_COLORS[label], label=label)
        for label in ("HH", "LL", "HL", "LH", "NS")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"{study.replace('_', ' ').title()} OOF residual LISA (BH-FDR q≤0.05)")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_semivariograms(all_rows: pd.DataFrame, output: Path) -> None:
    studies = all_rows["study"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(studies), figsize=(12, 5), constrained_layout=True)
    for axis, study in zip(np.atleast_1d(axes), studies, strict=True):
        subset = all_rows[all_rows["study"] == study]
        for model, color in (
            ("pooled_hgb", "#7570b3"),
            ("facility_gravity_hybrid", "#1b9e77"),
            ("facility_gravity_scale15", "#d95f02"),
        ):
            rows = subset[subset["model"] == model]
            axis.plot(
                rows["distance_mean_m"] / 1000.0,
                rows["semivariance"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=model.replace("_", " "),
            )
        axis.set_title(study.replace("_", " ").title())
        axis.set_xlabel("LSOA centroid separation (km)")
        axis.set_ylabel("Residual semivariance (AHAH²)")
        axis.grid(alpha=0.2)
    axes = np.atleast_1d(axes)
    axes[0].legend(frameon=False)
    fig.suptitle("OOF residual empirical semivariograms")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "settings": {
            "studies": args.study,
            "models": MODEL_COLUMNS,
            "lisa_models": LISA_MODELS,
            "permutations": args.permutations,
            "lisa_multiple_testing": "Benjamini-Hochberg FDR q <= 0.05",
            "distance_band_metres": args.distance_band_metres,
            "semivariogram_pairs": args.semivariogram_pairs,
            "semivariogram_bins": args.semivariogram_bins,
            "seed": args.seed,
            "residual_definition": "strict OOF observed minus prediction",
        },
        "studies": {},
    }
    global_rows: list[dict] = []
    semivariogram_rows: list[dict] = []
    for study_index, study in enumerate(args.study):
        frame = gpd.read_parquet(args.oof_dir / f"{study}.parquet")
        missing = [column for column in MODEL_COLUMNS.values() if column not in frame]
        if missing:
            raise ValueError(f"missing residual columns for {study}: {missing}")
        bundle = weights_for(frame, args.distance_band_metres)
        weight_names = ("queen", "knn8", "distance_band_5km")
        study_summary = {
            "rows": len(frame),
            "crs": str(frame.crs),
            "weights": {
                name: weight_audit(bundle[name]) for name in weight_names
            },
            "global_moran": [],
            "lisa": {},
        }
        for weight_index, weight_name in enumerate(weight_names):
            weights = bundle[weight_name]
            for model_index, (model, column) in enumerate(MODEL_COLUMNS.items()):
                row = {
                    "study": study,
                    "weights": weight_name,
                    "model": model,
                    **global_moran(
                        frame[column].to_numpy(dtype=float),
                        weights,
                        args.permutations,
                        args.seed
                        + study_index * 1000
                        + weight_index * 100
                        + model_index,
                    ),
                }
                study_summary["global_moran"].append(row)
                global_rows.append(row)
        for model_index, model in enumerate(LISA_MODELS):
            values = local_moran(
                frame[MODEL_COLUMNS[model]].to_numpy(dtype=float),
                bundle["queen"],
                args.permutations,
                args.seed + study_index * 1000 + 500 + model_index,
            )
            prefix = f"{model}_lisa"
            for name, values_array in values.items():
                frame[f"{prefix}_{name}"] = values_array
            frame[f"{prefix}_cluster"] = [
                LISA_LABELS[int(value)] for value in values["cluster_code"]
            ]
            counts = frame[f"{prefix}_cluster"].value_counts().to_dict()
            study_summary["lisa"][model] = {
                "cluster_counts": {name: int(counts.get(name, 0)) for name in LISA_LABELS.values()},
                "significant_fdr": int((values["q_bh"] <= 0.05).sum()),
            }
        lisa_path = args.output_dir / f"{study}_lisa.parquet"
        frame.to_parquet(lisa_path, index=False)
        lisa_figure = args.output_dir / f"{study}_lisa.png"
        plot_lisa(frame, study, lisa_figure)
        semivariogram_rows.extend(
            {
                "study": study,
                **row,
            }
            for row in semivariogram(
                bundle["coordinates"],
                {
                    model: frame[MODEL_COLUMNS[model]].to_numpy(dtype=float)
                    for model in LISA_MODELS
                },
                args.semivariogram_pairs,
                args.semivariogram_bins,
                args.seed + study_index,
            )
        )
        study_summary["outputs"] = {
            "lisa_parquet": str(lisa_path.resolve()),
            "lisa_figure": str(lisa_figure.resolve()),
        }
        summary["studies"][study] = study_summary
        print(json.dumps({"event": "study_complete", "study": study}), flush=True)

    global_frame = pd.DataFrame(global_rows)
    global_frame["q_bh_within_study"] = np.nan
    for study, index in global_frame.groupby("study").groups.items():
        global_frame.loc[index, "q_bh_within_study"] = bh_fdr(
            global_frame.loc[index, "p_permutation"].to_numpy()
        )
    global_path = args.output_dir / "global_moran.csv"
    global_frame.to_csv(global_path, index=False)
    semivariogram_frame = pd.DataFrame(semivariogram_rows)
    semivariogram_path = args.output_dir / "semivariogram.csv"
    semivariogram_frame.to_csv(semivariogram_path, index=False)
    semivariogram_figure = args.output_dir / "semivariogram.png"
    plot_semivariograms(semivariogram_frame, semivariogram_figure)
    summary["outputs"] = {
        "global_moran": str(global_path.resolve()),
        "semivariogram": str(semivariogram_path.resolve()),
        "semivariogram_figure": str(semivariogram_figure.resolve()),
    }
    summary_path = args.output_dir / "spatial_residual_diagnostics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
