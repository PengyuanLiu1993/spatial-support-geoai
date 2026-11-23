from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "case_studies"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "publication_outputs_heterogeneous_gnn"

STUDIES = {
    "greater_london": {
        "title": "Greater London",
        "panel": "a",
        "scale_km": 10,
    },
    "greater_manchester": {
        "title": "Greater Manchester",
        "panel": "b",
        "scale_km": 10,
    },
}

COLORS = {
    "ink": "#1C2532",
    "muted": "#657180",
    "frame": "#9BA6B1",
    "outside": "#E9EDF0",
    "land": "#F4F3EF",
    "road": "#66727C",
    "boundary": "#314957",
    "white": "#FFFFFF",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_study(study: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, np.ndarray]]:
    root = DATA_ROOT / study
    lsoas = gpd.read_parquet(root / "lsoas.parquet")
    boundary = gpd.read_parquet(root / "boundary.parquet")
    if lsoas.crs is None or boundary.crs is None:
        raise ValueError(f"{study} has missing CRS metadata")
    if lsoas.crs.to_epsg() != 27700:
        lsoas = lsoas.to_crs(27700)
    if boundary.crs.to_epsg() != 27700:
        boundary = boundary.to_crs(27700)
    network_file = np.load(root / "walk_network.npz")
    network = {name: network_file[name] for name in network_file.files}
    return lsoas, boundary, network


def road_segments(network: dict[str, np.ndarray], *, minimum_length_m: float) -> np.ndarray:
    x = network["road_x"]
    y = network["road_y"]
    u = network["edge_u"]
    v = network["edge_v"]
    length = network["edge_length_m"]
    keep = (u < v) & (length >= minimum_length_m)
    u = u[keep]
    v = v[keep]
    start = np.column_stack((x[u], y[u]))
    end = np.column_stack((x[v], y[v]))
    return np.stack((start, end), axis=1)


def padded_extent(boundary: gpd.GeoDataFrame, padding: float = 0.035) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    width = maxx - minx
    height = maxy - miny
    return (
        minx - width * padding,
        maxx + width * padding,
        miny - height * padding,
        maxy + height * padding,
    )


def add_scale_bar(axis: plt.Axes, length_km: int) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    width = xmax - xmin
    height = ymax - ymin
    length = length_km * 1000
    x0 = xmin + width * 0.065
    y0 = ymin + height * 0.070
    tick = height * 0.010
    effects = [patheffects.Stroke(linewidth=3.2, foreground=COLORS["white"]), patheffects.Normal()]
    axis.plot([x0, x0 + length], [y0, y0], color=COLORS["ink"], linewidth=1.8, zorder=20, path_effects=effects)
    axis.plot([x0, x0], [y0 - tick, y0 + tick], color=COLORS["ink"], linewidth=1.3, zorder=20, path_effects=effects)
    axis.plot([x0 + length, x0 + length], [y0 - tick, y0 + tick], color=COLORS["ink"], linewidth=1.3, zorder=20, path_effects=effects)
    label = axis.text(
        x0 + length / 2,
        y0 + height * 0.023,
        f"{length_km} km",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=21,
    )
    label.set_path_effects(effects)


def add_north_arrow(axis: plt.Axes) -> None:
    axis.text(
        0.925,
        0.890,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=30,
    )
    arrow = FancyArrowPatch(
        (0.925, 0.805),
        (0.925, 0.885),
        transform=axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        zorder=30,
    )
    arrow.set_path_effects([patheffects.Stroke(linewidth=3.2, foreground=COLORS["white"]), patheffects.Normal()])
    axis.add_patch(arrow)


def plot_study(
    axis: plt.Axes,
    lsoas: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    network: dict[str, np.ndarray],
    *,
    title: str,
    panel: str,
    scale_km: int,
    norm: mcolors.Normalize,
    cmap: mcolors.Colormap,
) -> None:
    xmin, xmax, ymin, ymax = padded_extent(boundary)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_facecolor(COLORS["outside"])

    # Reproducible local basemap from the same frozen OSM walking network used
    # in the analysis. Only longer links are retained to keep the figure quiet.
    boundary.plot(ax=axis, color=COLORS["land"], edgecolor="none", zorder=1)
    segments = road_segments(network, minimum_length_m=40.0)
    axis.add_collection(
        LineCollection(
            segments,
            colors=COLORS["road"],
            linewidths=0.16,
            alpha=0.20,
            zorder=2,
            rasterized=True,
        )
    )

    lsoas.plot(
        ax=axis,
        column="ahah",
        cmap=cmap,
        norm=norm,
        edgecolor=mcolors.to_rgba(COLORS["white"], 0.58),
        linewidth=0.075,
        alpha=0.89,
        zorder=4,
        rasterized=True,
    )

    # A sparse darker overlay keeps the main road structure legible without
    # resembling a river or competing with the AHAH surface.
    prominent = road_segments(network, minimum_length_m=80.0)
    axis.add_collection(
        LineCollection(
            prominent,
            colors=COLORS["road"],
            linewidths=0.14,
            alpha=0.16,
            zorder=5,
            rasterized=True,
        )
    )
    boundary.boundary.plot(ax=axis, color=COLORS["boundary"], linewidth=0.72, zorder=8)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["frame"])
        spine.set_linewidth(0.75)
    axis.set_xticks([])
    axis.set_yticks([])

    mean = float(lsoas["ahah"].mean())
    std = float(lsoas["ahah"].std())
    axis.set_title(
        f"({panel})  {title}",
        loc="left",
        fontsize=12.2,
        fontweight="bold",
        color=COLORS["ink"],
        pad=9,
    )
    axis.text(
        0.995,
        1.017,
        f"n = {len(lsoas):,} LSOAs  •  mean ± SD = {mean:.1f} ± {std:.1f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=COLORS["muted"],
    )
    add_scale_bar(axis, scale_km)
    add_north_arrow(axis)


def make_figure() -> tuple[plt.Figure, dict[str, dict[str, float]]]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": COLORS["white"],
            "savefig.edgecolor": COLORS["white"],
        }
    )
    loaded = {study: load_study(study) for study in STUDIES}
    all_values = np.concatenate([bundle[0]["ahah"].to_numpy(dtype=float) for bundle in loaded.values()])
    lower = float(np.floor(np.nanmin(all_values) / 5.0) * 5.0)
    upper = float(np.ceil(np.nanmax(all_values) / 5.0) * 5.0)
    norm = mcolors.Normalize(vmin=lower, vmax=upper)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ahah_environment",
        ["#EFF6F3", "#BFDCCF", "#78B4A1", "#E2C06C", "#C97552", "#873B45"],
        N=256,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14.8, 7.4), constrained_layout=False)
    figure.subplots_adjust(left=0.035, right=0.985, top=0.865, bottom=0.165, wspace=0.075)
    figure.suptitle(
        "Observed Access to Healthy Assets and Hazards (AHAH v5.1)",
        x=0.50,
        y=0.955,
        fontsize=16.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    figure.text(
        0.50,
        0.912,
        "LSOA-level environmental accessibility index used as the prediction target",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["muted"],
    )

    summary: dict[str, dict[str, float]] = {}
    for axis, (study, metadata) in zip(axes, STUDIES.items(), strict=True):
        lsoas, boundary, network = loaded[study]
        plot_study(
            axis,
            lsoas,
            boundary,
            network,
            title=metadata["title"],
            panel=metadata["panel"],
            scale_km=metadata["scale_km"],
            norm=norm,
            cmap=cmap,
        )
        summary[study] = {
            "n_lsoas": int(len(lsoas)),
            "mean": float(lsoas["ahah"].mean()),
            "sd": float(lsoas["ahah"].std()),
            "min": float(lsoas["ahah"].min()),
            "max": float(lsoas["ahah"].max()),
        }

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = figure.add_axes([0.330, 0.090, 0.340, 0.022])
    colorbar = figure.colorbar(scalar, cax=color_axis, orientation="horizontal")
    ticks = np.arange(np.ceil(lower / 10) * 10, upper + 0.1, 10)
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=7.7, colors=COLORS["ink"], length=2.5, width=0.6)
    colorbar.outline.set_edgecolor(COLORS["frame"])
    colorbar.outline.set_linewidth(0.6)
    colorbar.set_label("AHAH score", fontsize=8.6, color=COLORS["ink"], labelpad=4)
    figure.text(0.323, 0.101, "Lower / healthier", ha="right", va="center", fontsize=7.7, color=COLORS["muted"])
    figure.text(0.677, 0.101, "Higher / less healthy", ha="left", va="center", fontsize=7.7, color=COLORS["muted"])

    road_key = Line2D([0], [0], color=COLORS["road"], linewidth=1.0, alpha=0.55)
    figure.legend(
        [road_key],
        ["Frozen OpenStreetMap walking-road basemap"],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.071),
        frameon=False,
        fontsize=7.6,
        labelcolor=COLORS["muted"],
        handlelength=2.8,
    )
    figure.text(
        0.035,
        0.035,
        "Data: Geographic Data Service AHAH v5.1 (2026), 2021 LSOAs. Basemap: frozen OpenStreetMap extracts used by the study. CRS: British National Grid (EPSG:27700).",
        ha="left",
        va="center",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    return figure, summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure, summary = make_figure()
    stem = OUTPUT_ROOT / "figure0_ahah_observed_data_map"
    outputs: dict[str, dict[str, str | int]] = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.10)
        outputs[suffix] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    preview_path = stem.with_name(f"{stem.name}_preview.png")
    figure.savefig(preview_path, dpi=150, bbox_inches="tight", pad_inches=0.10)
    outputs["preview"] = {
        "path": str(preview_path.resolve()),
        "bytes": preview_path.stat().st_size,
        "sha256": sha256(preview_path),
    }
    plt.close(figure)
    manifest = {
        "title": "Observed AHAH v5.1 across the two case studies",
        "interpretation": "Higher AHAH scores indicate less healthy neighbourhood environments.",
        "summary": summary,
        "outputs": outputs,
    }
    manifest_path = stem.with_name(f"{stem.name}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
