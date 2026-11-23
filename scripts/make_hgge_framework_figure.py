from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import (
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
    RegularPolygon,
    Wedge,
)


matplotlib.use("Agg")


C = {
    "ink": "#1C2532",
    "muted": "#657180",
    "border": "#AEB7C2",
    "light": "#E9EDF1",
    "area": "#E8EEF3",
    "area_edge": "#62788C",
    "target": "#285F73",
    "gp": "#377E9F",
    "hospital": "#C45A50",
    "green": "#3E8965",
    "food": "#D88A3D",
    "model": "#27755F",
    "model_light": "#E3F0EA",
    "tabular": "#6E63A8",
    "tabular_light": "#ECEAF6",
    "white": "#FFFFFF",
}

RELATION_STYLE = {
    "GP": C["gp"],
    "Hospital": C["hospital"],
    "Green": C["green"],
    "Food": C["food"],
}


def axis_y_compensation(axis: plt.Axes) -> float:
    """Return the y-axis data scaling needed for equal on-screen distances."""
    position = axis.get_position()
    figure_width, figure_height = axis.figure.get_size_inches()
    x_span = abs(axis.get_xlim()[1] - axis.get_xlim()[0]) or 1.0
    y_span = abs(axis.get_ylim()[1] - axis.get_ylim()[0]) or 1.0
    x_pixels_per_unit = position.width * figure_width / x_span
    y_pixels_per_unit = position.height * figure_height / y_span
    return x_pixels_per_unit / y_pixels_per_unit


def display_circle(
    axis: plt.Axes,
    centre: tuple[float, float],
    radius: float,
    **kwargs,
) -> Ellipse:
    """Return a visually circular patch inside a non-square panel axis.

    The framework uses normalised panel coordinates. A standard matplotlib
    Circle is therefore stretched whenever a panel is tall or narrow. This
    helper compensates for the physical panel aspect ratio so every node is a
    true circle in the exported PNG, PDF and SVG.
    """
    y_compensation = axis_y_compensation(axis)
    return Ellipse(
        centre,
        width=2 * radius,
        height=2 * radius * y_compensation,
        **kwargs,
    )


def display_wedge(
    axis: plt.Axes,
    centre: tuple[float, float],
    radius: float,
    theta1: float,
    theta2: float,
    **kwargs,
) -> Wedge:
    """Return a wedge whose enclosing outline remains circular on screen."""
    y_compensation = axis_y_compensation(axis)
    patch = Wedge(centre, radius, theta1, theta2, **kwargs)
    cx, cy = centre
    patch.set_transform(
        transforms.Affine2D()
        .translate(-cx, -cy)
        .scale(1.0, y_compensation)
        .translate(cx, cy)
        + axis.transData
    )
    return patch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the HFGE academic framework figure")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication_outputs_heterogeneous_gnn"),
    )
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(study: str) -> dict[str, gpd.GeoDataFrame]:
    root = Path("data/processed/case_studies") / study
    return {
        "boundary": gpd.read_parquet(root / "boundary.parquet"),
        "lsoas": gpd.read_parquet(root / "lsoas.parquet"),
        "facilities": gpd.read_parquet(root / "osm_facilities.parquet"),
    }


def panel(
    figure: plt.Figure,
    bounds: tuple[float, float, float, float],
    letter: str,
    title: str,
) -> plt.Axes:
    x, y, width, height = bounds
    figure.patches.append(
        Rectangle(
            (x, y),
            width,
            height,
            transform=figure.transFigure,
            facecolor=C["white"],
            edgecolor=C["border"],
            linewidth=0.9,
            zorder=-3,
        )
    )
    axis = figure.add_axes(bounds)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.text(0.025, 0.965, f"({letter})", ha="left", va="top", fontsize=12.0, fontweight="bold", color=C["ink"])
    axis.text(0.13, 0.965, title, ha="left", va="top", fontsize=12.0, fontweight="bold", color=C["ink"])
    axis.plot([0.025, 0.975], [0.885, 0.885], color=C["border"], linewidth=0.8)
    return axis


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = C["ink"],
    width: float = 1.3,
    alpha: float = 1.0,
    rad: float = 0.0,
    scale: float = 11,
    zorder: int = 7,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def figure_arrow(
    figure: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
) -> None:
    figure.patches.append(
        FancyArrowPatch(
            start,
            end,
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.0,
            color=C["ink"],
            connectionstyle=f"arc3,rad={rad}",
            zorder=30,
        )
    )


def module_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str = C["model"],
    fill: str = C["white"],
    fontsize: float = 7.8,
    zorder: int = 6,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.0,
            zorder=zorder,
        )
    )
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=C["ink"], linespacing=1.1, zorder=zorder + 1)


def facility_node(
    axis: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    radius: float = 0.035,
    show_label: bool = False,
    zorder: int = 10,
) -> None:
    color = RELATION_STYLE[label]
    axis.add_patch(display_circle(axis, (x, y), radius, facecolor=color, edgecolor=C["white"], linewidth=1.0, zorder=zorder))
    short = {"GP": "GP", "Hospital": "H", "Green": "G", "Food": "F"}[label]
    axis.text(x, y, short, ha="center", va="center", fontsize=7.4 if short == "GP" else 8.0, fontweight="bold", color=C["white"], zorder=zorder + 1)
    if show_label:
        axis.text(x, y - radius - 0.022, label, ha="center", va="top", fontsize=7.0, color=C["muted"], zorder=zorder + 2)


def area_node(
    axis: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    radius: float = 0.045,
    target: bool = False,
    zorder: int = 9,
) -> None:
    axis.add_patch(
        RegularPolygon(
            (x, y),
            numVertices=6,
            radius=radius,
            orientation=np.pi / 6,
            facecolor=C["target"] if target else C["area"],
            edgecolor=C["target"] if target else C["area_edge"],
            linewidth=1.1,
            zorder=zorder,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=7.8, fontweight="bold", color=C["white"] if target else C["ink"], zorder=zorder + 1)


def plot_outline(axis: plt.Axes, boundary: gpd.GeoDataFrame, title: str) -> None:
    data = boundary.copy()
    data["geometry"] = data.geometry.simplify(130)
    data.plot(ax=axis, color=C["area"], edgecolor=C["area_edge"], linewidth=0.65)
    axis.set_axis_off()
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=7.8, fontweight="bold", color=C["ink"], pad=1)


def feature_matrix(axis: plt.Axes, x: float, y: float, rows: int = 5, columns: int = 7) -> None:
    rng = np.random.default_rng(15)
    values = rng.uniform(0.12, 0.95, size=(rows, columns))
    cmap = mcolors.LinearSegmentedColormap.from_list("feature_matrix", ["#F1F4F6", "#6E8EA1"])
    cell_w, cell_h = 0.026, 0.028
    for row in range(rows):
        for column in range(columns):
            axis.add_patch(Rectangle((x + column * cell_w, y + (rows - 1 - row) * cell_h), cell_w - 0.002, cell_h - 0.002, facecolor=cmap(values[row, column]), edgecolor="none", zorder=4))


def draw_inputs(axis: plt.Axes, london: dict, manchester: dict) -> None:
    london_axis = axis.inset_axes([0.05, 0.53, 0.42, 0.27])
    manchester_axis = axis.inset_axes([0.53, 0.53, 0.42, 0.27])
    plot_outline(london_axis, london["boundary"], "London")
    plot_outline(manchester_axis, manchester["boundary"], "Manchester")
    axis.text(0.08, 0.43, "LSOA covariates", ha="left", va="center", fontsize=7.8, fontweight="bold", color=C["ink"])
    feature_matrix(axis, 0.08, 0.22)
    axis.text(0.17, 0.16, r"$X_i$  (103 variables)", ha="center", va="center", fontsize=7.5, color=C["muted"])
    axis.text(0.55, 0.43, "Facility nodes", ha="left", va="center", fontsize=7.8, fontweight="bold", color=C["ink"])
    for x, label in zip([0.58, 0.70, 0.82, 0.92], RELATION_STYLE):
        facility_node(axis, x, 0.33, label, radius=0.033, show_label=False)
    axis.text(0.75, 0.22, r"type $r$, mass/quality $m_j$", ha="center", va="center", fontsize=7.4, color=C["muted"])
    axis.text(0.75, 0.14, "routed walking time to LSOA centroids", ha="center", va="center", fontsize=7.4, color=C["muted"])


def draw_unified_graph(axis: plt.Axes) -> None:
    lsoa = {
        "u1": (0.22, 0.27),
        "u2": (0.42, 0.40),
        "ui": (0.58, 0.31),
        "u4": (0.72, 0.49),
        "u5": (0.48, 0.65),
        "u6": (0.80, 0.25),
    }
    queen_edges = [("u1", "u2"), ("u1", "ui"), ("u2", "ui"), ("u2", "u5"), ("ui", "u4"), ("ui", "u6"), ("u4", "u5"), ("u4", "u6"), ("u5", "ui")]
    for first, second in queen_edges:
        axis.plot([lsoa[first][0], lsoa[second][0]], [lsoa[first][1], lsoa[second][1]], color=C["area_edge"], linewidth=1.0, zorder=2)
    for name, (x, y) in lsoa.items():
        area_node(axis, x, y, r"$u_i$" if name == "ui" else "", radius=0.041, target=name == "ui")

    facilities = [
        (0.11, 0.75, "GP", "u2"),
        (0.28, 0.79, "Hospital", "u1"),
        (0.46, 0.81, "Green", "u5"),
        (0.64, 0.79, "Food", "u4"),
        (0.83, 0.74, "GP", "u4"),
        (0.92, 0.58, "Hospital", "u6"),
    ]
    for index, (x, y, relation, destination) in enumerate(facilities):
        facility_node(axis, x, y, relation, radius=0.028, show_label=False)
        target = lsoa[destination]
        arrow(axis, (x, y - 0.03), target, color=RELATION_STYLE[relation], width=1.15, alpha=0.84, rad=0.06 if index % 2 == 0 else -0.05, scale=8, zorder=4)
    arrow(axis, (0.11, 0.72), lsoa["ui"], color=C["gp"], width=0.85, alpha=0.55, rad=-0.12, scale=7, zorder=3)
    axis.text(0.33, 0.67, r"$e_{ji}^{(r)}:\; (t_{ji},m_j,r)$", ha="center", va="center", fontsize=7.8, color=C["ink"], bbox={"boxstyle": "square,pad=0.12", "facecolor": C["white"], "edgecolor": "none", "alpha": 0.92}, zorder=12)

    axis.text(0.06, 0.11, "Node types", ha="left", va="center", fontsize=7.3, fontweight="bold", color=C["ink"])
    facility_node(axis, 0.25, 0.11, "GP", radius=0.021)
    axis.text(0.29, 0.11, r"facility $v_j$", ha="left", va="center", fontsize=7.0, color=C["muted"])
    area_node(axis, 0.50, 0.11, "", radius=0.026)
    axis.text(0.55, 0.11, r"LSOA $u_i$", ha="left", va="center", fontsize=7.0, color=C["muted"])
    axis.plot([0.72, 0.78], [0.11, 0.11], color=C["area_edge"], linewidth=1.2)
    axis.text(0.80, 0.11, "Queen edge", ha="left", va="center", fontsize=7.0, color=C["muted"])
    axis.text(0.50, 0.04, "One heterogeneous graph: facility relations + the final LSOA Queen relation", ha="center", va="center", fontsize=7.5, color=C["muted"])


def relation_row(axis: plt.Axes, y: float, label: str) -> None:
    color = RELATION_STYLE[label]
    facility_node(axis, 0.055, y, label, radius=0.025)
    module_box(axis, 0.11, y - 0.035, 0.21, 0.070, r"edge basis  $\phi_r$", edge=color, fill=mcolors.to_rgba(color, 0.07), fontsize=7.2)
    axis.add_patch(display_circle(axis, (0.365, y), 0.021, facecolor=C["white"], edgecolor=C["ink"], linewidth=0.9, zorder=7))
    axis.text(0.365, y, "×", ha="center", va="center", fontsize=10, color=C["ink"], zorder=8)
    module_box(axis, 0.41, y - 0.035, 0.18, 0.070, r"Optional monotone  $\kappa_r$", edge=color, fill=mcolors.to_rgba(color, 0.07), fontsize=7.2)
    module_box(axis, 0.64, y - 0.035, 0.16, 0.070, r"AGG$_r$", edge=color, fill=C["white"], fontsize=7.2)
    axis.add_patch(Rectangle((0.845, y - 0.025), 0.105, 0.050, facecolor=mcolors.to_rgba(color, 0.55), edgecolor=color, linewidth=0.8, zorder=7))
    for start, end in [((0.08, y), (0.11, y)), ((0.32, y), (0.344, y)), ((0.386, y), (0.41, y)), ((0.59, y), (0.64, y)), ((0.80, y), (0.845, y))]:
        arrow(axis, start, end, color=color, width=1.0, scale=7)


def draw_relation_encoder(axis: plt.Axes) -> None:
    axis.text(0.215, 0.82, "relation-specific edge state", ha="center", va="center", fontsize=7.3, color=C["muted"])
    axis.text(0.50, 0.82, "soft monotone prior", ha="center", va="center", fontsize=7.3, color=C["muted"])
    axis.text(0.72, 0.82, "destination aggregation", ha="center", va="center", fontsize=7.3, color=C["muted"])
    for y, label in zip([0.70, 0.56, 0.42, 0.28], RELATION_STYLE):
        relation_row(axis, y, label)
    axis.text(0.215, 0.14, r"$[1,\log m_j,t,t^2,e^{-2t/40},\log m_j\!\times\!t]$", ha="center", va="center", fontsize=7.0, color=C["muted"])
    axis.text(0.72, 0.14, "weighted mean + log strength + log count", ha="center", va="center", fontsize=7.0, color=C["muted"])
    axis.text(0.90, 0.08, r"$Z_i\in\mathbb{R}^{R\times H}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color=C["ink"])


def context_vector(axis: plt.Axes, x: float, y: float, width: float = 0.11) -> None:
    heights = [0.045, 0.075, 0.060, 0.088, 0.052, 0.080, 0.065]
    cell = width / len(heights)
    for index, height in enumerate(heights):
        axis.add_patch(Rectangle((x + index * cell, y), cell * 0.70, height, facecolor=C["area"], edgecolor=C["area_edge"], linewidth=0.5, zorder=6))


def relation_tensor(axis: plt.Axes, x: float, y: float, width: float = 0.11) -> None:
    colors = list(RELATION_STYLE.values())
    for row, color in enumerate(colors):
        for column in range(5):
            axis.add_patch(Rectangle((x + column * width / 5, y + (3 - row) * 0.028), width / 5 - 0.003, 0.024, facecolor=mcolors.to_rgba(color, 0.30 + column * 0.11), edgecolor="none", zorder=6))


def small_graph(axis: plt.Axes, centre: tuple[float, float], *, updated: bool = False) -> tuple[float, float]:
    cx, cy = centre
    offsets = np.array([[0, 0], [-0.055, 0.055], [0.055, 0.060], [-0.065, -0.040], [0.060, -0.045], [0, -0.082]])
    nodes = offsets + np.array([cx, cy])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (3, 5), (4, 5)]
    for i, j in edges:
        axis.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color=C["model"] if updated else C["area_edge"], linewidth=0.85, zorder=3)
    for index, (x, y) in enumerate(nodes):
        axis.add_patch(RegularPolygon((x, y), numVertices=6, radius=0.020, orientation=np.pi / 6, facecolor=C["model"] if index == 0 else (C["model_light"] if updated else C["area"]), edgecolor=C["model"] if updated or index == 0 else C["area_edge"], linewidth=0.8, zorder=6))
    return tuple(nodes[0])


def neural_block(axis: plt.Axes, x: float, y: float, width: float, height: float, label: str) -> None:
    module_box(axis, x, y, width, height, "", edge=C["model"], fill=C["white"])
    left = [(x + width * 0.25, y + height * value) for value in (0.28, 0.50, 0.72)]
    right = [(x + width * 0.72, y + height * value) for value in (0.36, 0.64)]
    for start in left:
        for end in right:
            axis.plot([start[0], end[0]], [start[1], end[1]], color=C["light"], linewidth=0.6, zorder=4)
    for point in left + right:
        axis.add_patch(display_circle(axis, point, 0.008, facecolor=C["model"], edgecolor=C["white"], linewidth=0.4, zorder=7))
    axis.text(x + width / 2, y - 0.035, label, ha="center", va="center", fontsize=7.1, color=C["muted"])


def draw_area_gnn(axis: plt.Axes) -> None:
    axis.text(0.08, 0.80, r"Area covariates  $X_i$", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C["ink"])
    context_vector(axis, 0.025, 0.61, width=0.11)
    module_box(axis, 0.17, 0.62, 0.14, 0.10, "Area encoder", edge=C["area_edge"], fill=C["white"], fontsize=7.2)
    arrow(axis, (0.14, 0.655), (0.17, 0.67), color=C["area_edge"], width=1.0, scale=7)

    axis.text(0.08, 0.42, r"Relation tensor  $Z_i$", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C["ink"])
    relation_tensor(axis, 0.025, 0.22, width=0.11)
    module_box(axis, 0.17, 0.25, 0.14, 0.10, "Facility fusion", edge=C["model"], fill=C["white"], fontsize=7.2)
    arrow(axis, (0.14, 0.28), (0.17, 0.30), color=C["model"], width=1.0, scale=7)

    axis.add_patch(display_circle(axis, (0.36, 0.49), 0.030, facecolor=C["white"], edgecolor=C["model"], linewidth=1.0, zorder=8))
    axis.text(0.36, 0.49, "⊕", ha="center", va="center", fontsize=12, fontweight="bold", color=C["model"], zorder=9)
    arrow(axis, (0.31, 0.67), (0.34, 0.52), color=C["area_edge"], width=1.0, rad=0.08, scale=7)
    arrow(axis, (0.31, 0.30), (0.34, 0.46), color=C["model"], width=1.0, rad=-0.08, scale=7)
    module_box(axis, 0.42, 0.43, 0.13, 0.12, "Joint fusion\n" + r"$h_i^{(0)}$", edge=C["model"], fill=C["model_light"], fontsize=7.4)
    arrow(axis, (0.39, 0.49), (0.42, 0.49), color=C["model"], width=1.0, scale=7)

    initial = small_graph(axis, (0.65, 0.49), updated=False)
    arrow(axis, (0.55, 0.49), (initial[0] - 0.025, initial[1]), color=C["model"], width=1.1, scale=8)
    neural_block(axis, 0.73, 0.38, 0.10, 0.22, "Queen residual layer ×2")
    arrow(axis, (0.71, 0.49), (0.73, 0.49), color=C["model"], width=1.1, scale=8)
    updated = small_graph(axis, (0.90, 0.49), updated=True)
    arrow(axis, (0.83, 0.49), (updated[0] - 0.025, updated[1]), color=C["model"], width=1.1, scale=8)
    axis.text(0.65, 0.25, "LSOA states", ha="center", va="center", fontsize=7.1, color=C["muted"])
    axis.text(0.90, 0.25, r"graph state  $h_i^{(G)}$", ha="center", va="center", fontsize=7.1, color=C["muted"])

    axis.add_patch(display_circle(axis, (0.72, 0.15), 0.027, facecolor=C["white"], edgecolor=C["model"], linewidth=1.0, zorder=8))
    axis.text(0.72, 0.15, "g", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C["model"], zorder=9)
    arrow(axis, (0.49, 0.43), (0.69, 0.17), color=C["model"], width=0.9, rad=0.08, scale=7)
    arrow(axis, (0.90, 0.39), (0.75, 0.17), color=C["model"], width=0.9, rad=-0.08, scale=7)
    module_box(axis, 0.80, 0.08, 0.12, 0.10, "Readout\n" + r"$\hat r_i^{GNN}$", edge=C["model"], fill=C["white"], fontsize=7.2)
    arrow(axis, (0.75, 0.15), (0.80, 0.13), color=C["model"], width=1.0, scale=7)


def draw_tree(axis: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    levels = [[(cx, cy + 0.065 * scale)], [(cx - 0.040 * scale, cy), (cx + 0.040 * scale, cy)], [(cx - 0.064 * scale, cy - 0.060 * scale), (cx - 0.018 * scale, cy - 0.060 * scale), (cx + 0.018 * scale, cy - 0.060 * scale), (cx + 0.064 * scale, cy - 0.060 * scale)]]
    for parent, children in [(levels[0][0], levels[1]), (levels[1][0], levels[2][:2]), (levels[1][1], levels[2][2:])]:
        for child in children:
            axis.plot([parent[0], child[0]], [parent[1], child[1]], color=C["tabular"], linewidth=0.9, zorder=3)
    for level in levels:
        for point in level:
            axis.add_patch(display_circle(axis, point, 0.010 * scale, facecolor=C["tabular"], edgecolor=C["white"], linewidth=0.4, zorder=6))


def transformer(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    module_box(axis, x, y, width, height, "", edge=C["tabular"], fill=C["tabular_light"])
    left = [(x + width * 0.25, y + height * value) for value in (0.25, 0.50, 0.75)]
    right = [(x + width * 0.75, y + height * value) for value in (0.25, 0.50, 0.75)]
    for start in left:
        for end in right:
            axis.plot([start[0], end[0]], [start[1], end[1]], color=C["tabular"], linewidth=0.5, alpha=0.45, zorder=4)
    for point in left + right:
        axis.add_patch(display_circle(axis, point, 0.007, facecolor=C["tabular"], edgecolor=C["white"], linewidth=0.3, zorder=7))


def plot_prediction(axis: plt.Axes, case: dict, study: str, title: str) -> None:
    predictions = pd.read_parquet(Path("results/heterogeneous_graph_ensemble") / f"{study}.parquet")[["lsoa21cd", "prediction_heterogeneous_graph_ensemble"]]
    mapped = case["lsoas"][["lsoa21cd", "geometry"]].merge(predictions, on="lsoa21cd", how="left")
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=case["lsoas"].crs)
    mapped["geometry"] = mapped.geometry.simplify(90)
    cmap = mcolors.LinearSegmentedColormap.from_list("prediction", ["#EFF4F5", "#9FC8C0", "#3C8B67", "#195B45"])
    mapped.plot(ax=axis, column="prediction_heterogeneous_graph_ensemble", cmap=cmap, linewidth=0, rasterized=True)
    case["boundary"].boundary.plot(ax=axis, color=C["area_edge"], linewidth=0.5)
    axis.set_axis_off()
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=7.4, fontweight="bold", color=C["ink"], pad=1)


def draw_prediction(axis: plt.Axes, london: dict, manchester: dict) -> None:
    axis.text(0.12, 0.80, "Context-HGB", ha="center", va="center", fontsize=7.4, fontweight="bold", color=C["ink"])
    draw_tree(axis, 0.12, 0.69, scale=0.85)
    axis.add_patch(display_circle(axis, (0.31, 0.69), 0.028, facecolor=C["white"], edgecolor=C["model"], linewidth=1.0, zorder=8))
    axis.text(0.31, 0.69, "+", ha="center", va="center", fontsize=12, fontweight="bold", color=C["model"], zorder=9)
    arrow(axis, (0.20, 0.69), (0.28, 0.69), color=C["tabular"], width=1.0, scale=7)
    arrow(axis, (0.00, 0.55), (0.285, 0.67), color=C["model"], width=1.1, rad=-0.10, scale=7)
    module_box(axis, 0.37, 0.64, 0.18, 0.10, "graph hybrid", edge=C["model"], fill=C["model_light"], fontsize=7.2)
    arrow(axis, (0.34, 0.69), (0.37, 0.69), color=C["model"], width=1.0, scale=7)

    transformer(axis, 0.06, 0.40, 0.13, 0.13)
    axis.text(0.125, 0.35, "Context TabPFN", ha="center", va="center", fontsize=7.2, color=C["muted"])
    centre = (0.48, 0.44)
    axis.add_patch(display_wedge(axis, centre, 0.038, 90, 270, facecolor=C["tabular"], edgecolor=C["white"], linewidth=0.6, zorder=7))
    axis.add_patch(display_wedge(axis, centre, 0.038, -90, 90, facecolor=C["model"], edgecolor=C["white"], linewidth=0.6, zorder=7))
    axis.text(centre[0], centre[1], "1:1", ha="center", va="center", fontsize=6.8, fontweight="bold", color=C["white"], zorder=8)
    arrow(axis, (0.19, 0.46), (0.44, 0.44), color=C["tabular"], width=1.0, scale=7)
    arrow(axis, (0.46, 0.64), (0.48, 0.48), color=C["model"], width=1.0, rad=0.05, scale=7)
    axis.text(0.48, 0.35, "fixed equal-weight HFGE", ha="center", va="center", fontsize=7.1, color=C["muted"])

    london_axis = axis.inset_axes([0.60, 0.48, 0.24, 0.29])
    manchester_axis = axis.inset_axes([0.79, 0.49, 0.19, 0.27])
    plot_prediction(london_axis, london, "greater_london", "London")
    plot_prediction(manchester_axis, manchester, "greater_manchester", "Manchester")
    arrow(axis, (0.52, 0.44), (0.66, 0.46), color=C["ink"], width=1.1, scale=8)
    axis.text(0.79, 0.34, "strict out-of-fold spatial predictions", ha="center", va="center", fontsize=7.2, color=C["muted"])

    gradient = np.linspace(0, 1, 128).reshape(1, -1)
    ramp = axis.inset_axes([0.67, 0.25, 0.22, 0.025])
    cmap = mcolors.LinearSegmentedColormap.from_list("prediction_legend", ["#EFF4F5", "#9FC8C0", "#3C8B67", "#195B45"])
    ramp.imshow(gradient, aspect="auto", cmap=cmap)
    ramp.set_axis_off()
    axis.text(0.65, 0.262, "low", ha="right", va="center", fontsize=6.6, color=C["muted"])
    axis.text(0.91, 0.262, "high", ha="left", va="center", fontsize=6.6, color=C["muted"])


def modular_panel(
    figure: plt.Figure,
    bounds: tuple[float, float, float, float],
    letter: str,
    title: str,
) -> plt.Axes:
    """Create a restrained, dashed module panel similar to journal architecture figures."""
    x, y, width, height = bounds
    figure.patches.append(
        Rectangle(
            (x, y),
            width,
            height,
            transform=figure.transFigure,
            facecolor=C["white"],
            edgecolor=C["ink"],
            linewidth=0.9,
            linestyle=(0, (7, 5)),
            zorder=-3,
        )
    )
    axis = figure.add_axes(bounds)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.text(
        0.035,
        0.970,
        f"({letter})  {title}",
        ha="left",
        va="top",
        fontsize=10.6,
        fontweight="bold",
        color=C["ink"],
    )
    return axis


def submodule(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    *,
    edge: str = C["border"],
    fill: str = C["white"],
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.8,
            zorder=1,
        )
    )
    axis.text(
        x + 0.018,
        y + height - 0.025,
        title,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=C["ink"],
        zorder=12,
    )


def draw_facility_icon(
    axis: plt.Axes,
    x: float,
    y: float,
    kind: str,
    *,
    scale: float = 0.032,
    label: str | None = None,
) -> None:
    color = RELATION_STYLE[kind]
    axis.add_patch(display_circle(axis, (x, y), scale, facecolor=mcolors.to_rgba(color, 0.10), edgecolor=color, linewidth=0.8, zorder=6))
    if kind == "GP":
        axis.add_patch(Rectangle((x - scale * 0.15, y - scale * 0.58), scale * 0.30, scale * 1.16, facecolor=color, edgecolor="none", zorder=8))
        axis.add_patch(Rectangle((x - scale * 0.58, y - scale * 0.15), scale * 1.16, scale * 0.30, facecolor=color, edgecolor="none", zorder=8))
    elif kind == "Hospital":
        axis.add_patch(Rectangle((x - scale * 0.52, y - scale * 0.58), scale * 1.04, scale * 1.08, facecolor=C["white"], edgecolor=color, linewidth=0.7, zorder=7))
        axis.add_patch(Rectangle((x - scale * 0.10, y - scale * 0.45), scale * 0.20, scale * 0.42, facecolor=color, edgecolor="none", zorder=8))
        axis.add_patch(Rectangle((x - scale * 0.10, y + scale * 0.12), scale * 0.20, scale * 0.34, facecolor=color, edgecolor="none", zorder=8))
        axis.add_patch(Rectangle((x - scale * 0.28, y + scale * 0.19), scale * 0.56, scale * 0.20, facecolor=color, edgecolor="none", zorder=8))
    elif kind == "Green":
        axis.add_patch(Rectangle((x - scale * 0.10, y - scale * 0.56), scale * 0.20, scale * 0.54, facecolor="#8A6B4B", edgecolor="none", zorder=7))
        for dx, dy, radius in ((-0.25, 0.05, 0.34), (0.23, 0.07, 0.34), (0.0, 0.32, 0.40)):
            axis.add_patch(display_circle(axis, (x + scale * dx, y + scale * dy), scale * radius, facecolor=color, edgecolor="none", zorder=8))
    else:
        axis.add_patch(FancyBboxPatch((x - scale * 0.55, y + scale * 0.08), scale * 1.10, scale * 0.30, boxstyle="round,pad=0.001,rounding_size=0.006", facecolor=color, edgecolor="none", zorder=8))
        axis.add_patch(Rectangle((x - scale * 0.50, y - scale * 0.10), scale, scale * 0.14, facecolor="#E3A43B", edgecolor="none", zorder=8))
        axis.add_patch(Rectangle((x - scale * 0.50, y - scale * 0.27), scale, scale * 0.12, facecolor=color, edgecolor="none", zorder=8))
        axis.add_patch(FancyBboxPatch((x - scale * 0.55, y - scale * 0.52), scale * 1.10, scale * 0.20, boxstyle="round,pad=0.001,rounding_size=0.005", facecolor="#E8B65C", edgecolor="none", zorder=8))
    if label:
        axis.text(x, y - scale - 0.018, label, ha="center", va="top", fontsize=5.8, color=C["muted"])


def draw_inputs_modular(axis: plt.Axes, london: dict, manchester: dict) -> None:
    axis.text(0.50, 0.885, "Case-study geography", ha="center", va="center", fontsize=7.4, fontweight="bold", color=C["ink"])
    london_axis = axis.inset_axes([0.08, 0.685, 0.38, 0.155])
    manchester_axis = axis.inset_axes([0.54, 0.685, 0.38, 0.155])
    plot_outline(london_axis, london["boundary"], "London")
    plot_outline(manchester_axis, manchester["boundary"], "Manchester")
    arrow(axis, (0.50, 0.675), (0.50, 0.615), width=0.9, scale=7)

    submodule(axis, 0.075, 0.405, 0.85, 0.205, "Node attributes")
    feature_matrix(axis, 0.12, 0.455, rows=4, columns=5)
    axis.text(0.205, 0.435, "103 area attributes", ha="center", va="top", fontsize=6.6, color=C["muted"])
    icon_specs = [
        (0.49, 0.535, "GP", "GP"),
        (0.73, 0.535, "Hospital", "Hospital"),
        (0.49, 0.455, "Green", "Green space"),
        (0.73, 0.455, "Food", "Fast food"),
    ]
    for x, y, kind, label in icon_specs:
        draw_facility_icon(axis, x, y, kind, scale=0.022)
        axis.text(x + 0.045, y, label, ha="left", va="center", fontsize=5.8, color=C["muted"])
    arrow(axis, (0.50, 0.395), (0.50, 0.335), width=0.9, scale=7)

    submodule(axis, 0.075, 0.125, 0.85, 0.205, "Spatial evaluation protocol", fill="#F7F8FA")
    for shift in (0.00, 0.016, 0.032):
        axis.add_patch(Rectangle((0.12 + shift, 0.165 + shift), 0.20, 0.088, facecolor=C["white"], edgecolor=C["area_edge"], linewidth=0.65, zorder=5))
    axis.text(0.24, 0.148, "5 contiguous folds", ha="center", va="top", fontsize=7.0, color=C["muted"])
    arrow(axis, (0.39, 0.215), (0.49, 0.215), color=C["area_edge"], width=0.8, scale=6)
    module_box(axis, 0.52, 0.160, 0.36, 0.105, "Fit on training folds\nTest on held-out fold", edge=C["area_edge"], fill=C["white"], fontsize=6.1)


def draw_queen_grid(axis: plt.Axes, x: float, y: float, size: float) -> None:
    cell = size / 3
    centres: dict[tuple[int, int], tuple[float, float]] = {}
    for row in range(3):
        for col in range(3):
            cx = x + (col + 0.5) * cell
            cy = y + (row + 0.5) * cell
            centres[(row, col)] = (cx, cy)
            target = row == 1 and col == 1
            axis.add_patch(
                Rectangle(
                    (x + col * cell, y + row * cell),
                    cell,
                    cell,
                    facecolor=C["target"] if target else C["area"],
                    edgecolor=C["white"],
                    linewidth=1.2,
                    zorder=3,
                )
            )
    centre = centres[(1, 1)]
    for key, point in centres.items():
        if key != (1, 1):
            arrow(axis, centre, point, color=C["area_edge"], width=0.65, alpha=0.85, scale=5, zorder=6)
    axis.text(*centre, r"$u_i$", ha="center", va="center", fontsize=7.1, fontweight="bold", color=C["white"], zorder=9)


def draw_graph_construction_modular(axis: plt.Axes, case: dict[str, gpd.GeoDataFrame]) -> None:
    """Show the two edge families together in their actual spatial setting."""
    lsoas = case["lsoas"].to_crs(27700).copy()
    facilities = case["facilities"].to_crs(27700).copy()
    centres = lsoas.geometry.representative_point()
    london_centre = np.array([530200.0, 181200.0])
    distance_to_centre = np.hypot(centres.x - london_centre[0], centres.y - london_centre[1])
    candidates = np.argsort(distance_to_centre.to_numpy())[:30]
    touch_counts = {
        int(index): int(lsoas.geometry.touches(lsoas.geometry.iloc[int(index)]).sum())
        for index in candidates
    }
    target_index = max(touch_counts, key=touch_counts.get)
    target_point = centres.iloc[target_index]
    neighbour_mask = lsoas.geometry.touches(lsoas.geometry.iloc[target_index])
    neighbour_indices = list(np.flatnonzero(neighbour_mask.to_numpy()))
    local_distance = np.hypot(centres.x - target_point.x, centres.y - target_point.y)
    local_indices = list(np.argsort(local_distance.to_numpy())[:22])
    display_indices = sorted(set(local_indices + neighbour_indices + [target_index]))
    local_lsoas = lsoas.iloc[display_indices]

    map_axis = axis.inset_axes([0.055, 0.215, 0.89, 0.650])
    local_lsoas.plot(ax=map_axis, color="#F1F4F6", edgecolor=C["border"], linewidth=0.55, zorder=1)
    lsoas.iloc[neighbour_indices].plot(ax=map_axis, color="#E4EDF0", edgecolor=C["area_edge"], linewidth=0.85, zorder=2)
    lsoas.iloc[[target_index]].plot(ax=map_axis, color=C["target"], edgecolor=C["target"], linewidth=1.0, zorder=3)

    target_xy = (float(target_point.x), float(target_point.y))
    for neighbour_index in neighbour_indices:
        neighbour_point = centres.iloc[neighbour_index]
        map_axis.plot(
            [target_xy[0], float(neighbour_point.x)],
            [target_xy[1], float(neighbour_point.y)],
            color=C["area_edge"],
            linewidth=0.75,
            alpha=0.80,
            zorder=4,
        )
        map_axis.scatter(neighbour_point.x, neighbour_point.y, s=6, color=C["area_edge"], zorder=5)

    category_map = {
        "GP": ("clinic_doctor", 700.0),
        "Hospital": ("hospital", 1450.0),
        "Green": ("greenspace", 1050.0),
        "Food": ("fast_food", 850.0),
    }
    for relation, (category, desired_distance) in category_map.items():
        selected = facilities[facilities["category"].eq(category)].copy()
        distances = selected.geometry.distance(target_point)
        pool = selected.loc[distances.le(2600.0)].copy()
        if pool.empty:
            pool = selected.iloc[np.argsort(distances.to_numpy())[:20]].copy()
        pool_distance = pool.geometry.distance(target_point)
        facility = pool.iloc[int(np.argmin(np.abs(pool_distance.to_numpy() - desired_distance)))]
        fx, fy = float(facility.geometry.x), float(facility.geometry.y)
        map_axis.annotate(
            "",
            xy=target_xy,
            xytext=(fx, fy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": RELATION_STYLE[relation],
                "lw": 1.2,
                "mutation_scale": 8,
                "shrinkA": 7,
                "shrinkB": 6,
            },
            zorder=7,
        )
        map_axis.scatter(fx, fy, s=72, color=RELATION_STYLE[relation], edgecolor=C["white"], linewidth=0.8, zorder=8)
        short = {"GP": "GP", "Hospital": "H", "Green": "G", "Food": "F"}[relation]
        map_axis.text(fx, fy, short, ha="center", va="center", fontsize=6.6, fontweight="bold", color=C["white"], zorder=9)

    map_axis.scatter(*target_xy, s=17, color=C["white"], edgecolor=C["target"], linewidth=0.7, zorder=10)
    map_axis.text(target_xy[0], target_xy[1], "  target area", ha="left", va="center", fontsize=6.5, fontweight="bold", color=C["white"], zorder=11)
    map_axis.set_aspect("equal")
    map_axis.set_axis_off()
    map_axis.set_title("One spatial graph: accessibility edges and Queen adjacency", fontsize=7.7, fontweight="bold", color=C["ink"], pad=3)

    axis.plot([0.08, 0.16], [0.150, 0.150], color=C["gp"], linewidth=1.4)
    axis.add_patch(FancyArrowPatch((0.08, 0.150), (0.16, 0.150), arrowstyle="-|>", mutation_scale=7, linewidth=1.0, color=C["gp"], zorder=7))
    axis.text(0.18, 0.150, "facility-to-area edge: category, travel time and quality", ha="left", va="center", fontsize=6.6, color=C["ink"])
    axis.plot([0.08, 0.16], [0.095, 0.095], color=C["area_edge"], linewidth=1.0)
    axis.text(0.18, 0.095, "Queen edge between areas sharing a boundary or vertex", ha="left", va="center", fontsize=6.6, color=C["ink"])
    axis.text(0.93, 0.050, "Unified graph with typed facility edges and Queen adjacency", ha="right", va="center", fontsize=6.7, fontweight="bold", color=C["model"])


def draw_dense_network(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
    layers: tuple[int, ...] = (3, 5, 3),
    label: str | None = None,
    label_y: float = -0.018,
) -> None:
    """Draw a neural module as connected layers rather than a labelled box."""
    layer_x = np.linspace(x, x + width, len(layers))
    coordinates: list[list[tuple[float, float]]] = []
    for lx, count in zip(layer_x, layers, strict=True):
        ys = np.linspace(y, y + height, count) if count > 1 else np.array([y + height / 2])
        coordinates.append([(float(lx), float(node_y)) for node_y in ys])
    for left, right in zip(coordinates[:-1], coordinates[1:], strict=True):
        for start in left:
            for end in right:
                axis.plot([start[0], end[0]], [start[1], end[1]], color=mcolors.to_rgba(color, 0.35), linewidth=0.42, zorder=3)
    radius = min(0.008, height / 14)
    for layer in coordinates:
        for point in layer:
            axis.add_patch(display_circle(axis, point, radius, facecolor=C["white"], edgecolor=color, linewidth=0.75, zorder=6))
    if label:
        axis.text(x + width / 2, y + label_y, label, ha="center", va="top", fontsize=6.6, color=C["ink"])


def draw_feature_bars(axis: plt.Axes, x: float, y: float, color: str) -> None:
    heights = (0.032, 0.052, 0.041, 0.061, 0.047)
    for index, height in enumerate(heights):
        axis.add_patch(Rectangle((x + index * 0.012, y - height / 2), 0.008, height, facecolor=mcolors.to_rgba(color, 0.48 + index * 0.08), edgecolor="none", zorder=6))


def draw_gravity_curve(axis: plt.Axes, x: float, y: float, width: float, height: float, color: str) -> None:
    axis.plot([x, x, x + width], [y, y + height, y], color=C["border"], linewidth=0.55, zorder=3)
    values = np.linspace(0, 1, 30)
    axis.plot(x + width * values, y + height * 0.82 * np.exp(-2.6 * values), color=color, linewidth=1.15, zorder=6)


def gravity_encoder_row(axis: plt.Axes, y: float, relation: str) -> None:
    color = RELATION_STYLE[relation]
    facility_node(axis, 0.050, y, relation, radius=0.022)
    draw_feature_bars(axis, 0.105, y, color)
    arrow(axis, (0.165, y), (0.205, y), color=color, width=0.7, scale=5)
    draw_dense_network(axis, 0.220, y - 0.042, 0.155, 0.084, color=color, layers=(3, 5, 3))
    axis.add_patch(display_circle(axis, (0.425, y), 0.018, facecolor=C["white"], edgecolor=C["ink"], linewidth=0.75, zorder=7))
    axis.text(0.425, y, "apply", ha="center", va="center", fontsize=4.9, color=C["ink"], zorder=8)
    draw_gravity_curve(axis, 0.485, y - 0.030, 0.110, 0.060, color)
    axis.add_patch(display_circle(axis, (0.690, y), 0.023, facecolor=C["white"], edgecolor=color, linewidth=0.9, zorder=7))
    axis.text(0.690, y, "pool", ha="center", va="center", fontsize=5.1, fontweight="bold", color=color, zorder=8)
    for index in range(4):
        axis.add_patch(Rectangle((0.790 + index * 0.026, y - 0.028), 0.018, 0.056, facecolor=mcolors.to_rgba(color, 0.36 + index * 0.13), edgecolor="none", zorder=6))
    for start, end in [((0.074, y), (0.100, y)), ((0.375, y), (0.405, y)), ((0.445, y), (0.482, y)), ((0.598, y), (0.665, y)), ((0.715, y), (0.785, y))]:
        arrow(axis, start, end, color=color, width=0.7, scale=5)


def draw_relation_encoder_modular(axis: plt.Axes) -> None:
    headings = [(0.130, "edge state"), (0.298, "MLP"), (0.540, "decay"), (0.690, "pool"), (0.835, "vector")]
    for x, text_value in headings:
        axis.text(x, 0.860, text_value, ha="center", va="center", fontsize=6.7, color=C["muted"])
    for y, relation in zip([0.740, 0.590, 0.440, 0.290], RELATION_STYLE):
        gravity_encoder_row(axis, y, relation)
    axis.text(0.13, 0.200, "Travel time, facility quality and nonlinear edge terms", ha="left", va="center", fontsize=6.2, color=C["muted"])
    colors = list(RELATION_STYLE.values())
    for row, color in enumerate(colors):
        for column in range(6):
            axis.add_patch(Rectangle((0.135 + column * 0.031, 0.085 + row * 0.021), 0.025, 0.016, facecolor=mcolors.to_rgba(color, 0.30 + column * 0.09), edgecolor="none", zorder=6))
    axis.text(0.60, 0.118, "relation-preserving facility representation", ha="center", va="center", fontsize=7.0, fontweight="bold", color=C["model"])


def draw_state_graph(axis: plt.Axes, cx: float, cy: float, *, updated: bool = False) -> None:
    colour = C["model"] if updated else C["area_edge"]
    points = np.array([[0.0, 0.0], [-0.050, 0.055], [0.050, 0.055], [-0.060, -0.045], [0.060, -0.045]]) + np.array([cx, cy])
    for index in range(1, len(points)):
        axis.plot([points[0, 0], points[index, 0]], [points[0, 1], points[index, 1]], color=colour, linewidth=0.8, zorder=3)
    for index, point in enumerate(points):
        axis.add_patch(display_circle(axis, tuple(point), 0.010 if index else 0.014, facecolor=colour if index == 0 else C["white"], edgecolor=colour, linewidth=0.8, zorder=6))


def draw_state_vector(axis: plt.Axes, x: float, y: float, color: str, *, count: int = 6) -> None:
    for index in range(count):
        opacity = 0.25 + 0.11 * index
        axis.add_patch(Rectangle((x + index * 0.018, y - 0.020), 0.013, 0.040, facecolor=mcolors.to_rgba(color, opacity), edgecolor="none", zorder=6))


def draw_operator(axis: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    axis.add_patch(display_circle(axis, (x, y), 0.023, facecolor=C["white"], edgecolor=color, linewidth=0.8, zorder=8))
    axis.text(x, y, label, ha="center", va="center", fontsize=8.0 if len(label) <= 2 else 5.2, fontweight="bold", color=color, zorder=9)


def thin_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    rad: float = 0.0,
) -> None:
    arrow(axis, start, end, color=color, width=0.62, scale=5.2, rad=rad)


def routed_arrow(
    axis: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
) -> None:
    """Route a connector through whitespace and place the arrowhead only at entry."""
    if len(points) < 2:
        raise ValueError("a routed arrow requires at least two points")
    if len(points) > 2:
        axis.plot(
            [point[0] for point in points[:-1]],
            [point[1] for point in points[:-1]],
            color=color,
            linewidth=0.62,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )
    thin_arrow(axis, points[-2], points[-1], color=color)


def draw_message_graph(
    axis: plt.Axes,
    centre: tuple[float, float],
    *,
    scale: float,
    stage: int,
    boxed: bool = False,
) -> None:
    """Draw a compact graph-message-passing state, not a dense MLP."""
    cx, cy = centre
    y_scale = axis_y_compensation(axis)
    if boxed:
        axis.add_patch(
            FancyBboxPatch(
                (cx - scale * 1.55, cy - scale * y_scale * 1.45),
                scale * 3.10,
                scale * y_scale * 2.90,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=C["white"],
                edgecolor=C["model"],
                linewidth=0.8,
                zorder=2,
            )
        )
    offsets = [
        (0.0, 0.0),
        (-0.95, 0.0),
        (0.95, 0.0),
        (-0.48, 0.82),
        (0.48, 0.82),
        (-0.48, -0.82),
        (0.48, -0.82),
    ]
    nodes = [(cx + scale * dx, cy + scale * y_scale * dy) for dx, dy in offsets]
    edges = [(0, index) for index in range(1, 7)] + [(1, 3), (3, 4), (4, 2), (1, 5), (5, 6), (6, 2)]
    for first, second in edges:
        axis.plot(
            [nodes[first][0], nodes[second][0]],
            [nodes[first][1], nodes[second][1]],
            color=mcolors.to_rgba(C["model"], 0.48 if stage else 0.32),
            linewidth=0.65,
            zorder=3,
        )
    if stage:
        for index in (1, 3, 5):
            start = nodes[index]
            end = (cx + (start[0] - cx) * 0.20, cy + (start[1] - cy) * 0.20)
            thin_arrow(axis, start, end, color=C["model"])
    for index, point in enumerate(nodes):
        if index == 0:
            fill = C["model"]
        elif stage == 2:
            fill = C["model_light"]
        elif stage == 1 and index in (1, 3, 5):
            fill = mcolors.to_rgba(C["model"], 0.20)
        else:
            fill = C["white"]
        axis.add_patch(
            display_circle(
                axis,
                point,
                scale * (0.24 if index == 0 else 0.19),
                facecolor=fill,
                edgecolor=C["model"] if stage else C["area_edge"],
                linewidth=0.75,
                zorder=7,
            )
        )


def draw_gnn_modular(axis: plt.Axes) -> None:
    """Draw the heterogeneous GNN with explicit graph-to-graph message passing."""
    # Top row: tabular area context and typed facility relations are encoded separately.
    axis.text(0.105, 0.858, "Area attributes", ha="center", va="center", fontsize=6.7, fontweight="bold", color=C["area_edge"])
    draw_feature_bars(axis, 0.045, 0.800, C["area_edge"])
    draw_dense_network(axis, 0.135, 0.758, 0.165, 0.084, color=C["area_edge"], layers=(4, 6, 3))
    draw_state_vector(axis, 0.330, 0.800, C["area_edge"], count=4)
    axis.text(0.218, 0.720, "area encoder", ha="center", va="center", fontsize=6.2, color=C["area_edge"])
    axis.text(0.365, 0.720, "area state", ha="center", va="center", fontsize=6.2, color=C["area_edge"])
    thin_arrow(axis, (0.105, 0.800), (0.125, 0.800), color=C["area_edge"])
    thin_arrow(axis, (0.302, 0.800), (0.323, 0.800), color=C["area_edge"])

    axis.text(0.620, 0.858, "Facility relations", ha="center", va="center", fontsize=6.7, fontweight="bold", color=C["model"])
    for row, color in enumerate(RELATION_STYLE.values()):
        axis.add_patch(Rectangle((0.565, 0.775 + row * 0.016), 0.065, 0.012, facecolor=mcolors.to_rgba(color, 0.45), edgecolor="none", zorder=6))
    draw_dense_network(axis, 0.675, 0.758, 0.165, 0.084, color=C["model"], layers=(4, 6, 3))
    draw_state_vector(axis, 0.870, 0.800, C["model"], count=4)
    axis.text(0.758, 0.720, "facility encoder", ha="center", va="center", fontsize=6.2, color=C["model"])
    axis.text(0.905, 0.720, "facility state", ha="center", va="center", fontsize=6.2, color=C["model"])
    thin_arrow(axis, (0.630, 0.800), (0.665, 0.800), color=C["model"])
    thin_arrow(axis, (0.842, 0.800), (0.865, 0.800), color=C["model"])

    # Fusion is placed in its own vertical band; labels sit outside the drawings.
    draw_operator(axis, 0.500, 0.684, "join", C["model"])
    routed_arrow(axis, [(0.365, 0.778), (0.365, 0.684), (0.475, 0.684)], color=C["area_edge"])
    routed_arrow(axis, [(0.905, 0.778), (0.905, 0.684), (0.525, 0.684)], color=C["model"])
    draw_dense_network(axis, 0.420, 0.575, 0.160, 0.068, color=C["model"], layers=(4, 6, 3))
    axis.text(0.500, 0.553, "joint-fusion MLP", ha="center", va="center", fontsize=6.2, color=C["model"])
    thin_arrow(axis, (0.500, 0.660), (0.500, 0.646), color=C["model"])
    draw_state_vector(axis, 0.455, 0.510, C["model"], count=6)
    axis.text(0.510, 0.478, "joint state", ha="center", va="center", fontsize=6.2, color=C["model"])
    thin_arrow(axis, (0.500, 0.571), (0.500, 0.535), color=C["model"])

    # Middle row: graph states remain graphs throughout both GraphConv layers.
    axis.text(0.500, 0.445, "Queen-neighbour graph message passing", ha="center", va="center", fontsize=7.0, fontweight="bold", color=C["model"])
    draw_message_graph(axis, (0.120, 0.365), scale=0.045, stage=0, boxed=False)
    draw_message_graph(axis, (0.365, 0.365), scale=0.042, stage=1, boxed=True)
    draw_message_graph(axis, (0.635, 0.365), scale=0.042, stage=2, boxed=True)
    draw_state_vector(axis, 0.840, 0.365, C["model"], count=5)
    thin_arrow(axis, (0.175, 0.365), (0.300, 0.365), color=C["area_edge"])
    thin_arrow(axis, (0.430, 0.365), (0.570, 0.365), color=C["model"])
    thin_arrow(axis, (0.700, 0.365), (0.832, 0.365), color=C["model"])
    axis.text(0.120, 0.292, "Queen graph", ha="center", va="center", fontsize=6.2, color=C["area_edge"])
    axis.text(0.365, 0.292, "GraphConv layer 1", ha="center", va="center", fontsize=6.2, color=C["model"])
    axis.text(0.635, 0.292, "GraphConv layer 2", ha="center", va="center", fontsize=6.2, color=C["model"])
    axis.text(0.880, 0.292, "graph state", ha="center", va="center", fontsize=6.2, color=C["model"])
    routed_arrow(axis, [(0.485, 0.492), (0.485, 0.462), (0.120, 0.462), (0.120, 0.420)], color=C["model"])

    # Bottom row: a clean gated skip/readout path with dedicated label space.
    draw_state_vector(axis, 0.080, 0.180, C["model"], count=4)
    draw_state_vector(axis, 0.235, 0.180, C["model"], count=4)
    axis.add_patch(
        FancyBboxPatch(
            (0.385, 0.158),
            0.090,
            0.044,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            facecolor=C["model_light"],
            edgecolor=C["model"],
            linewidth=0.8,
            zorder=8,
        )
    )
    axis.text(0.430, 0.180, "gate", ha="center", va="center", fontsize=5.8, fontweight="bold", color=C["model"], zorder=9)
    draw_state_vector(axis, 0.515, 0.180, C["model"], count=4)
    draw_state_vector(axis, 0.650, 0.180, C["area_edge"], count=3)
    draw_operator(axis, 0.735, 0.180, "join", C["model"])
    draw_dense_network(axis, 0.785, 0.148, 0.120, 0.064, color=C["model"], layers=(4, 5, 1))
    axis.add_patch(display_circle(axis, (0.955, 0.180), 0.017, facecolor=C["model"], edgecolor=C["model"], linewidth=0.7, zorder=7))

    routed_arrow(axis, [(0.145, 0.180), (0.175, 0.148), (0.350, 0.148), (0.382, 0.168)], color=C["model"])
    thin_arrow(axis, (0.300, 0.180), (0.380, 0.180), color=C["model"])
    thin_arrow(axis, (0.478, 0.180), (0.508, 0.180), color=C["model"])
    routed_arrow(axis, [(0.580, 0.180), (0.600, 0.148), (0.700, 0.148), (0.712, 0.168)], color=C["model"])
    thin_arrow(axis, (0.700, 0.180), (0.710, 0.180), color=C["area_edge"])
    thin_arrow(axis, (0.760, 0.180), (0.780, 0.180), color=C["model"])
    thin_arrow(axis, (0.910, 0.180), (0.935, 0.180), color=C["model"])

    axis.text(0.110, 0.235, "joint", ha="center", va="center", fontsize=6.1, color=C["model"])
    axis.text(0.265, 0.235, "graph", ha="center", va="center", fontsize=6.1, color=C["model"])
    axis.text(0.545, 0.235, "context", ha="center", va="center", fontsize=6.1, color=C["model"])
    axis.text(0.670, 0.235, "area skip", ha="center", va="center", fontsize=6.1, color=C["area_edge"])
    axis.text(0.845, 0.112, "readout", ha="center", va="center", fontsize=5.9, color=C["model"])
    axis.text(0.955, 0.120, "GNN\nresidual", ha="center", va="center", fontsize=5.8, linespacing=0.92, fontweight="bold", color=C["model"])
    axis.text(0.500, 0.067, "Repeated state bars denote gated skip connections", ha="center", va="center", fontsize=5.9, color=C["muted"])


def draw_prediction_modular(axis: plt.Axes, london: dict, manchester: dict) -> None:
    # Stage 1: HGB base prediction plus a training-fold-selected GNN residual.
    axis.text(0.07, 0.865, "1  Residual graph hybrid", ha="left", va="center", fontsize=7.3, fontweight="bold", color=C["ink"])
    draw_tree(axis, 0.190, 0.755, scale=0.50)
    axis.text(0.190, 0.680, "Context-HGB prediction", ha="center", va="top", fontsize=6.3, color=C["tabular"])
    axis.add_patch(display_circle(axis, (0.780, 0.755), 0.022, facecolor=C["model"], edgecolor=C["model"], linewidth=0.7, zorder=8))
    axis.text(0.780, 0.715, "weighted GNN residual", ha="center", va="top", fontsize=6.1, color=C["model"])
    draw_operator(axis, 0.490, 0.755, "add", C["model"])
    thin_arrow(axis, (0.300, 0.755), (0.462, 0.755), color=C["tabular"])
    thin_arrow(axis, (0.755, 0.755), (0.518, 0.755), color=C["model"])
    draw_state_vector(axis, 0.435, 0.635, C["model"], count=7)
    thin_arrow(axis, (0.490, 0.730), (0.490, 0.662), color=C["model"])
    axis.text(0.500, 0.585, "Graph hybrid prediction", ha="center", va="center", fontsize=6.7, fontweight="bold", color=C["model"])
    axis.text(0.500, 0.550, "Residual weight selected only within training folds", ha="center", va="center", fontsize=5.8, color=C["muted"])

    # Stage 2: transparent fixed 50:50 ensemble.
    axis.plot([0.07, 0.93], [0.515, 0.515], color=C["border"], linewidth=0.55)
    axis.text(0.07, 0.485, "2  Fixed HFGE ensemble", ha="left", va="center", fontsize=7.3, fontweight="bold", color=C["ink"])
    draw_dense_network(axis, 0.105, 0.355, 0.190, 0.095, color=C["tabular"], layers=(4, 7, 4), label="Context-TabPFN")
    draw_state_vector(axis, 0.725, 0.402, C["model"], count=6)
    axis.text(0.775, 0.350, "graph hybrid prediction", ha="center", va="top", fontsize=6.0, color=C["model"])
    centre = (0.500, 0.402)
    axis.add_patch(display_wedge(axis, centre, 0.034, 90, 270, facecolor=C["tabular"], edgecolor=C["white"], linewidth=0.5, zorder=7))
    axis.add_patch(display_wedge(axis, centre, 0.034, -90, 90, facecolor=C["model"], edgecolor=C["white"], linewidth=0.5, zorder=7))
    axis.text(*centre, "50:50", ha="center", va="center", fontsize=5.0, fontweight="bold", color=C["white"], zorder=8)
    thin_arrow(axis, (0.305, 0.402), (0.462, 0.402), color=C["tabular"])
    thin_arrow(axis, (0.715, 0.402), (0.538, 0.402), color=C["model"])
    axis.text(0.500, 0.310, "Final HFGE prediction", ha="center", va="center", fontsize=6.7, fontweight="bold", color=C["ink"])

    london_axis = axis.inset_axes([0.08, 0.075, 0.40, 0.155])
    manchester_axis = axis.inset_axes([0.52, 0.075, 0.40, 0.155])
    plot_prediction(london_axis, london, "greater_london", "London")
    plot_prediction(manchester_axis, manchester, "greater_manchester", "Manchester")
    thin_arrow(axis, (0.500, 0.288), (0.500, 0.248), color=C["ink"])
    axis.text(0.500, 0.045, "strict out-of-fold spatial predictions", ha="center", va="center", fontsize=6.4, color=C["muted"])


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": C["white"],
            "savefig.edgecolor": C["white"],
        }
    )
    london = load_case("greater_london")
    manchester = load_case("greater_manchester")
    figure = plt.figure(figsize=(21.0, 8.4), facecolor=C["white"])
    figure.text(0.5, 0.965, "Heterogeneous Facility Graph Ensemble (HFGE)", ha="center", va="center", fontsize=16.5, fontweight="bold", color=C["ink"])

    y, height = 0.115, 0.790
    inputs = modular_panel(figure, (0.015, y, 0.120, height), "a", "Spatial inputs")
    graph = modular_panel(figure, (0.151, y, 0.215, height), "b", "Spatial heterogeneous graph")
    encoder = modular_panel(figure, (0.382, y, 0.195, height), "c", "Typed relation encoder")
    area_gnn = modular_panel(figure, (0.593, y, 0.215, height), "d", "Heterogeneous facility GNN")
    prediction = modular_panel(figure, (0.824, y, 0.161, height), "e", "Two-stage HFGE prediction")

    draw_inputs_modular(inputs, london, manchester)
    draw_graph_construction_modular(graph, london)
    draw_relation_encoder_modular(encoder)
    draw_gnn_modular(area_gnn)
    draw_prediction_modular(prediction, london, manchester)

    for start, end in [((0.136, 0.510), (0.149, 0.510)), ((0.367, 0.510), (0.380, 0.510)), ((0.578, 0.510), (0.591, 0.510)), ((0.809, 0.510), (0.822, 0.510))]:
        figure_arrow(figure, start, end)
    figure.text(0.5, 0.050, "Five frozen contiguous spatial folds  •  training-fold-only fitting  •  strict out-of-fold evaluation", ha="center", va="center", fontsize=8.5, color=C["muted"])
    return figure


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    outputs = {}
    stem = args.output_dir / "figure1_hfge_framework"
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.08)
        outputs[suffix] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
    plt.close(figure)
    manifest = {
        "title": "Heterogeneous Facility Graph Ensemble (HFGE) framework",
        "status": "publication framework figure",
        "design": "modular left-to-right academic architecture with explicit graph construction",
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "figure1_hfge_framework_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
