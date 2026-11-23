from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "work" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmium
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import from_wkb
from shapely.geometry import Point


WALKING_METRES_PER_MINUTE = 80.0
THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
EXPLICIT_RESIDENTIAL = {
    "apartments",
    "bungalow",
    "cabin",
    "detached",
    "dormitory",
    "ger",
    "house",
    "houseboat",
    "residential",
    "semidetached_house",
    "static_caravan",
    "terrace",
}


class PolygonGridLocator:
    def __init__(self, polygons: gpd.GeoSeries, cell_size: float = 2000.0) -> None:
        self.polygons = list(polygons)
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, geometry in enumerate(self.polygons):
            min_x, min_y, max_x, max_y = geometry.bounds
            for cell_x in range(
                math.floor(min_x / self.cell_size), math.floor(max_x / self.cell_size) + 1
            ):
                for cell_y in range(
                    math.floor(min_y / self.cell_size), math.floor(max_y / self.cell_size) + 1
                ):
                    self.cells[(cell_x, cell_y)].append(index)

    def locate(self, x: float, y: float) -> int | None:
        key = (math.floor(x / self.cell_size), math.floor(y / self.cell_size))
        point = Point(x, y)
        for index in self.cells.get(key, ()):
            if self.polygons[index].covers(point):
                return index
        return None


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


class BuildingOriginHandler(osmium.SimpleHandler):
    """Keep deterministic bottom-hash samples of building anchors in each LSOA."""

    def __init__(
        self,
        locator: PolygonGridLocator,
        lsoa_count: int,
        sample_size: int,
    ) -> None:
        super().__init__()
        self.locator = locator
        self.transformer = Transformer.from_crs(4326, 27700, always_xy=True)
        self.factory = osmium.geom.WKBFactory()
        self.sample_size = int(sample_size)
        self.all_count = np.zeros(lsoa_count, dtype=np.int64)
        self.residential_count = np.zeros(lsoa_count, dtype=np.int64)
        self.all_heaps: list[list[tuple[int, float, float]]] = [
            [] for _ in range(lsoa_count)
        ]
        self.residential_heaps: list[list[tuple[int, float, float]]] = [
            [] for _ in range(lsoa_count)
        ]
        self.tag_counts: dict[str, int] = defaultdict(int)

    def _sample(
        self,
        heaps: list[list[tuple[int, float, float]]],
        index: int,
        key: int,
        x: float,
        y: float,
    ) -> None:
        heap = heaps[index]
        item = (-key, x, y)
        if len(heap) < self.sample_size:
            heapq.heappush(heap, item)
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, item)

    def area(self, area: osmium.osm.Area) -> None:
        building = area.tags.get("building")
        if not building:
            return
        try:
            geometry = from_wkb(bytes.fromhex(self.factory.create_multipolygon(area)))
        except Exception:
            return
        if geometry.is_empty:
            return
        anchor = geometry.representative_point()
        x, y = self.transformer.transform(anchor.x, anchor.y)
        index = self.locator.locate(x, y)
        if index is None:
            return
        value = str(building).strip().lower()
        self.tag_counts[value] += 1
        self.all_count[index] += 1
        object_id = int(area.orig_id()) * 2 + int(not area.from_way())
        key = splitmix64(object_id)
        self._sample(self.all_heaps, index, key, x, y)
        if value in EXPLICIT_RESIDENTIAL:
            self.residential_count[index] += 1
            self._sample(self.residential_heaps, index, key, x, y)

    @staticmethod
    def frame_from_heaps(
        heaps: list[list[tuple[int, float, float]]],
        lsoa_codes: np.ndarray,
        origin_type: str,
    ) -> pd.DataFrame:
        records: list[tuple[str, str, float, float]] = []
        for index, heap in enumerate(heaps):
            records.extend(
                (str(lsoa_codes[index]), origin_type, float(x), float(y))
                for _, x, y in heap
            )
        return pd.DataFrame(records, columns=["lsoa21cd", "origin_type", "x", "y"])


def load_codepoints(
    codepoint_dir: Path,
    bounds: tuple[float, float, float, float],
    buffer_metres: float = 5000.0,
) -> pd.DataFrame:
    min_x, min_y, max_x, max_y = bounds
    parts: list[pd.DataFrame] = []
    for path in sorted(codepoint_dir.glob("*.csv")):
        part = pd.read_csv(
            path,
            header=None,
            usecols=[0, 2, 3],
            names=["postcode", "x", "y"],
            dtype={"postcode": str, "x": float, "y": float},
        )
        selected = part[
            part["x"].between(min_x - buffer_metres, max_x + buffer_metres)
            & part["y"].between(min_y - buffer_metres, max_y + buffer_metres)
        ].copy()
        if len(selected):
            selected["postcode"] = (
                selected["postcode"].str.upper().str.replace(" ", "", regex=False)
            )
            parts.append(selected)
    return pd.concat(parts, ignore_index=True).drop_duplicates("postcode")


def assign_lsoa(
    points: pd.DataFrame,
    locator: PolygonGridLocator,
    lsoa_codes: np.ndarray,
    origin_type: str,
) -> pd.DataFrame:
    indices = np.fromiter(
        (
            -1 if (index := locator.locate(float(x), float(y))) is None else index
            for x, y in points[["x", "y"]].itertuples(index=False)
        ),
        dtype=np.int64,
        count=len(points),
    )
    valid = indices >= 0
    result = points.loc[valid, ["x", "y"]].copy()
    result.insert(0, "origin_type", origin_type)
    result.insert(0, "lsoa21cd", lsoa_codes[indices[valid]])
    return result.reset_index(drop=True)


def load_gp_facilities(gp_csv: Path, codepoints: pd.DataFrame) -> gpd.GeoDataFrame:
    organisations = pd.read_csv(gp_csv, header=None, dtype=str)
    organisations = organisations[organisations.iloc[:, 12].eq("ACTIVE")].copy()
    organisations["postcode"] = (
        organisations.iloc[:, 9]
        .fillna("")
        .str.upper()
        .str.replace(" ", "", regex=False)
    )
    organisations = organisations[organisations["postcode"].ne("")]
    organisations = organisations.drop_duplicates("postcode")
    located = organisations[["postcode"]].merge(
        codepoints, on="postcode", how="inner", validate="one_to_one"
    )
    return gpd.GeoDataFrame(
        located[["postcode"]].copy(),
        geometry=gpd.points_from_xy(located["x"], located["y"]),
        crs=27700,
    )


def load_road_graph(network_path: Path) -> tuple[csr_matrix, np.ndarray]:
    network = np.load(network_path)
    road_xy = np.column_stack((network["road_x"], network["road_y"])).astype(float)
    edge_u = network["edge_u"].astype(np.int64)
    edge_v = network["edge_v"].astype(np.int64)
    length = network["edge_length_m"].astype(float)
    graph = csr_matrix(
        (
            np.concatenate((length, length)),
            (np.concatenate((edge_u, edge_v)), np.concatenate((edge_v, edge_u))),
        ),
        shape=(len(road_xy), len(road_xy)),
    )
    graph.sum_duplicates()
    return graph, road_xy


def distance_to_nearest_facility(
    graph: csr_matrix,
    road_xy: np.ndarray,
    facilities: gpd.GeoDataFrame,
    maximum_snap_metres: float = 500.0,
) -> tuple[np.ndarray, dict[str, float]]:
    tree = cKDTree(road_xy)
    facility_xy = np.column_stack((facilities.geometry.x, facilities.geometry.y))
    snap, road = tree.query(facility_xy, k=1)
    valid = snap <= maximum_snap_metres
    snap = snap[valid]
    road = road[valid].astype(np.int64)
    initial = np.full(len(road_xy), np.inf, dtype=float)
    np.minimum.at(initial, road, snap)
    source_nodes = np.flatnonzero(np.isfinite(initial))
    n = graph.shape[0]
    coo = graph.tocoo(copy=False)
    augmented = csr_matrix(
        (
            np.concatenate((coo.data, initial[source_nodes])),
            (
                np.concatenate((coo.row, np.full(len(source_nodes), n, dtype=np.int64))),
                np.concatenate((coo.col, source_nodes)),
            ),
        ),
        shape=(n + 1, n + 1),
    )
    distances = dijkstra(
        augmented, directed=True, indices=n, return_predecessors=False
    )[:n]
    audit = {
        "facilities_input": int(len(facilities)),
        "facilities_snapped": int(valid.sum()),
        "facility_snap_p50_m": float(np.median(snap)),
        "facility_snap_p95_m": float(np.percentile(snap, 95)),
    }
    return distances, audit


def attach_times(
    origins: pd.DataFrame,
    road_xy: np.ndarray,
    road_tree: cKDTree,
    network_distances: dict[str, np.ndarray],
) -> pd.DataFrame:
    xy = origins[["x", "y"]].to_numpy(dtype=float)
    snap, road = road_tree.query(xy, k=1)
    output = origins.copy()
    output["origin_snap_m"] = snap
    for relation, distance in network_distances.items():
        output[f"{relation}_minutes"] = (distance[road] + snap) / WALKING_METRES_PER_MINUTE
    return output


def aggregate_origins(
    origins: pd.DataFrame,
    lsoa_codes: np.ndarray,
    relation: str,
) -> pd.DataFrame:
    value = f"{relation}_minutes"
    keys = ["lsoa21cd", "origin_type"]
    grouped = origins.groupby(keys, observed=True)[value]
    counts = grouped.size().rename("n_origins").reset_index()
    finite_origins = origins.assign(
        finite_time=origins[value].where(np.isfinite(origins[value]))
    )
    finite_grouped = finite_origins.groupby(keys, observed=True)["finite_time"]
    result = finite_grouped.agg(
        median_minutes="median",
        mean_minutes="mean",
        p10_minutes=lambda x: x.quantile(0.10),
        p90_minutes=lambda x: x.quantile(0.90),
    ).reset_index()
    result = counts.merge(result, on=keys, how="left")
    connected = (
        origins.assign(connected=np.isfinite(origins[value]))
        .groupby(keys, observed=True)["connected"]
        .mean()
        .rename("share_connected")
        .reset_index()
    )
    result = result.merge(connected, on=keys, how="left")
    capped = origins.assign(capped40=np.minimum(origins[value], 40.0))
    capped_grouped = capped.groupby(keys, observed=True)["capped40"]
    capped_stats = capped_grouped.agg(
        median_capped40_minutes="median",
        p90_capped40_minutes=lambda x: x.quantile(0.90),
    ).reset_index()
    within_40 = (
        origins.assign(within_40=origins[value].le(40.0))
        .groupby(keys, observed=True)["within_40"]
        .mean()
        .rename("share_within_40")
        .reset_index()
    )
    result = result.merge(capped_stats, on=keys, how="left")
    result = result.merge(within_40, on=keys, how="left")
    result["median_right_censored_40"] = result[
        "median_capped40_minutes"
    ].ge(40.0 - 1e-9)
    result["p90_right_censored_40"] = result["p90_capped40_minutes"].ge(
        40.0 - 1e-9
    )
    for threshold in THRESHOLDS:
        share = (
            origins.assign(within=origins[value].le(threshold))
            .groupby(["lsoa21cd", "origin_type"], observed=True)["within"]
            .mean()
            .rename(f"share_within_{int(threshold)}")
            .reset_index()
        )
        result = result.merge(share, on=["lsoa21cd", "origin_type"], how="left")
    complete = pd.MultiIndex.from_product(
        [lsoa_codes, sorted(origins["origin_type"].unique())],
        names=["lsoa21cd", "origin_type"],
    ).to_frame(index=False)
    result = complete.merge(result, on=["lsoa21cd", "origin_type"], how="left")
    result.insert(2, "relation", relation)
    return result


def validate_representative_times(
    case_dir: Path,
    relation: str,
    calculated: pd.DataFrame,
) -> dict[str, float]:
    graph = np.load(case_dir / "facility_lsoa_graph.npz", allow_pickle=True)
    relation_names = graph["relation_names"].astype(str)
    relation_index = int(np.flatnonzero(relation_names == relation)[0])
    mask = graph["edge_type"] == relation_index
    destination = graph["edge_index"][1, mask]
    minutes = graph["edge_impedance"][mask].astype(float)
    n_lsoa = len(graph["lsoa_codes"])
    stored = np.full(n_lsoa, np.inf, dtype=float)
    np.minimum.at(stored, destination, minutes)
    current = calculated.set_index("lsoa21cd").loc[
        graph["lsoa_codes"].astype(str), "median_minutes"
    ].to_numpy(dtype=float)
    finite = np.isfinite(current) & np.isfinite(stored)
    difference = current[finite] - stored[finite]
    return {
        "validation_n": int(finite.sum()),
        "validation_median_abs_difference_minutes": float(np.median(np.abs(difference))),
        "validation_p95_abs_difference_minutes": float(np.percentile(np.abs(difference), 95)),
        "validation_max_abs_difference_minutes": float(np.max(np.abs(difference))),
    }


def summarise_comparison(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (study, relation, origin_type), group in long.groupby(
        ["study", "relation", "origin_type"], observed=True
    ):
        if origin_type == "representative_point":
            continue
        rep = long[
            long["study"].eq(study)
            & long["relation"].eq(relation)
            & long["origin_type"].eq("representative_point")
        ][
            [
                "lsoa21cd",
                "median_minutes",
                "share_connected",
                *[f"share_within_{int(value)}" for value in THRESHOLDS],
            ]
        ].rename(
            columns={
                "median_minutes": "rep_minutes",
                "share_connected": "rep_share_connected",
                **{
                    f"share_within_{int(value)}": f"rep_share_within_{int(value)}"
                    for value in THRESHOLDS
                },
            }
        )
        merged = group.merge(rep, on="lsoa21cd", how="inner")
        valid = merged["n_origins"].notna()
        base = merged.loc[valid].copy()
        if base.empty:
            continue
        finite_time = np.isfinite(base["median_minutes"]) & np.isfinite(base["rep_minutes"])
        time_base = base.loc[finite_time]
        finite_spread = np.isfinite(time_base["p90_minutes"]) & np.isfinite(
            time_base["p10_minutes"]
        )
        common = {
            "study": study,
            "relation": relation,
            "origin_type": origin_type,
            "lsoas_with_origins": int(len(base)),
            "median_origins_per_lsoa": float(base["n_origins"].median()),
            "median_time_difference_minutes": float(
                np.median(time_base["median_minutes"] - time_base["rep_minutes"])
            ),
            "median_absolute_time_difference_minutes": float(
                np.median(np.abs(time_base["median_minutes"] - time_base["rep_minutes"]))
            ),
            "p90_absolute_time_difference_minutes": float(
                np.percentile(
                    np.abs(time_base["median_minutes"] - time_base["rep_minutes"]), 90
                )
            ),
            "median_within_lsoa_p90_p10_minutes": float(
                np.median(
                    time_base.loc[finite_spread, "p90_minutes"]
                    - time_base.loc[finite_spread, "p10_minutes"]
                )
            ),
            "origin_points_unreachable_percent": float(
                100.0 * (1.0 - base["share_connected"].mean())
            ),
            "lsoas_with_any_unreachable_origin_percent": float(
                100.0 * base["share_connected"].lt(1.0).mean()
            ),
        }
        for threshold in THRESHOLDS:
            share = base[f"share_within_{int(threshold)}"]
            rep_class = base[f"rep_share_within_{int(threshold)}"].ge(0.5)
            majority_class = share.ge(0.5)
            flip = rep_class.ne(majority_class)
            mixed = share.gt(0.25) & share.lt(0.75)
            row = dict(common)
            row.update(
                {
                    "threshold_minutes": threshold,
                    "classification_flip_count": int(flip.sum()),
                    "classification_flip_percent": float(100.0 * flip.mean()),
                    "flip_into_access_percent": float(
                        100.0 * ((~rep_class) & majority_class).mean()
                    ),
                    "flip_out_of_access_percent": float(
                        100.0 * (rep_class & (~majority_class)).mean()
                    ),
                    "mixed_access_count": int(mixed.sum()),
                    "mixed_access_percent": float(100.0 * mixed.mean()),
                    "representative_within_percent": float(100.0 * rep_class.mean()),
                    "majority_within_percent": float(100.0 * majority_class.mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_connected_robustness(
    long: pd.DataFrame, minimum_origin_connected_share: float = 0.95
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    alternatives = long[long["origin_type"].ne("representative_point")]
    for (study, relation, origin_type), group in alternatives.groupby(
        ["study", "relation", "origin_type"], observed=True
    ):
        rep = long[
            long["study"].eq(study)
            & long["relation"].eq(relation)
            & long["origin_type"].eq("representative_point")
        ][
            [
                "lsoa21cd",
                "share_connected",
                *[f"share_within_{int(value)}" for value in THRESHOLDS],
            ]
        ].rename(
            columns={
                "share_connected": "rep_share_connected",
                **{
                    f"share_within_{int(value)}": f"rep_share_within_{int(value)}"
                    for value in THRESHOLDS
                },
            }
        )
        merged = group.merge(rep, on="lsoa21cd", how="inner")
        merged = merged[
            merged["n_origins"].notna()
            & merged["share_connected"].ge(minimum_origin_connected_share)
            & merged["rep_share_connected"].eq(1.0)
        ]
        for threshold in THRESHOLDS:
            rep_class = merged[f"rep_share_within_{int(threshold)}"].ge(0.5)
            majority_class = merged[f"share_within_{int(threshold)}"].ge(0.5)
            rows.append(
                {
                    "study": study,
                    "relation": relation,
                    "origin_type": origin_type,
                    "minimum_origin_connected_share": minimum_origin_connected_share,
                    "threshold_minutes": threshold,
                    "lsoas": int(len(merged)),
                    "classification_flip_count": int(rep_class.ne(majority_class).sum()),
                    "classification_flip_percent": float(
                        100.0 * rep_class.ne(majority_class).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    label_map = {
        "postcode_centres": "Postcode centres",
        "all_building_centres_sample": "All OSM buildings",
        "explicit_residential_centres_sample": "Explicit residential OSM buildings",
    }
    city_map = {"greater_london": "Greater London", "greater_manchester": "Greater Manchester"}
    relation_map = {"nhs_gp": "GP", "osm_pharmacy": "Pharmacy"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    colors = {
        "postcode_centres": "#1f77b4",
        "all_building_centres_sample": "#d95f02",
        "explicit_residential_centres_sample": "#2ca02c",
    }
    markers = {
        "postcode_centres": "o",
        "all_building_centres_sample": "s",
        "explicit_residential_centres_sample": "^",
    }
    for row_index, study in enumerate(("greater_london", "greater_manchester")):
        for column_index, relation in enumerate(("nhs_gp", "osm_pharmacy")):
            ax = axes[row_index, column_index]
            selected = summary[
                summary["study"].eq(study) & summary["relation"].eq(relation)
            ]
            for origin_type, group in selected.groupby("origin_type", observed=True):
                ax.plot(
                    group["threshold_minutes"],
                    group["classification_flip_percent"],
                    color=colors[origin_type],
                    marker=markers[origin_type],
                    linewidth=1.8,
                    label=label_map[origin_type],
                )
            ax.set_title(f"{city_map[study]} — {relation_map[relation]}")
            ax.grid(alpha=0.25)
            ax.set_xticks(THRESHOLDS)
            if row_index == 1:
                ax.set_xlabel("Candidate walking-time boundary (minutes)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Sensitivity of threshold classification to the choice of LSOA origin", y=0.98)
    fig.supylabel("LSOAs whose majority classification flips (%)", x=0.01)
    fig.tight_layout(rect=(0.025, 0.08, 1, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_study(
    study: str,
    project_root: Path,
    pbf: Path,
    sample_size: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    case_dir = project_root / "data" / "processed" / "case_studies" / study
    lsoas = gpd.read_parquet(case_dir / "lsoas.parquet").to_crs(27700)
    lsoa_codes = lsoas["lsoa21cd"].astype(str).to_numpy()
    locator = PolygonGridLocator(lsoas.geometry)
    bounds = tuple(float(value) for value in lsoas.total_bounds)
    codepoint_dir = project_root / "data" / "raw" / "code_point_open_2026_08" / "extracted" / "Data" / "CSV"
    print(f"{study}: loading Code-Point Open", flush=True)
    codepoints = load_codepoints(codepoint_dir, bounds)
    postcode_origins = assign_lsoa(codepoints, locator, lsoa_codes, "postcode_centres")

    building_cache = cache_dir / f"{study}_building_origins.parquet"
    building_meta_cache = cache_dir / f"{study}_building_origin_metadata.json"
    if building_cache.exists() and building_meta_cache.exists():
        print(f"{study}: loading cached building samples", flush=True)
        building_origins = pd.read_parquet(building_cache)
        building_metadata = json.loads(building_meta_cache.read_text(encoding="utf-8"))
    else:
        print(f"{study}: extracting deterministic OSM building samples", flush=True)
        handler = BuildingOriginHandler(locator, len(lsoas), sample_size)
        handler.apply_file(str(pbf), locations=True, idx="flex_mem")
        all_buildings = handler.frame_from_heaps(
            handler.all_heaps, lsoa_codes, "all_building_centres_sample"
        )
        residential = handler.frame_from_heaps(
            handler.residential_heaps,
            lsoa_codes,
            "explicit_residential_centres_sample",
        )
        building_origins = pd.concat((all_buildings, residential), ignore_index=True)
        building_metadata = {
            "sample_size_cap_per_lsoa": sample_size,
            "all_building_count": int(handler.all_count.sum()),
            "explicit_residential_count": int(handler.residential_count.sum()),
            "lsoas_with_any_building": int(np.sum(handler.all_count > 0)),
            "lsoas_with_explicit_residential_building": int(
                np.sum(handler.residential_count > 0)
            ),
            "top_building_tags": dict(
                sorted(handler.tag_counts.items(), key=lambda item: (-item[1], item[0]))[:30]
            ),
        }
        building_origins.to_parquet(building_cache, index=False)
        building_meta_cache.write_text(
            json.dumps(building_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    representative_points = lsoas.geometry.representative_point()
    representative = pd.DataFrame(
        {
            "lsoa21cd": lsoa_codes,
            "origin_type": "representative_point",
            "x": representative_points.x.to_numpy(dtype=float),
            "y": representative_points.y.to_numpy(dtype=float),
        }
    )
    origins = pd.concat(
        (representative, postcode_origins, building_origins), ignore_index=True
    )

    print(f"{study}: loading road graph", flush=True)
    road_graph, road_xy = load_road_graph(case_dir / "walk_network.npz")
    road_tree = cKDTree(road_xy)
    gp_csv = project_root / "data" / "raw" / "nhs_ods_2026_08_26" / "gp_practices.csv"
    gp = load_gp_facilities(gp_csv, codepoints)
    pharmacies = gpd.read_parquet(case_dir / "osm_facilities.parquet").to_crs(27700)
    pharmacies = pharmacies[pharmacies["category"].eq("pharmacy")].copy()
    facilities = {"nhs_gp": gp, "osm_pharmacy": pharmacies}
    network_distances: dict[str, np.ndarray] = {}
    facility_audit: dict[str, dict[str, float]] = {}
    for relation, frame in facilities.items():
        print(f"{study}: routing nearest {relation}", flush=True)
        distance, relation_audit = distance_to_nearest_facility(
            road_graph, road_xy, frame
        )
        network_distances[relation] = distance
        facility_audit[relation] = relation_audit

    print(f"{study}: attaching travel times to {len(origins):,} origins", flush=True)
    origins = attach_times(origins, road_xy, road_tree, network_distances)
    parts: list[pd.DataFrame] = []
    validation: dict[str, dict[str, float]] = {}
    for relation in facilities:
        aggregated = aggregate_origins(origins, lsoa_codes, relation)
        parts.append(aggregated)
        rep_aggregated = aggregated[
            aggregated["origin_type"].eq("representative_point")
        ]
        validation[relation] = validate_representative_times(
            case_dir, relation, rep_aggregated
        )
    long = pd.concat(parts, ignore_index=True)
    long.insert(0, "study", study)
    metadata: dict[str, object] = {
        "study": study,
        "lsoas": int(len(lsoas)),
        "postcode_centres_in_lsoas": int(len(postcode_origins)),
        "lsoas_with_postcode_centres": int(postcode_origins["lsoa21cd"].nunique()),
        "building_origins": building_metadata,
        "facility_routing": facility_audit,
        "representative_point_validation": validation,
        "origin_snap_p50_m": {
            key: float(value)
            for key, value in origins.groupby("origin_type")["origin_snap_m"].median().items()
        },
        "origin_snap_p95_m": {
            key: float(value)
            for key, value in origins.groupby("origin_type")["origin_snap_m"].quantile(0.95).items()
        },
    }
    return long, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=80)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pbf_by_study = {
        "greater_london": args.project_root
        / "data"
        / "raw"
        / "osm"
        / "greater_london_2026_08_20"
        / "greater-london-260820.osm.pbf",
        "greater_manchester": args.project_root
        / "data"
        / "raw"
        / "osm"
        / "greater_manchester_2026_08_26"
        / "greater-manchester-260826.osm.pbf",
    }
    all_long: list[pd.DataFrame] = []
    metadata: dict[str, object] = {
        "walking_speed_metres_per_minute": WALKING_METRES_PER_MINUTE,
        "thresholds_minutes": THRESHOLDS,
        "building_sample_size_cap_per_lsoa": args.sample_size,
        "studies": {},
    }
    for study, pbf in pbf_by_study.items():
        long, study_metadata = run_study(
            study, args.project_root, pbf, args.sample_size, cache_dir
        )
        all_long.append(long)
        metadata["studies"][study] = study_metadata
    long = pd.concat(all_long, ignore_index=True)
    summary = summarise_comparison(long)
    connected_robustness = summarise_connected_robustness(long)
    long.to_csv(args.output_dir / "origin_sensitivity_by_lsoa.csv", index=False)
    summary.to_csv(args.output_dir / "origin_sensitivity_summary.csv", index=False)
    connected_robustness.to_csv(
        args.output_dir / "origin_sensitivity_connected_robustness.csv", index=False
    )
    (args.output_dir / "origin_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_summary(summary, args.output_dir / "figure_origin_sensitivity.png")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
