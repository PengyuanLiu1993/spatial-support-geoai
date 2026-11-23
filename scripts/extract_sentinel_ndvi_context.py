from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from gravity_gnn.case_studies import sha256


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
VALID_SCL = np.asarray([2, 4, 5, 6, 7], dtype=np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build target-independent growing-season Sentinel-2 NDVI context "
            "for one LSOA case study"
        )
    )
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2025-09-30")
    parser.add_argument("--resolution", type=float, default=50.0)
    parser.add_argument("--max-cloud", type=float, default=80.0)
    parser.add_argument("--scenes-per-tile-month", type=int, default=1)
    parser.add_argument("--stac-url", default=STAC_URL)
    return parser


def aligned_grid(bounds: np.ndarray, resolution: float):
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    left = math.floor(min_x / resolution) * resolution
    bottom = math.floor(min_y / resolution) * resolution
    right = math.ceil(max_x / resolution) * resolution
    top = math.ceil(max_y / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    return from_origin(left, top, resolution, resolution), width, height


def select_items(items, per_tile_month: int):
    grouped = defaultdict(list)
    for item in items:
        tile = item.properties.get("s2:mgrs_tile")
        timestamp = item.datetime
        if tile is None or timestamp is None:
            continue
        grouped[(tile, timestamp.year, timestamp.month)].append(item)

    selected = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda item: (
                float(item.properties.get("eo:cloud_cover", 100.0)),
                item.datetime,
                item.id,
            ),
        )
        selected.extend(ranked[:per_tile_month])
    return selected


def read_scene_ndvi(item, *, crs, transform, width: int, height: int) -> np.ndarray:
    signed = planetary_computer.sign(item)
    arrays = {}
    for name, resampling in {
        "B04": Resampling.bilinear,
        "B08": Resampling.bilinear,
        "SCL": Resampling.nearest,
    }.items():
        with rasterio.open(signed.assets[name].href) as source:
            with WarpedVRT(
                source,
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
            ) as vrt:
                arrays[name] = vrt.read(1, masked=True)

    red = np.asarray(arrays["B04"].filled(0), dtype=np.float32)
    nir = np.asarray(arrays["B08"].filled(0), dtype=np.float32)
    scl = np.asarray(arrays["SCL"].filled(0), dtype=np.uint8)
    denominator = nir + red
    valid = (
        ~np.ma.getmaskarray(arrays["B04"])
        & ~np.ma.getmaskarray(arrays["B08"])
        & ~np.ma.getmaskarray(arrays["SCL"])
        & np.isin(scl, VALID_SCL)
        & (denominator > 0)
    )
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan
    return ndvi


def zonal_features(
    labels: np.ndarray,
    composite: np.ndarray,
    temporal_std: np.ndarray,
    observations: np.ndarray,
    zone_count: int,
) -> dict[str, np.ndarray]:
    flat_labels = labels.ravel()
    in_zone = flat_labels > 0
    zone_pixels = np.bincount(flat_labels[in_zone], minlength=zone_count + 1)[1:]

    flat_values = composite.ravel()
    valid = in_zone & np.isfinite(flat_values)
    valid_labels = flat_labels[valid]
    values = flat_values[valid].astype(float)
    counts = np.bincount(valid_labels, minlength=zone_count + 1)[1:]
    sums = np.bincount(valid_labels, weights=values, minlength=zone_count + 1)[1:]
    squares = np.bincount(
        valid_labels, weights=values**2, minlength=zone_count + 1
    )[1:]
    safe_counts = np.maximum(counts, 1)
    means = sums / safe_counts
    variances = np.maximum(squares / safe_counts - means**2, 0.0)

    features = {
        "s2_ndvi_mean": means,
        "s2_ndvi_spatial_std": np.sqrt(variances),
        "s2_ndvi_valid_fraction": counts / np.maximum(zone_pixels, 1),
    }
    for threshold in (0.2, 0.4, 0.6):
        hits = np.bincount(
            valid_labels,
            weights=(values >= threshold).astype(float),
            minlength=zone_count + 1,
        )[1:]
        features[f"s2_ndvi_fraction_ge_{str(threshold).replace('.', '_')}"] = (
            hits / safe_counts
        )

    for name, array in {
        "s2_ndvi_temporal_std_mean": temporal_std,
        "s2_ndvi_observations_mean": observations,
    }.items():
        flat = array.ravel()
        keep = in_zone & np.isfinite(flat)
        feature_counts = np.bincount(
            flat_labels[keep], minlength=zone_count + 1
        )[1:]
        feature_sums = np.bincount(
            flat_labels[keep], weights=flat[keep], minlength=zone_count + 1
        )[1:]
        features[name] = feature_sums / np.maximum(feature_counts, 1)

    for values_array in features.values():
        values_array[counts == 0] = np.nan
    return features


def main() -> None:
    args = build_parser().parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.scenes_per_tile_month <= 0:
        raise ValueError("scenes-per-tile-month must be positive")

    case_dir = args.case_root / args.study
    source_path = case_dir / "lsoas.parquet"
    lsoas = gpd.read_parquet(source_path).to_crs(27700).reset_index(drop=True)
    query_bounds = lsoas.to_crs(4326).total_bounds.tolist()
    catalog = pystac_client.Client.open(args.stac_url)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=query_bounds,
        datetime=f"{args.start}/{args.end}",
        query={"eo:cloud_cover": {"lt": args.max_cloud}},
    )
    all_items = list(search.items())
    items = select_items(all_items, args.scenes_per_tile_month)
    if not items:
        raise RuntimeError("Sentinel-2 query returned no usable scenes")

    transform, width, height = aligned_grid(lsoas.total_bounds, args.resolution)
    labels = rasterize(
        ((geometry, index + 1) for index, geometry in enumerate(lsoas.geometry)),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    stack = []
    scene_records = []
    environment = {
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": 64_000_000,
    }
    with rasterio.Env(**environment):
        for position, item in enumerate(items, start=1):
            print(
                f"[{position}/{len(items)}] {item.id} "
                f"cloud={float(item.properties.get('eo:cloud_cover', np.nan)):.1f}%",
                flush=True,
            )
            ndvi = read_scene_ndvi(
                item,
                crs=lsoas.crs,
                transform=transform,
                width=width,
                height=height,
            )
            stack.append(ndvi)
            scene_records.append(
                {
                    "id": item.id,
                    "datetime": item.datetime.isoformat(),
                    "tile": item.properties.get("s2:mgrs_tile"),
                    "cloud_cover": float(
                        item.properties.get("eo:cloud_cover", np.nan)
                    ),
                    "valid_grid_fraction": float(np.isfinite(ndvi).mean()),
                }
            )

    cube = np.stack(stack, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        composite = np.nanmedian(cube, axis=0).astype(np.float32)
        temporal_std = np.nanstd(cube, axis=0).astype(np.float32)
    observations = np.isfinite(cube).sum(axis=0).astype(np.float32)
    features = zonal_features(
        labels, composite, temporal_std, observations, len(lsoas)
    )

    output = lsoas[["lsoa21cd", "geometry"]].copy()
    for name, values in features.items():
        output[name] = values
    output_path = case_dir / "sentinel_ndvi_context.parquet"
    output.to_parquet(output_path, index=False)
    feature_names = list(features)
    summary = {
        "study": args.study,
        "source": {
            "collection": "sentinel-2-l2a",
            "stac_url": args.stac_url,
            "date_range": [args.start, args.end],
            "max_scene_cloud_cover": args.max_cloud,
            "selection": "lowest scene cloud cover per MGRS tile and calendar month",
            "scenes_per_tile_month": args.scenes_per_tile_month,
            "scl_valid_classes": VALID_SCL.tolist(),
        },
        "method": {
            "grid_crs": "EPSG:27700",
            "grid_resolution_m": args.resolution,
            "temporal_composite": "per-pixel median NDVI",
            "zonal_aggregation": "LSOA grid-cell summaries",
            "target_derived_features": False,
        },
        "query_items": len(all_items),
        "selected_items": len(items),
        "scenes": scene_records,
        "features": feature_names,
        "lsoas": len(lsoas),
        "missing_lsoas": int(output[feature_names].isna().all(axis=1).sum()),
        "output": str(output_path.resolve()),
        "sha256": {
            "lsoas": sha256(source_path),
            "output": sha256(output_path),
        },
    }
    summary_path = case_dir / "sentinel_ndvi_context_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
