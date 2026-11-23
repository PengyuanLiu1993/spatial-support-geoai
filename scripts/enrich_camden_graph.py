from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from gravity_gnn.context import (
    area_weighted_grid_values,
    load_pcm_grid,
    radius_counts,
    radius_weighted_sums,
)
from gravity_gnn.data import load_graph_npz


POI_GROUPS = {
    "primary_care": {"clinic_doctor", "dentist", "pharmacy"},
    "hospital": {"hospital"},
    "physical_activity": {"greenspace", "green_blue_space", "sport_fitness"},
    "fast_food": {"fast_food"},
    "pub_bar": {"pub_bar"},
    "gambling": {"gambling"},
    "tobacco": {"tobacco"},
    "food_retail": {"convenience", "supermarket"},
    "education": {"school", "university"},
    "social_food": {"restaurant_cafe"},
}
ROAD_GROUPS = {
    "major": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"},
    "local": {"tertiary", "tertiary_link", "residential", "unclassified", "living_street", "service"},
    "active": {"footway", "cycleway", "path", "pedestrian", "steps", "bridleway"},
}
RADII = (500.0, 1000.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Add target-free spatial context and DEFRA PCM inputs")
    result.add_argument("--base-graph", type=Path, required=True)
    result.add_argument("--base-buildings", type=Path, required=True)
    result.add_argument("--lsoas", type=Path, required=True)
    result.add_argument("--raw-buildings", type=Path, required=True)
    result.add_argument("--raw-pois", type=Path, required=True)
    result.add_argument("--raw-roads", type=Path, required=True)
    result.add_argument("--defra-dir", type=Path, required=True)
    result.add_argument("--os-greenspace", type=Path)
    result.add_argument("--nhs-gp", type=Path)
    result.add_argument("--nhs-dentist", type=Path)
    result.add_argument("--codepoint-dir", type=Path)
    result.add_argument("--fhrs-xml", type=Path)
    result.add_argument("--gambling-csv", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = np.load(args.base_graph, allow_pickle=False)
    data = load_graph_npz(args.base_graph)
    buildings = gpd.read_parquet(args.base_buildings).to_crs(27700)
    lsoas = gpd.read_parquet(args.lsoas).to_crs(27700)
    raw_buildings = gpd.read_parquet(args.raw_buildings).to_crs(27700)
    raw_pois = gpd.read_parquet(args.raw_pois).to_crs(27700)
    roads = gpd.read_parquet(args.raw_roads).to_crs(27700)
    building_points = buildings.geometry.representative_point()
    building_xy = np.column_stack((building_points.x, building_points.y))

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    study_centroid = lsoas.geometry.union_all().centroid
    centre_x = float(study_centroid.x)
    centre_y = float(study_centroid.y)
    x_km = (building_xy[:, 0] - centre_x) / 1000.0
    y_km = (building_xy[:, 1] - centre_y) / 1000.0
    feature_blocks.append(np.column_stack((x_km, y_km, x_km * x_km, y_km * y_km, x_km * y_km)))
    feature_names.extend(("spatial_x_km", "spatial_y_km", "spatial_x2", "spatial_y2", "spatial_xy"))

    raw_building_points = raw_buildings.geometry.representative_point()
    raw_building_xy = np.column_stack((raw_building_points.x, raw_building_points.y))
    building_counts = radius_counts(building_xy, raw_building_xy, (100.0, 500.0, 1000.0))
    feature_blocks.append(np.log1p(np.maximum(building_counts - 1.0, 0.0)))
    feature_names.extend(("log_full_building_count_100m", "log_full_building_count_500m", "log_full_building_count_1000m"))

    poi_points = raw_pois.geometry
    for group_name, categories in POI_GROUPS.items():
        selected = raw_pois[raw_pois["category"].isin(categories)]
        selected_xy = np.column_stack((selected.geometry.x, selected.geometry.y)) if len(selected) else np.empty((0, 2))
        counts = radius_counts(building_xy, selected_xy, RADII)
        feature_blocks.append(np.log1p(counts))
        feature_names.extend(f"log_{group_name}_count_{int(radius)}m" for radius in RADII)

    road_midpoints = roads.geometry.interpolate(0.5, normalized=True)
    road_xy = np.column_stack((road_midpoints.x, road_midpoints.y))
    road_length = roads.geometry.length.to_numpy(dtype=float)
    for group_name, classes in ROAD_GROUPS.items():
        selected = roads["highway"].isin(classes).to_numpy()
        length_sums = radius_weighted_sums(building_xy, road_xy[selected], road_length[selected], RADII)
        area_km2 = np.asarray([np.pi * (radius / 1000.0) ** 2 for radius in RADII])
        feature_blocks.append(np.log1p(length_sums / area_km2[None, :]))
        feature_names.extend(f"log_{group_name}_road_m_per_km2_{int(radius)}m" for radius in RADII)

    nhs_sources = {"nhs_gp": args.nhs_gp, "nhs_dentist": args.nhs_dentist}
    supplied_nhs = {name: path for name, path in nhs_sources.items() if path is not None}
    if supplied_nhs:
        if args.codepoint_dir is None:
            raise ValueError("--codepoint-dir is required with NHS organisation data")
        organisation_frames: dict[str, pd.DataFrame] = {}
        needed_postcodes: set[str] = set()
        for source_name, path in supplied_nhs.items():
            organisations = pd.read_csv(path, header=None, dtype=str)
            organisations = organisations[organisations.iloc[:, 12].eq("ACTIVE")].copy()
            organisations["postcode"] = (
                organisations.iloc[:, 9].fillna("").str.upper().str.replace(" ", "", regex=False)
            )
            organisations = organisations[organisations["postcode"].ne("")]
            organisations = organisations.drop_duplicates("postcode")
            organisation_frames[source_name] = organisations
            needed_postcodes.update(organisations["postcode"])
        coordinate_parts: list[pd.DataFrame] = []
        for codepoint_file in sorted(args.codepoint_dir.glob("*.csv")):
            part = pd.read_csv(
                codepoint_file,
                header=None,
                usecols=[0, 2, 3],
                names=["postcode", "easting", "northing"],
                dtype={"postcode": str, "easting": float, "northing": float},
            )
            part["postcode"] = part["postcode"].str.upper().str.replace(" ", "", regex=False)
            selected = part[part["postcode"].isin(needed_postcodes)]
            if len(selected):
                coordinate_parts.append(selected)
        postcode_coordinates = pd.concat(coordinate_parts, ignore_index=True).drop_duplicates("postcode")
        min_x, min_y, max_x, max_y = lsoas.total_bounds
        for source_name, organisations in organisation_frames.items():
            located = organisations.merge(postcode_coordinates, on="postcode", how="inner")
            located = located[
                located["easting"].between(min_x - 5000.0, max_x + 5000.0)
                & located["northing"].between(min_y - 5000.0, max_y + 5000.0)
            ]
            source_xy = located[["easting", "northing"]].to_numpy(dtype=float)
            if not len(source_xy):
                raise ValueError(f"no geocoded {source_name} locations near the study area")
            from scipy.spatial import cKDTree

            nearest, _ = cKDTree(source_xy).query(building_xy, k=1)
            counts = radius_counts(building_xy, source_xy, (1000.0, 3000.0))
            feature_blocks.extend((np.log1p(nearest)[:, None], np.log1p(counts)))
            feature_names.extend(
                (
                    f"log_{source_name}_nearest_m",
                    f"log_{source_name}_count_1000m",
                    f"log_{source_name}_count_3000m",
                )
            )

    if args.fhrs_xml is not None:
        fhrs_groups = {
            "fsa_takeaway": {"Takeaway/sandwich shop"},
            "fsa_pub_bar": {"Pub/bar/nightclub"},
            "fsa_food_retail": {"Retailers - other", "Retailers - supermarkets/hypermarkets"},
        }
        records: list[tuple[str, float, float]] = []
        root = ET.parse(args.fhrs_xml).getroot()
        for establishment in root.findall(".//EstablishmentDetail"):
            business_type = establishment.findtext("BusinessType")
            longitude = establishment.findtext("Geocode/Longitude")
            latitude = establishment.findtext("Geocode/Latitude")
            if business_type and longitude and latitude:
                records.append((business_type, float(longitude), float(latitude)))
        fhrs = gpd.GeoDataFrame(
            {"business_type": [record[0] for record in records]},
            geometry=gpd.points_from_xy(
                [record[1] for record in records], [record[2] for record in records]
            ),
            crs=4326,
        ).to_crs(27700)
        for source_name, business_types in fhrs_groups.items():
            selected = fhrs[fhrs["business_type"].isin(business_types)]
            source_xy = np.column_stack((selected.geometry.x, selected.geometry.y))
            from scipy.spatial import cKDTree

            nearest, _ = cKDTree(source_xy).query(building_xy, k=1)
            counts = radius_counts(building_xy, source_xy, RADII)
            feature_blocks.extend((np.log1p(nearest)[:, None], np.log1p(counts)))
            feature_names.extend(
                (
                    f"log_{source_name}_nearest_m",
                    f"log_{source_name}_count_500m",
                    f"log_{source_name}_count_1000m",
                )
            )

    if args.gambling_csv is not None:
        if args.codepoint_dir is None:
            raise ValueError("--codepoint-dir is required with gambling premises")
        gambling = pd.read_csv(args.gambling_csv, dtype=str)
        gambling = gambling[
            gambling["Local Authority"].str.contains("Camden", case=False, na=False)
        ].copy()
        gambling["postcode"] = (
            gambling["Postcode"].fillna("").str.upper().str.replace(" ", "", regex=False)
        )
        needed = set(gambling.loc[gambling["postcode"].ne(""), "postcode"])
        parts: list[pd.DataFrame] = []
        for codepoint_file in sorted(args.codepoint_dir.glob("*.csv")):
            part = pd.read_csv(
                codepoint_file,
                header=None,
                usecols=[0, 2, 3],
                names=["postcode", "easting", "northing"],
                dtype={"postcode": str, "easting": float, "northing": float},
            )
            part["postcode"] = part["postcode"].str.upper().str.replace(" ", "", regex=False)
            selected = part[part["postcode"].isin(needed)]
            if len(selected):
                parts.append(selected)
        coordinates = pd.concat(parts, ignore_index=True).drop_duplicates("postcode")
        located = gambling.merge(coordinates, on="postcode", how="inner")
        source_xy = located[["easting", "northing"]].to_numpy(dtype=float)
        from scipy.spatial import cKDTree

        nearest, _ = cKDTree(source_xy).query(building_xy, k=1)
        counts = radius_counts(building_xy, source_xy, RADII)
        feature_blocks.extend((np.log1p(nearest)[:, None], np.log1p(counts)))
        feature_names.extend(
            (
                "log_gambling_register_nearest_m",
                "log_gambling_register_count_500m",
                "log_gambling_register_count_1000m",
            )
        )

    if args.os_greenspace is not None:
        min_x, min_y, max_x, max_y = lsoas.total_bounds
        green_bbox = (min_x - 1000.0, min_y - 1000.0, max_x + 1000.0, max_y + 1000.0)
        access_points = gpd.read_file(
            args.os_greenspace, layer="access_point", bbox=green_bbox
        ).to_crs(27700)
        green_sites = gpd.read_file(
            args.os_greenspace, layer="greenspace_site", bbox=green_bbox
        ).to_crs(27700)
        access_xy = np.column_stack(
            (access_points.geometry.x, access_points.geometry.y)
        )
        access_counts = radius_counts(building_xy, access_xy, RADII)
        if len(access_xy):
            from scipy.spatial import cKDTree

            nearest_access, _ = cKDTree(access_xy).query(building_xy, k=1)
        else:
            nearest_access = np.full(len(building_xy), 5000.0)
        feature_blocks.extend((np.log1p(nearest_access)[:, None], np.log1p(access_counts)))
        feature_names.extend(
            ("log_os_green_nearest_access_m", "log_os_green_access_count_500m", "log_os_green_access_count_1000m")
        )
        site_points = green_sites.geometry.representative_point()
        site_xy = np.column_stack((site_points.x, site_points.y))
        site_area = green_sites.geometry.area.to_numpy(dtype=float)
        nearby_area = radius_weighted_sums(
            building_xy, site_xy, site_area, RADII
        )
        feature_blocks.append(np.log1p(nearby_area))
        feature_names.extend(
            ("log_os_green_site_area_500m", "log_os_green_site_area_1000m")
        )

    pollutant_files = {
        "defra_no2_2024": args.defra_dir / "no2_2024.csv",
        "defra_pm10_2024": args.defra_dir / "pm10_2024.csv",
        "defra_so2_2024": args.defra_dir / "so2_2024.csv",
    }
    lsoa_values = lsoas[["lsoa21cd", "geometry"]].copy()
    for feature_name, path in pollutant_files.items():
        grid = load_pcm_grid(path, tuple(lsoas.total_bounds))
        value_column = str(grid.columns[-2])
        values = area_weighted_grid_values(lsoas, grid, value_column, "lsoa21cd")
        lsoa_values[feature_name] = lsoa_values["lsoa21cd"].map(values)
        building_values = buildings["lsoa21cd"].map(values).to_numpy(dtype=float)
        if not np.isfinite(building_values).all():
            raise ValueError(f"missing {feature_name} for a model building")
        feature_blocks.append(building_values[:, None])
        feature_names.append(feature_name)

    building_context = np.column_stack(feature_blocks).astype(np.float32)
    context = np.zeros((data.num_nodes, building_context.shape[1]), dtype=np.float32)
    context[: len(buildings)] = building_context
    enriched_x = np.concatenate((base["x"].astype(np.float32), context), axis=1)
    original_feature_names = base["feature_names"].astype(str).tolist()
    payload = {key: base[key] for key in base.files if key not in {"x", "feature_names"}}
    np.savez_compressed(
        args.output,
        **payload,
        x=enriched_x,
        feature_names=np.asarray(original_feature_names + feature_names),
    )
    load_graph_npz(args.output)
    lsoa_values.to_parquet(args.output.parent / "camden_lsoa_context.parquet", index=False)
    summary = {
        "base_graph": str(args.base_graph.resolve()),
        "base_features": len(original_feature_names),
        "added_features": feature_names,
        "total_features": enriched_x.shape[1],
        "nodes": data.num_nodes,
        "buildings": int(data.building_mask.sum()),
        "sha256": {
            "base_graph": sha256(args.base_graph),
            "defra_no2_2024": sha256(pollutant_files["defra_no2_2024"]),
            "defra_pm10_2024": sha256(pollutant_files["defra_pm10_2024"]),
            "defra_so2_2024": sha256(pollutant_files["defra_so2_2024"]),
            "output_graph": sha256(args.output),
        },
    }
    if args.os_greenspace is not None:
        summary["sha256"]["os_open_greenspace"] = sha256(args.os_greenspace)
    for source_name, path in supplied_nhs.items():
        summary["sha256"][source_name] = sha256(path)
    if args.fhrs_xml is not None:
        summary["sha256"]["fhrs_camden"] = sha256(args.fhrs_xml)
    if args.gambling_csv is not None:
        summary["sha256"]["gambling_premises"] = sha256(args.gambling_csv)
    (args.output.parent / "camden_graph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
