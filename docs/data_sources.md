# Data sources and required inputs

The analysis covers 4,994 Lower Layer Super Output Areas (LSOAs) in Greater London and 1,702 in Greater Manchester, using 2021 LSOA boundaries. The target is the Access to Healthy Assets and Hazards (AHAH) v5.1 index.

| Input | Source | Frozen study version |
| --- | --- | --- |
| AHAH index and geographical data | [Geographic Data Service](https://data.geods.ac.uk/dataset/access-to-healthy-assets-hazards-ahah) | v5.1 |
| Walking networks, facilities and buildings | [Geofabrik England downloads](https://download.geofabrik.de/europe/united-kingdom/england/) | London 20 August 2026; Manchester 26 August 2026 |
| GP and dentist premises | [NHS Organisation Data Service](https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads) | Project snapshot 26 August 2026 |
| Postcode coordinates | [Ordnance Survey Code-Point Open](https://osdatahub.os.uk/downloads/open/CodePointOpen) | August 2026 snapshot |
| Postcode resident counts | [Nomis Census 2021 postcode data](https://www.nomisweb.co.uk/sources/census_2021_pc) | Table P001, 2021 residents |
| Population-weighted LSOA centres | [ONS ArcGIS service](https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LSOA_PopCentroids_EW_2021_V4/FeatureServer/0) | England and Wales 2021 v4 |
| Green space | [OS Open Greenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace) | Frozen project extract |
| Pollution background variables | [UK Air modelling data](https://uk-air.defra.gov.uk/data/pcm-data) | 2024 PCM data |
| Vegetation context | [Microsoft Planetary Computer Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) | Selection rules in `scripts/extract_sentinel_ndvi_context.py` |

OpenStreetMap-derived materials require attribution to © OpenStreetMap contributors; see [OpenStreetMap copyright and licence](https://www.openstreetmap.org/copyright). Government and Ordnance Survey inputs must be used with the licences and attribution supplied by their providers. The MIT licence for this repository's code does not replace data or model-weight licences.

`official_input_sources.json` records the source URLs and SHA-256 hashes for the ONS centroid and Nomis downloads used in September 2026. Other local input paths and expected snapshot dates are retained in `configs/case_studies.json` and the data-preparation scripts. A current download can differ from the frozen study snapshot.

## Local directory contract

The following files are generated or acquired locally and are excluded from Git:

```text
data/raw/ahah_v5_1/ahah_v5_1_geo.parquet
data/raw/osm/greater_london_2026_08_20/greater-london-260820.osm.pbf
data/raw/osm/greater_manchester_2026_08_26/greater-manchester-260826.osm.pbf
data/raw/code_point_open_2026_08/extracted/Data/CSV/
data/raw/nhs_ods_2026_08_26/
data/raw/revision_20260925/
data/processed/case_studies/greater_london/
data/processed/case_studies/greater_manchester/
results/oof_predictions/<city>.parquet
results/tabpfn_gravity/nested_blend_<city>.csv
results/heterogeneous_graph_ensemble_relation_access_seeds5_e5/<city>.parquet
results/revision_20260925/
```

Each processed city directory contains its LSOA table, context features, frozen fold assignments, walking network and `facility_lsoa_graph.npz`. The preparation scripts define the additional intermediate filenames. The original graph-array interface is documented in [data_contract.md](data_contract.md).

The release includes aggregate reference results, not the full raw or processed dataset. Rebuilding all empirical results requires acquiring the inputs and running the preprocessing and model stages. Postcode counts describe 2021 residents at 2026 postcode locations; unmatched locations are excluded from population-weighted summaries.

The additional vulnerability analysis uses Census 2021 LSOA tables TS007A, TS045, TS067, TS011 and TS037. Obtain these from the official Census provider and place each CSV at `data/raw/census2021_vulnerability_2026_08_28/<table>/census2021-<table>-lsoa.csv`. The integration test checks the full national tables; the remaining tests need no external data.
