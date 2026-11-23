# Graph data contract

Real experiments are loaded from one compressed NumPy file (`.npz`). This
keeps model training independent from slow geospatial extraction and makes the
exact graph hashable.

In the current facility-to-LSOA model, the historical field name `building_mask`
identifies labelled LSOA nodes. The additional facility nodes are unlabelled.
The same interface also supports the earlier building-level synthetic examples.

Required arrays:

| Key | Shape | Dtype | Meaning |
|---|---:|---|---|
| `x` | `[N,F]` | float32 | Padded, finite node features |
| `node_mass` | `[N]` | float32 | Positive, median-scaled mass |
| `edge_index` | `[2,E]` | int64 | Directed source and destination indices |
| `edge_impedance` | `[E]` | float32 | Non-negative relation-specific impedance |
| `edge_type` | `[E]` | int64 | Contiguous relation id in `[0,R)` |
| `y` | `[N]` | float32 | Target; use NaN for unlabelled amenity nodes |
| `building_mask` | `[N]` | bool | True for building nodes |
| `group` | `[N]` | int64 | LSOA/spatial-block id; `-1` for unlabelled context nodes |

Optional arrays:

- `node_id`: stable external identifier.
- `node_type`: integer node type.
- `label_group`: source label unit such as LSOA; metrics are also reported after
  aggregating predictions back to this unit.
- `sample_weight`: positive building weights; use inverse label-group frequency
  when an area-level label is repeated across buildings.
- `x_coord`, `y_coord`: projected coordinates for audit and mapping.
- `relation_names`: string array defining each edge relation and unit.
- `feature_names`: string array defining every feature.

Validation fails when indices are out of range, values are non-finite, masses
are non-positive, labelled buildings lack groups, a group is split across
partitions, or edge types are not contiguous.

## Provenance sidecar

Every `.npz` used for a paper table should have a sibling JSON file containing:

- raw source URL and licence;
- snapshot/version date;
- SHA-256 of every raw input and the final `.npz`;
- CRS and study boundary hash;
- feature definitions and units;
- counts by node/edge type;
- missingness and exclusions;
- software commit and configuration id.

## Split contract

The code creates partitions at `group` level. For strict evaluation,
building-building edges crossing partitions are removed. Amenity context nodes
may be shared because they have no target label, but their features must be
computed without test-label information.
