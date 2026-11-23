# Spatial support validation in GeoAI

Source code for **When One Neighbourhood Node Represents Many Residents: Validating Spatial Support in GeoAI Models of Accessibility**, prepared for *Transactions in Urban Data, Science, and Technology*.

The study compares single-point representations of neighbourhoods with postcode and building origins in Greater London and Greater Manchester. It examines walking access to general practitioner practices and pharmacies, tests recovery of an injected accessibility signal, and evaluates the Heterogeneous Facility Graph Ensemble (HFGE).

This release corresponds to the manuscript revised on 10 November 2026. The final minor revision changed reference formatting and equation numbering; the numerical analyses are those completed in September 2026.

## Start here

Use Python 3.12. Create and activate a virtual environment, then run these commands from the repository root:

```bash
python -m pip install -e ".[dev,geo,analysis]"
python -m pytest
python scripts/check_reference_results.py
python examples/aggregation_recovery.py
python -m gravity_gnn.cli smoke --epochs 5 --device cpu --output results/smoke.json
```

These checks run without downloading the study data. The synthetic examples test the implementation; `check_reference_results.py` checks the supplied aggregate results against the manuscript's reported values. Full empirical reproduction requires the inputs described in [Data sources](docs/data_sources.md) and [Reproduction](docs/reproduction.md).

For TabPFN, additionally install `.[tabpfn]` and obtain access to the upstream model weights. The paper uses TabPFN v2 with five estimators. For vector figure preparation, install `.[figures]`.

## Repository contents

| Location | Purpose |
| --- | --- |
| `src/gravity_gnn/` | Graph data structures, routing, models, training and spatial validation |
| `scripts/` | Data preparation, model fitting, graph ablations, diagnostics and revision analyses |
| `analysis/` | Original origin-sensitivity, threshold and signal-recovery analysis functions |
| `reference_results/` | Frozen aggregate tables used to check the manuscript results |
| `assets/framework/` | Vector sources for the five-panel HFGE framework figure |
| `tests/` | Tests of preprocessing, routing, partitioning, models and analysis helpers |
| `examples/aggregation_recovery.py` | Small synthetic illustration of signal aggregation and coefficient recovery |
| `docs/` | Data requirements, reproduction steps, source provenance and release notes |

The main paper entry points are:

1. `scripts/revision_origin_audit.py` — postcode, population-weighted centre and random-origin comparisons.
2. `scripts/revision_inference.py` — flexible trends, independently calibrated signal injection and coverage-share recovery.
3. `scripts/run_heterogeneous_facility_gnn.py` and `scripts/assemble_heterogeneous_graph_ensemble.py` — HFGE graph hybrids and the fixed equal-weight ensemble.
4. `scripts/summarise_hgge_robustness.py`, `scripts/summarise_paired_graph_relation_ablation.py` and `scripts/summarise_hgge_exposure_reliability.py` — ablations and reproducibility of fitted relations.

Historical `hgge` names remain in filenames to preserve the experimental provenance. The model name in the current paper is **HFGE**. Some additional scripts support baseline and exploratory analyses; use the documented commands and frozen configurations for the reported results.

## Reproducibility

The study uses five contiguous spatial folds in each city and five neural seeds (20260827–20260831). Preprocessing is fitted within the training folds. Absolute coordinates and target-derived variables are excluded from the model predictors. The two cities are comparative cases; cross-city pooling is part of the training design.

The reference tables contain aggregate results. Raw geographical data, postcode-level routes, fitted weights and complete prediction caches must be acquired or generated separately. Their expected locations are listed in the reproduction guide. The tested analysis environment is recorded in `requirements/analysis-tested.txt`; TabPFN was run in a separate environment.

## Citation and licence

When using this work, cite the paper title above and the repository version or commit used. Publication metadata can be added when the final article DOI is available.

Original code is provided under the [MIT licence](LICENSE). Third-party data and model weights retain their own licences. See [Data sources](docs/data_sources.md) for attribution and access links.
