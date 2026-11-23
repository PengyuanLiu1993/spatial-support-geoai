# Reproduction guide

Run commands from the repository root after installing the package. Python 3.12 was used for the release checks. `requirements/analysis-tested.txt` records the installed scientific package versions. TabPFN was run separately with package version 8.2.0 and model version v2.

## Checks available with this repository

```bash
python -m pytest
python scripts/check_reference_results.py
python examples/aggregation_recovery.py
python -m gravity_gnn.cli smoke --epochs 5 --device cpu --output results/smoke.json
```

The reference checker verifies table consistency and the headline comparisons, including 91.93% point-aligned detection and 11.96%, 41.24% and 45.64% detection in the postcode point-fit, postcode share-fit and population-weighted share-fit experiments. It reads frozen results; it does not retrain the empirical model. The synthetic example recomputes an aggregation and coefficient-recovery example from generated observations.

One Census integration test is skipped when its external inputs are absent. To run it, supply the five Census tables described in the data guide, or set `HFGE_CENSUS_ROOT` to their existing directory. All other tests use generated fixtures.

## Build the empirical inputs

Acquire the versions listed in [data_sources.md](data_sources.md), retaining their licences and checksums. Prepare the study boundaries with:

```bash
python scripts/prepare_case_studies.py --config configs/case_studies.json
```

Use `scripts/extract_facility_network.py` for each city and its matching OSM PBF; `scripts/build_case_context.py` and the `extract_*_context.py` scripts construct the context inputs. `scripts/build_facility_lsoa_graph.py --study <city>` builds the facility graph. Each command's `--help` describes the input paths and parameters. The frozen routing limit is 40 minutes. `src/gravity_gnn/routing.py` contains the network-routing implementation.

These stages depend on the separately acquired geographical inputs. The repository does not provide a single-command download of every source or frozen model checkpoint.

## Model fitting and graph evaluation

The following command fits the graph-hybrid component over both cities, using the five neural seeds and exporting the relation contrasts:

```bash
python scripts/run_paper_graph.py --device cuda
```

The wrapper reads `configs/paper_graph.json`: hidden dimension 32, message dimension 8, one layer, dropout 0.15, learning rate 0.001 and weight decay 0.001, with a maximum of 240 epochs and patience 30. Use `--device cpu` for CPU execution or `--dry-run` to inspect the arguments without fitting. The general training script's defaults differ from this frozen paper configuration.

For ablations, call `scripts/run_heterogeneous_facility_gnn.py` with the same frozen arguments and separate output paths: `--kernel-mode uniform` removes the explicit gravity kernel, `--kernel-mode learned` supplies the hard-gravity variant, and `--exclude-relations nhs_gp,nhs_dentist,osm_hospital,osm_pharmacy` removes the four primary-care relations.

`scripts/prepare_tabpfn_gravity_inputs.py` prepares the context input bundle. Fit the context-only component with:

```bash
python scripts/run_tabpfn_gravity_screen.py --variant context_only --model-version v2 --n-estimators 5 --output results/tabpfn_gravity/screen_v2_e5.json
```

After producing the required graph and comparator predictions, use `scripts/assemble_heterogeneous_graph_ensemble.py`. Set `--graph-dir`, `--hard-gravity-dir`, `--no-gravity-dir`, `--tabpfn-dir`, `--nested-reference` and `--distance-oof-dir` to their actual output locations, and set `--tabpfn-estimators 5`. The HFGE prediction combines the mean graph hybrid and context-only TabPFN with equal weights.

The `summarise_hgge_robustness.py`, `summarise_paired_graph_relation_ablation.py` and `summarise_hgge_exposure_reliability.py` scripts calculate the corresponding graph diagnostics. Consult their `--help` for the prediction directories. Aggregate reference metrics are in `reference_results/hfge/`.

## Origin comparisons and signal recovery

With the processed city inputs and held-out prediction files available, run:

```bash
python scripts/revision_download_official_data.py
python scripts/revision_origin_audit.py
python scripts/revision_inference.py
python scripts/revision_verify.py
python scripts/revision_summarise.py
```

The official download step retrieves ONS population-weighted centres and Nomis P001 resident counts. The origin audit reconstructs postcode routes, attaches population counts, calculates centre and random-postcode comparisons, and checks the 16 original postcode classification counts. Its earlier comparison table is included at `reference_results/paper/origin_sensitivity/origin_sensitivity_summary.csv`.

The inference stage runs 2,000 bootstrap replicates, 2,000 calibration replicates and 1,000 evaluation replicates, with 25 spatial blocks. The four candidate boundaries are 5, 10, 15 and 20 minutes. The block seed is 20260903 for London and 20260904 for Manchester. The fixed settings are recorded in `reference_results/revision_20260925/inference_settings.json`.

Full verification checks reconstructed counts, coverage shares, population matching, the conditional-projection identity and coefficient recovery. The lightweight checker at the start of this guide is available when the external inputs have not been acquired.

## Figures

`scripts/revision_make_figures.py` generates the revised geographical example, origin comparison and signal-recovery plots from the empirical outputs. The framework source is preserved at `assets/framework/figure3.svg`, with its original vector inputs alongside it.

To reproduce the two-row framework layout, run `scripts/revision_relayout_original_framework.py` with the `figures` dependencies. `scripts/repair_framework_labels.py` generates the final label layer from `assets/framework/two_row.svg`; set the environment variables `HFGE_ARIAL` and `HFGE_ARIAL_BOLD` to your licensed Arial font files before running it. Font files are not redistributed. The supplied final SVG preserves the manuscript's framework design.

## Release provenance

`docs/source_manifest.json` records the source and release hashes of copied files. Packaging changes remove machine-specific paths, place the four paper-analysis modules under `analysis/`, and distinguish reference results from generated outputs. The numerical model and estimation methods are retained.

Text files use LF line endings so their release hashes remain stable across platforms. `docs/ci-example.yml` contains an optional GitHub Actions configuration; it is supplied as an inactive example.
