# Comparison notebook suite

This directory contains notebook-driven, paper-facing evaluation for the trained disc-lifing checkpoints. The notebooks are designed to **reconstruct models from saved checkpoints, run validation-split inference on local HDF5 assets, and save tables/figures under `Comparison/results/`**.

## 1. Purpose of paper-results evaluation

The goal of the paper-results notebooks is not to retrain models; it is to produce a transparent, checkpoint-auditable comparison of node-level stress and fatigue-life prediction quality. The emphasis is on **local fatigue-critical behaviour** (minimum-life regions, lower-transition performance, and representative geometry maps), not only pooled global fit.

## 2. Notebook suite

The suite is organized around four paper analyses:

1. **`01_fp_vs_argent.ipynb`** – compares LC-PointNet, GC-PointNet, and ArGEnT-A on Uniform/Edge and Zonal/Edge.
2. **`02_engineered_geometric_features.ipynb`** – evaluates the `Edge_arc_feat` ablation for ArGEnT-A + GF, GC-PointNet + GF, and LC-PointNet + GF. The `Edge_arc_feat` ArGEnT_self_att_noSDF checkpoint consumes the same engineered geometric descriptors through its adapted cross-attention point-token input (`Training_script.py` `INPUT_COLS`), so it is reported as `ArGEnT-A + GF`, not plain `ArGEnT-A`; unlike PointNet checkpoints, its feature configuration is defined by the training-script `INPUT_COLS`.
3. **`03_data_efficiency.ipynb`** – evaluates the `Edge_10`, `Edge_25`, `Edge_50`, and `Edge_75` subset comparison.
4. **`04_joint_stress_supervision.ipynb`** – evaluates the `Edge_no_stress` / joint stress-supervision study.

## 3. Example HDF5 files

Small illustrative/example files referenced by this suite should live at:

- `Comparison/Examples/disc_example_edge_deriv_uniform.h5`
- `Comparison/Examples/disc_example_edge_deriv_zonal.h5`

These examples are useful for schema inspection and demonstration, but they are **not a substitute** for the full local datasets needed to regenerate paper-scale tables.

## 4. Instructions for running notebooks

1. Open Jupyter from the repository root, or from `Comparison/` with the repository as the working tree parent.
2. Activate a kernel that has `torch`, `h5py`, `numpy`, `pandas`, and `matplotlib` installed.
3. Ensure the required HDF5 assets exist in `Data_gen/output/`.
4. Run the notebook from a **clean kernel** from top to bottom.
5. Inspect `Comparison/results/<notebook_name>/` for saved CSV, JSON, PNG, and PDF artifacts.

Notebook 01 auto-discovers checkpoints; it does not require manual editing of checkpoint paths when run in the tracked repository layout.

## 5. Required local assets

Paper-result notebooks rely on local assets that are usually not committed to Git:

- `Data_gen/output/disc_dataset_edge_deriv_uniform.h5`
- `Data_gen/output/disc_dataset_edge_deriv_zonal.h5`
- The trained checkpoint `.pt` files already present under the regime/model folders
- A valid Git checkout so the notebook can record `git rev-parse HEAD`

Optional/example-only assets:

- `Comparison/Examples/disc_example_edge_deriv_uniform.h5`
- `Comparison/Examples/disc_example_edge_deriv_zonal.h5`

## 6. Result provenance labels

All outputs from the current suite should be labelled **validation-split evaluation** unless you can independently prove that the evaluated geometries were not involved in training-time model selection, validation, or checkpoint choice. Notebook 01 writes this label into saved provenance and summary artifacts.

## 7. Principal metric definitions

The helper module `eval_helpers.py` computes the main paper diagnostics:

- **Pooled metrics**: node-level pooled stress and log-life MSE / RMSE / MAE / R².
- **Life-band metrics**: log-life errors over fixed bands (`<2`, `2-3`, `3-4`, `4-6`, `>=6`).
- **Grouped-region metrics**: MAE / RMSE over fixed physical groups (critical lower transition, rim features, remaining contour).
- **Geometry-level metrics**: per-geometry whole-field LogLife MAE, whole-field Stress MAE (where available), absolute minimum-life error, absolute maximum-stress error, and same-zone agreement.
- **Paired comparisons**: geometry-by-geometry differences between competing families so sign conventions stay explicit.

## 8. Why pooled R² and raw-life Max_PE must not be interpreted alone

A very high pooled **R²** can coexist with poor behaviour in the fatigue-critical tail because most nodes are easier than the minimum-life region. Likewise, raw-life **`Max_PE (%)`** can be dominated by a small number of very low true-life values, making it too volatile to rank models by itself.

For this reason, the notebooks treat pooled metrics as necessary but insufficient evidence. Full life-band metrics, grouped physical-region metrics, and geometry-level diagnostics must all be checked before drawing conclusions about local fatigue-critical usefulness.

## 9. Output artifact structure under `Comparison/results/`

Each notebook writes into its own subdirectory. The artifact budget is kept compact:
**no per-node CSV** (`node_results.csv`), **no per-geometry CSV** (`geometry_level_metrics.csv`).

```text
Comparison/results/
  01_fp_vs_argent/
    pooled_metrics.csv             (whole-field Stress/LogLife metrics per model/regime)
    life_band_metrics.csv          (full LogLife band metrics)
    grouped_region_metrics.csv     (three grouped physical regions)
    summary_table.csv              (compact: pooled + critical-region metrics in one table)
    paired_summary.csv             (geometry-paired min-life comparison: ArGEnT-FP and PointNet-FP)
    representative_geometry_ids.json
    run_metadata.json
    figures/
      uniform/
      zonal/
  02_engineered_geometric_features/
    summary_table.csv
    paired_summary.csv
    cross_notebook_edge_vs_edge_arc_feat.csv
    representative_geometry_ids.json
    run_metadata.json
    figures/
  03_data_efficiency/
    summary_by_fraction.csv        (pooled + critical-region metrics per training fraction)
    run_metadata.json
    figures/
  04_joint_stress_supervision/
    summary_table.csv
    paired_pooled_summary.csv
    paired_bins_summary.csv
    paired_zones_summary.csv
    run_metadata.json
    figures/
```

The `run_metadata.json` in each subdirectory records the git commit SHA, evaluation label, and model families for provenance.

## 10. Scope exclusions

These notebooks do **not**:

- train or fine-tune models,
- change checkpoints,
- certify physical validity beyond the stored datasets,
- claim that any single model is universally superior,
- replace an independent external evaluation split,
- or claim deployment readiness.

Results are labelled **validation-split evaluation**: the same deterministic 20% geometry holdout (seed 42) used across all notebooks. Do not call this a test set unless independent provenance of the split from checkpoint selection is established.

They are paper-support tools for transparent comparison of already trained research models.
