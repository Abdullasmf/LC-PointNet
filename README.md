# Disc Fatigue-Life Learning (Disc_lifing_paper_v2)

This repository contains the data-generation, training, and paper-comparison assets used to study graph/point-cloud surrogates for predicting stress and fatigue life on axisymmetric turbine-disc geometries. The focus is local, geometry-resolved prediction on edge-based disc contour representations rather than only global scalar outputs.

## 1. Project title and purpose

**Disc_lifing_paper_v2** is a research repository for building synthetic disc datasets, training neural surrogates, and comparing model families for node-level prediction of von Mises stress and fatigue life. The repository is organized around reproducible paper figures and checkpoint-based evaluation rather than a packaged software release.

## 2. Research objective and scope

The central objective is to test whether richer geometric encodings and feature-propagation (FP) architectures improve **fatigue-critical local prediction** compared with baseline PointNet-style and ArGEnT-style models. Current tracked comparisons cover:

- **Uniform lifing**: one fatigue-life law applied across the disc.
- **Zonal lifing**: fatigue-life behaviour varies by bore / transition / web / rim regions.
- **Edge-derived representations**: contour-resampled node sets with optional engineered geometric features.
- **Paper-oriented ablations**: FP vs non-FP, engineered head features, data efficiency, and joint stress supervision.

Out of scope for this repository snapshot are experimental deployment, certification claims, uncertainty quantification, and independent industrial validation.

## 3. Geometry and physics summary

The generated problem is a 2-D axisymmetric disc with a bore, web, upper/lower transitions, and a rim that includes a front C-groove and a rear drive-arm feature set. The physics workflow in `Data_gen/` builds geometry, meshes it, solves an axisymmetric FEM stress problem for Ti-6Al-4V, and converts phase-equivalent stress histories into Palmgren-Miner fatigue life.

Key modelling assumptions documented in `Data_gen/PIPELINE_NOTES.md` include:

- Ti-6Al-4V material properties.
- Seven weighted flight phases.
- Blade-equivalent rim-top traction loading.
- Uniform or zonal S-N lifing laws.
- Node-level stress and raw-life targets stored in HDF5.

## 4. Dataset representations table

| Dataset family | Lifing regime | Representation | Typical HDF5 name | Extra node features | Primary use |
|---|---|---|---|---|---|
| `Uniform/Edge/*` | Uniform | Edge contour nodes with derivative features | `disc_dataset_edge_deriv_uniform.h5` | None or representation-derived | Baseline paper comparisons |
| `Zonal/Edge/*` | Zonal | Edge contour nodes with derivative features | `disc_dataset_edge_deriv_zonal.h5` | None or representation-derived | Baseline paper comparisons |
| `Zonal/Edge_arc_feat/*` | Zonal | Edge nodes plus engineered head features | `disc_dataset_edge_deriv_zonal.h5` | Arc-length / head features | Engineered-feature ablation |
| `Zonal/Edge_10`, `Edge_25`, `Edge_50`, `Edge_75` | Zonal | Edge contour nodes | `disc_dataset_edge_deriv_zonal.h5` | Representation-derived | Data-efficiency subsampling |
| `Zonal/Edge_no_stress/*` | Zonal | Edge contour nodes | `disc_dataset_edge_deriv_zonal.h5` | Representation-derived | Joint-stress-supervision ablation |

## 5. Models

### `PointNetMLPJoint`
Point-set encoder plus MLP prediction head for **joint stress + log-life** prediction. This is the main non-FP PointNet baseline used throughout the comparisons.

### `PointNetMLPJoint_FP`
PointNet-style model with **feature propagation** added to improve local recovery. This is the main hypothesis model for the paper comparisons.

### `ArGEnT_self_att_noSDF`
ArGEnT/DeepONet-style attention model without signed-distance features. It serves as an alternative geometry-conditioned baseline.

### `PointNetMLPJoint_weighted`
Variant of `PointNetMLPJoint` trained with a weighted loss to emphasise fatigue-critical regions or targets. It is used in the joint-supervision / weighting comparisons, not in the main FP-vs-ArGEnT notebook.

## 6. Repository layout

```text
Data_gen/                     Data generation, FEM, HDF5 IO, diagnostics
Uniform/Edge/                Uniform-lifing edge checkpoints and training code
Zonal/Edge/                  Zonal-lifing edge checkpoints and training code
Zonal/Edge_arc_feat/         Zonal engineered-feature ablations
Zonal/Edge_10|25|50|75/      Zonal data-efficiency subsets
Zonal/Edge_no_stress/        Zonal no-stress / supervision ablation
Comparison/                  Notebook-based paper evaluation and helper utilities
run1.sh ... run4.sh          Batch/cluster training launch scripts
```

Within each model directory you should expect training scripts, `pn_models.py`, optional `benchmarks.py`, preset JSON, and `Trained_models/*.pt` checkpoints.

## 7. Installation and environment

Use a Python environment that includes at least:

- Python 3.10+
- PyTorch
- NumPy
- h5py
- pandas
- matplotlib
- scikit-learn

Depending on which parts of `Data_gen/` you run, you may also need the FEM/meshing stack used by the generator code. The repository does not currently ship a pinned `requirements.txt`, so environment management is left to the researcher.

A minimal notebook-oriented install often looks like:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy h5py pandas matplotlib scikit-learn jupyter
```

## 8. Data generation and training

### Data generation
The `Data_gen/` package contains the end-to-end geometry → mesh → FEM → fatigue-life → HDF5 pipeline. A representative command from `Data_gen/PIPELINE_NOTES.md` is:

```bash
python -m Data_gen.dataset_generator \
    --output-h5 Data_gen/output/disc_dataset_edge.h5 \
    --representation edge \
    --include-derivatives \
    --seed 7 \
    --num-samples 200 \
    --lifing-mode zonal
```

Useful validation utilities include:

- `python Data_gen/validate_contour.py --skip-stress`
- `python Data_gen/validate_fem_nominal.py`
- `python Data_gen/validate_rim_load_and_physics.py --case nominal --mesh medium --save-plots`

### Training
Training code is stored next to each model family, for example:

- `Uniform/Edge/PointNetMLPJoint/Training_script.py`
- `Uniform/Edge/PointNetMLPJoint_FP/Training_script.py`
- `Zonal/Edge/ArGEnT_self_att_noSDF/Training_script.py`

The root `run1.sh`-`run4.sh` files show the intended cluster-launch pattern for selected experiments. Adjust paths, modules, and conda environments for your own machine or scheduler.

## 9. Evaluation and paper notebooks

All paper-facing evaluation notes live under `Comparison/`. Start with [`Comparison/README.md`](Comparison/README.md), which documents the notebook suite, required local assets, expected result directories, and interpretation guidance.

The reference implementation already present in the repository is `Comparison/model_comparison.ipynb`; the paper-result notebook created for this repository snapshot is `Comparison/01_fp_vs_argent.ipynb`.

## 10. Limitations

- Checkpoints are compared by **post hoc notebook evaluation**; notebook results do not prove independence from training-time validation or checkpoint selection.
- Raw-life percentage metrics can be unstable when true life is very small.
- The repository is research code, not a hardened package.
- HDF5 datasets are typically local assets and are ignored by Git, so a fresh clone may not be runnable until datasets are regenerated or copied in.
- Results support comparative analysis, not design certification.

## 11. Reproducibility requirements

To reproduce the reported comparisons you need:

1. The exact checkpoint files under the tracked `Trained_models/` directories.
2. Matching HDF5 datasets in `Data_gen/output/` (for example `disc_dataset_edge_deriv_uniform.h5` and `disc_dataset_edge_deriv_zonal.h5`).
3. The same geometry-level split seed and evaluation fraction used by the notebook (`seed=42`, `fraction=0.20` for current paper comparisons).
4. A notebook/kernel environment with compatible versions of PyTorch, NumPy, pandas, matplotlib, and h5py.
5. The repository commit SHA recorded by the notebook outputs.

For paper tables, keep result labels explicit: the comparison notebooks describe current outputs as **validation-split evaluation** unless independent test provenance is proven separately.

## 12. Citation placeholder

If you use this repository in academic work, please cite the corresponding paper once bibliographic details are finalized.

```text
Author(s), “Title to be added,” journal / conference to be added, year to be added.
Repository: Disc_lifing_paper_v2
```
