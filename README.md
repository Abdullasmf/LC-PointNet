# LC-PointNet: Local-Context Point-Cloud Surrogates for Low-Life Fatigue Prediction

Code, trained checkpoints and evaluation notebooks accompanying:

> A. Fathalla and K. Höschler, **"Beyond Global Error: Low-Life Fatigue Prediction with Local-Context Point-Cloud Surrogates,"** ASME Turbo Expo 2027, Calgary, June 20-25, 2027, paper GT2027-200391. Brandenburgische Technische Universität (BTU) Cottbus-Senftenberg.

**Keywords:** fatigue life, turbine disc, surrogate model, point cloud, local context, attention.

## What the paper shows

A pooled error over all nodes hides how accurate a surrogate is where it matters: fatigue life is a steep nonlinear function of stress, so the few short-life nodes dominate design decisions yet barely move a pooled metric. The paper therefore reports errors both pooled and in life bins, on a synthetic turbine-disc benchmark, for four surrogates of about 250k parameters each:

| Model (paper name) | Idea |
|---|---|
| **LC-PointNet** (proposed) | PointNet-style encoder with a feature-propagation pathway returning native-resolution *local* features, concatenated with a *global* context vector and the Fourier-encoded query coordinate |
| **GC-PointNet** | LC-PointNet without the feature-propagation pathway (global context only) |
| **ArGEnT-A** | Adapted cross-attention variant of ArGEnT (Galerkin-type linear attention, RoPE) |
| **GINOT-A** | Adapted GINOT (farthest-point sampling + kNN local tokens, cross/self-attention) |

Headline findings (single training run per model, see [Limitations](#limitations)):

- LC-PointNet's pooled MAE is about 20% below GC-PointNet's; GC-PointNet and ArGEnT-A are within about 2% pooled, yet ArGEnT-A's MAE in the shortest-life bin (y < 2) is 3.4x higher.
- Adding engineered geometric features (+GF) helps ArGEnT-A most (pooled MAE -39%), then LC-PointNet (-37%) and GC-PointNet (-12%); GINOT-A shows no clear effect.
- Data study: LC-PointNet has the lowest y < 2 error at every training-set size, but the lowest *pooled* MAE only from 3,000 training geometries on.
- Stress-target study: training on life only (no stress target) raises LC-PointNet's y < 2 MAE by 44%, lowers GINOT-A's by 20%, and barely changes ArGEnT-A.

## Benchmark and dataset (paper Section on dataset)

A parameterised axisymmetric turbine-disc-like cross-section with five zones along the radius (bore, lower transition, web, upper transition, rim), a C-groove on the rim front and an annular drive arm on the rear.

- 11 disc-body dimensions + 11 rim-feature dimensions define a geometry (5 rim dimensions are derived from normalised controls).
- Latin hypercube sampling, **seed 7, 5,000 geometries**; body dimensions vary by at most +-5%, rim features by up to 30%, with wall-thickness, fillet and clearance constraints enforced.
- 3,395,000 boundary nodes in total; peak von Mises stress 75.5-1334 MPa; target y = log10(life) spans 0.98-10.21.
- **FEA:** linear-elastic axisymmetric analysis (scikit-fem, quadratic triangles, unstructured gmsh mesh; 0.3 mm near transitions/rim features, up to 2 mm in the interior). E = 114 GPa, nu = 0.34, rho = 4430 kg/m^3, omega_ref = 4000 rad/s. Loads: centrifugal body force plus a uniform blade-equivalent radial traction on the rim-top face (about 331 kN resultant from 60 blades x 3 g x omega^2 x 115 mm). One bore node fixed axially. Nodal von Mises stress via L2 projection.
- **Lifing:** 7 flight phases (taxi, takeoff, climb, cruise, descent, reverse thrust, taxi return) with speed ratios 0.20/1.00/0.86/0.78/0.55/0.46/0.18 and weights 0.20/0.08/0.15/0.32/0.12/0.05/0.08; phase stress = (speed ratio)^2 x sigma_vm, amplitude = stress/2, no mean-stress correction; piecewise power-law S-N curve with a knee, chosen per node by **radial zone**; damage D = sum(w_k / N_k); target y = log10(1/D). The S-N parameters are **synthetic**, and "life" is an equivalent life, not real flights. Only boundary nodes enter the dataset.

| Zone | knee stress (MPa) | knee cycles | m1 | m2 |
|---|---|---|---|---|
| Bore | 210 | 1e7 | 9.5 | 4 |
| Lower transition | 200 | 5e6 | 13 | 4.5 |
| Web | 140 | 1e7 | 8.5 | 4 |
| Upper transition | 180 | 5e6 | 13 | 4.5 |
| Rim | 120 | 1e7 | 9 | 4 |

The implementation lives in `Data_gen/` (`config.py` for constants and S-N curves, `geometry.py`, `mesh_ops.py`, `physics.py`, `sample_generator.py`, `dataset_generator.py`, `features.py`); see [`Data_gen/PIPELINE_NOTES.md`](Data_gen/PIPELINE_NOTES.md).

### Regenerating the paper dataset

The dataset (`.h5`) is **not committed** (`*.h5` is git-ignored). The paper dataset was generated with:

```bash
python -m Data_gen.dataset_generator \
    --representation edge \
    --output-h5 Data_gen/output/disc_dataset_edge_deriv_zonal.h5 \
    --num-samples 5000 \
    --lifing-mode zonal \
    --seed 7 \
    --include-derivatives
```

(Windows PowerShell: the same command with `python` replaced by your interpreter, e.g. a conda env.) Run it from the repository root. Generating 5,000 FEA samples is slow; budget accordingly. The training scripts expect exactly this file name in `Data_gen/output/`.

- `--representation edge` = boundary-contour nodes only (as in the paper).
- `--lifing-mode zonal` = per-zone S-N curves above. `uniform` also exists in the code but is **not used in the paper**.
- `--include-derivatives` = stores the engineered edge features (arc length, tangent components, curvature, curvature gradient) used in the +GF study (columns 3-7 of the node array; columns 0-1 are the (x, r) coordinates).

I could not verify bit-for-bit reproduction of the paper's dataset on a different machine (gmsh/scikit-fem versions affect meshes); check your node count against the paper's 3,395,000.

## Models and training protocol

- **Inputs:** boundary-node (x, r), scaled to [-1, 1] with training-set extremes; no zone label is given. **Outputs:** peak von Mises stress and log10 life at every node.
- **Loss:** MSE on standardised targets (stress and log-life), all nodes weighted equally. AdamW, lr 3e-4, weight decay 1e-4, gradient-norm clip 0.5, **batch size 2 geometries**; LR halved after 20 epochs without validation improvement, stop after 100; best-validation checkpoint kept. One run per model from scratch, seed 42.
- **Split:** 80/20 by geometry (4,000 train / 1,000 validation, `train_test_split(..., test_size=0.2, random_state=42)`).
- **Metrics:** MAE and RMSE of log-life (decades), pooled and in five life bins by true y: `<2`, `2-3`, `3-4`, `4-6`, `>=6`.
- **+GF:** the five engineered features enter the geometry path (LC/GC-PointNet, ArGEnT-A) or the local MLP (GINOT-A).

### Paper name to repo name

| Paper | Directory / class in this repo |
|---|---|
| GC-PointNet | `PointNetMLPJoint` |
| LC-PointNet | `PointNetMLPJoint_FP` |
| ArGEnT-A | `ArGEnT_self_att_noSDF` |
| GINOT-A | `GINOT` |
| +GF variants | `Zonal/Edge_arc_feat/*` (`PointNetMLPJoint_headfeat` = GC + GF, `PointNetMLPJoint_FP_headfeat` = LC + GF) |
| Data study (10/25/50/75%) | `Zonal/Edge_10`, `Edge_25`, `Edge_50`, `Edge_75` (100% = `Zonal/Edge`) |
| Life-only (no stress target) | `Zonal/Edge_no_stress` |

## Repository layout

```text
Data_gen/        Geometry -> mesh -> FEA -> lifing -> HDF5 pipeline (+ validation scripts)
Zonal/           Everything used in the paper (zonal lifing)
  Edge/                  baseline: one folder per model
  Edge_arc_feat/         +GF study
  Edge_10|25|50|75/      data study (train_data_percent in each Training_script.py)
  Edge_no_stress/        life-only study (TARGET_NAMES = ["LogLife"])
Uniform/Edge/    Uniform-lifing variants; NOT part of the paper
Comparison/      Evaluation notebooks, helpers, saved results (see Comparison/README.md)
```

Each model folder holds `Training_script.py`, `pn_models.py`, presets (`model_presets.json`/`.py`) and `Trained_models/*.pt`. `Trained_models/DEP/` holds deprecated checkpoints; `PointNetMLPJoint_weighted` (weighted loss) and the `compare_*`/`*_audit` helpers are exploratory and **not in the paper**.

## Installation

There is no pinned `requirements.txt`. You need Python 3.10+, `torch`, `numpy`, `h5py`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter`, plus `scikit-fem`, `gmsh` (and their dependencies, e.g. `scipy`) for data generation.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy h5py pandas matplotlib scikit-learn jupyter scikit-fem gmsh scipy
```

## Training

Training is configured per folder (the ablation is fixed by constants at the top of each `Training_script.py`: target set, feature columns, dataset file). Scripts read `Data_gen/output/disc_dataset_edge_deriv_zonal.h5` automatically. Example, from a model folder:

```bash
cd Zonal/Edge/GINOT
python Training_script.py --preset MATCH_250K --batch 2        # GINOT-A has an argparse CLI
```

The PointNet and ArGEnT scripts have no CLI: their `__main__` block calls `main(<preset>, <batch>)` with hard-coded values (e.g. LC-PointNet `main("S_full_ln_pos12_fp", 1)`, ArGEnT-A `main("S", batch=1)`). **These defaults do not equal the paper's batch size of 2** - edit the call (or import `main`) before reproducing. Presets and batch sizes are code-level settings; check `model_presets.*` and the parameter count printed at start-up (target about 250k).

Known caveats: the `GPU*.py` / `GPUL1.py` wrappers import a git-ignored `Training_script_L1` and will not run from a clean clone; the old `run1.sh`-`run4.sh` scripts no longer exist.

## Evaluation: reproducing the paper's tables and figures

Notebooks in `Comparison/` rebuild models from the checkpoints, evaluate on the validation split (seed 42, 20%) and write CSV/JSON/figures to `Comparison/results/`:

| Notebook | Paper content |
|---|---|
| `01_fp_vs_argent.ipynb` | Baseline comparison (Table 2) |
| `02_engineered_geometric_features.ipynb` | +GF study (Table 3) |
| `03_data_efficiency.ipynb` | Data study |
| `04_joint_stress_supervision.ipynb` | Stress-target study |

Requires the dataset above in `Data_gen/output/`. Details and artifact layout: [`Comparison/README.md`](Comparison/README.md). (`Comparison/model_comparison.ipynb` is an earlier notebook; `01_`-`04_` are the paper ones. Uniform-lifing outputs in `results/01_*` are extra.)

## Key results (from the paper; validation split, log-life in decades)

Baseline MAE (Table 2):

| Model | Pooled | y<2 | 2-3 | 3-4 | 4-6 | >=6 |
|---|---|---|---|---|---|---|
| GC-PointNet | 0.0164 | 0.0759 | 0.0748 | 0.0773 | 0.0170 | 0.0142 |
| LC-PointNet | 0.0130 | 0.0479 | 0.0522 | 0.0535 | 0.0128 | 0.0118 |
| ArGEnT-A | 0.0160 | 0.2565 | 0.1887 | 0.0955 | 0.0162 | 0.0124 |
| GINOT-A | 0.0196 | 0.2360 | 0.1793 | 0.1003 | 0.0202 | 0.0157 |

Baseline RMSE (Table 2):

| Model | Pooled | y<2 | 2-3 | 3-4 | 4-6 | >=6 |
|---|---|---|---|---|---|---|
| GC-PointNet | 0.0357 | 0.1034 | 0.1070 | 0.1071 | 0.0391 | 0.0285 |
| LC-PointNet | 0.0287 | 0.0722 | 0.0792 | 0.0833 | 0.0313 | 0.0235 |
| ArGEnT-A | 0.0449 | 0.2722 | 0.2276 | 0.1423 | 0.0477 | 0.0303 |
| GINOT-A | 0.0506 | 0.2620 | 0.2247 | 0.1462 | 0.0544 | 0.0373 |

With engineered geometric features, MAE (Table 3):

| Model | Pooled | y<2 | 2-3 | 3-4 | 4-6 | >=6 |
|---|---|---|---|---|---|---|
| GC-PointNet + GF | 0.0144 | 0.0586 | 0.0594 | 0.0684 | 0.0148 | 0.0126 |
| LC-PointNet + GF | 0.0081 | 0.0262 | 0.0274 | 0.0273 | 0.0074 | 0.0080 |
| ArGEnT-A + GF | 0.0098 | 0.0315 | 0.0330 | 0.0355 | 0.0094 | 0.0093 |
| GINOT-A + GF | 0.0205 | 0.2494 | 0.1691 | 0.1007 | 0.0208 | 0.0169 |

With engineered geometric features, RMSE (Table 3):

| Model | Pooled | y<2 | 2-3 | 3-4 | 4-6 | >=6 |
|---|---|---|---|---|---|---|
| GC-PointNet + GF | 0.0312 | 0.0864 | 0.0849 | 0.0964 | 0.0345 | 0.0249 |
| LC-PointNet + GF | 0.0177 | 0.0474 | 0.0545 | 0.0494 | 0.0177 | 0.0156 |
| ArGEnT-A + GF | 0.0215 | 0.0460 | 0.0495 | 0.0568 | 0.0226 | 0.0189 |
| GINOT-A + GF | 0.0478 | 0.2694 | 0.2137 | 0.1430 | 0.0501 | 0.0356 |

Validation life bins hold 1,030 / 3,491 / 8,723 / 239,075 / 426,681 nodes (679,000 total): the shortest-life bins are tiny, so their numbers are noisy.

## Limitations

- **Single seed, single training run per model**; no confidence intervals. Small differences are not established.
- The 1,000-geometry validation split also drove early stopping and checkpoint selection, so it is **not an independent test set**.
- GC-PointNet is absent from the data and stress-target studies.
- S-N parameters and the disc geometry are **synthetic**; this is a benchmark, not a real component, and not a design or certification tool.
- The loss weights all nodes equally; short-life weighting was not tested. The paper's hypotheses on *why* the models differ are untested.
- Research code, not a packaged library; the dataset must be regenerated locally.

## Citation

The paper is under review.

## Acknowledgements

AI tools (Claude, Perplexity, GitHub Copilot) were used during development, as disclosed in the paper.
