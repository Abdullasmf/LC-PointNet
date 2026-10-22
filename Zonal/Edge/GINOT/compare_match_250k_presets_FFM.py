"""Compare trained GINOT-A MATCH_250K and MATCH_250K_FFM12 checkpoints.

Place this file in Zonal/Edge/GINOT and run, for example:
    python compare_match_250k_presets.py --device cuda

The script performs evaluation only. It never trains models, modifies checkpoints,
or changes source data. It evaluates the deterministic geometry-level validation
split and writes FFM12-specific artifacts under ffm12_comparison_results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import Training_script as ts
from pn_models import GINOT_A, count_trainable_parameters

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "Trained_models"
DEFAULT_OUTDIR = ROOT / "ffm12_comparison_results"
THRESHOLDS: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)
MODEL_KEYS = ("MATCH_250K", "MATCH_250K_FFM12")


@dataclass
class LoadedModel:
    key: str
    path: Path
    model: torch.nn.Module
    coord_center: torch.Tensor
    coord_half_range: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    target_names: List[str]
    parameter_count: int
    arch: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MATCH_250K versus MATCH_250K_FFM12 on the deterministic "
            "geometry-level validation split."
        )
    )
    parser.add_argument(
        "--standard-ckpt",
        type=Path,
        default=None,
        help="Baseline MATCH_250K checkpoint path.",
    )
    parser.add_argument(
        "--ffm12-ckpt",
        type=Path,
        default=None,
        help="MATCH_250K_FFM12 checkpoint path.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of paired geometry-level bootstrap resamples; 0 disables CIs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_one(candidates: List[Path], description: str, flag: str) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    rendered = "\n  ".join(str(path) for path in candidates) if candidates else "<none>"
    raise RuntimeError(
        f"Expected exactly one {description} checkpoint in {CHECKPOINT_DIR}; found "
        f"{len(candidates)}:\n  {rendered}\nUse {flag} to choose explicitly."
    )


def checkpoint_paths(args: argparse.Namespace) -> Dict[str, Path]:
    if args.standard_ckpt is not None:
        standard = args.standard_ckpt
    else:
        standard = _select_one(
            [
                path
                for path in sorted(CHECKPOINT_DIR.glob("ginot_a_match_250k_*.pt"))
                if "_hires_" not in path.stem.lower()
                and "_ffm12_" not in path.stem.lower()
            ],
            "baseline MATCH_250K",
            "--standard-ckpt",
        )

    if args.ffm12_ckpt is not None:
        ffm12 = args.ffm12_ckpt
    else:
        ffm12 = _select_one(
            sorted(CHECKPOINT_DIR.glob("ginot_a_match_250k_ffm12_*.pt")),
            "MATCH_250K_FFM12",
            "--ffm12-ckpt",
        )

    if standard.resolve() == ffm12.resolve():
        raise RuntimeError(
            "MATCH_250K and MATCH_250K_FFM12 checkpoint paths resolve to the same file."
        )

    for path in (standard, ffm12):
        if not path.is_file():
            raise FileNotFoundError(path)

    return {
        "MATCH_250K": standard,
        "MATCH_250K_FFM12": ffm12,
    }


def load_model(key: str, path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_family") != "GINOT-A":
        raise RuntimeError(f"{path.name} is not tagged model_family='GINOT-A'.")

    arch = checkpoint.get("arch")
    if not isinstance(arch, dict) or not isinstance(arch.get("ginot_cfg"), dict):
        raise RuntimeError(f"{path.name} has no usable arch['ginot_cfg'] configuration.")

    cfg = dict(arch["ginot_cfg"])
    out_dim = int(arch.get("out_dim", len(checkpoint.get("target_mean", []))))
    in_channels = int(arch.get("in_channels", len(checkpoint.get("extra_feat_cols", []))))

    if in_channels != 0:
        raise RuntimeError(
            f"This evaluator is for coordinate-only Zonal/Edge GINOT checkpoints, but "
            f"{path.name} declares in_channels={in_channels}."
        )

    model = GINOT_A(cfg=cfg, out_dim=out_dim, in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    target_names = list(checkpoint.get("target_names", ts.TARGET_NAMES))
    if target_names != ["Stress", "LogLife"]:
        raise RuntimeError(f"Unexpected target order in {path.name}: {target_names}")

    return LoadedModel(
        key=key,
        path=path,
        model=model,
        coord_center=torch.as_tensor(checkpoint["coord_center"], dtype=torch.float32),
        coord_half_range=torch.as_tensor(checkpoint["coord_half_range"], dtype=torch.float32),
        target_mean=torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32),
        target_std=torch.as_tensor(checkpoint["target_std"], dtype=torch.float32),
        target_names=target_names,
        parameter_count=count_trainable_parameters(model),
        arch=arch,
    )


def validate_ffm12_control(baseline: LoadedModel, ffm12: LoadedModel) -> None:
    """Require architecture parity except coordinate Fourier frequency."""
    base_cfg = dict(baseline.arch["ginot_cfg"])
    test_cfg = dict(ffm12.arch["ginot_cfg"])

    required_same = [
        "geom_coord_dim",
        "query_coord_dim",
        "n_centroids",
        "n_neighbors",
        "local_mlp_widths",
        "token_dim",
        "encoder_heads",
        "encoder_cross_attn_layers",
        "encoder_self_attn_layers",
        "decoder_cross_attn_layers",
        "decoder_heads",
        "decoder_mlp_widths",
        "head_mlp_widths",
        "dropout",
        "encoder_gf_dim",
        "head_gf_dim",
    ]

    mismatches = [
        f"{key}: baseline={base_cfg.get(key)!r}, FFM12={test_cfg.get(key)!r}"
        for key in required_same
        if base_cfg.get(key) != test_cfg.get(key)
    ]

    if mismatches:
        raise RuntimeError(
            "MATCH_250K_FFM12 differs from MATCH_250K in settings other than "
            "Fourier frequency. This is not a clean FFM sensitivity control:\n  "
            + "\n  ".join(mismatches)
        )

    base_geom_freq = int(base_cfg.get("geom_posenc_freqs", -1))
    base_query_freq = int(base_cfg.get("query_posenc_freqs", -1))
    ffm_geom_freq = int(test_cfg.get("geom_posenc_freqs", -1))
    ffm_query_freq = int(test_cfg.get("query_posenc_freqs", -1))

    if ffm_geom_freq != 12 or ffm_query_freq != 12:
        raise RuntimeError(
            "The selected MATCH_250K_FFM12 checkpoint must use 12 geometry and "
            f"12 query Fourier frequencies; found geometry={ffm_geom_freq}, "
            f"query={ffm_query_freq}."
        )

    if base_geom_freq == ffm_geom_freq and base_query_freq == ffm_query_freq:
        raise RuntimeError("Baseline and FFM12 have identical Fourier-frequency settings.")

    if baseline.target_names != ffm12.target_names:
        raise RuntimeError(
            f"Target-name mismatch: baseline={baseline.target_names}, "
            f"FFM12={ffm12.target_names}."
        )

    print("\nFourier-frequency control verified:")
    print(
        f"  MATCH_250K: geometry={base_geom_freq}, query={base_query_freq}, "
        f"parameters={baseline.parameter_count:,}"
    )
    print(
        f"  MATCH_250K_FFM12: geometry={ffm_geom_freq}, query={ffm_query_freq}, "
        f"parameters={ffm12.parameter_count:,}"
    )


def validation_tensors() -> Tuple[List[torch.Tensor], List[int], Path]:
    h5_path = ROOT.parents[2] / "Data_gen" / "output" / ts.H5_FILENAME
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Validation HDF5 file was not found: {h5_path}. This script must be run "
            "from Zonal/Edge/GINOT in the repository containing Data_gen/output/."
        )

    all_tensors = ts.load_h5_pointsets(h5_path)
    indices = list(range(len(all_tensors)))
    _, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    return [all_tensors[i] for i in val_indices], list(val_indices), h5_path


def predict(model: LoadedModel, tensor: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    width = int(tensor.shape[1])
    target_cols = ts.target_cols_for_width(width)
    coords = tensor[:, [0, 1]].to(torch.float32)
    true_raw = tensor[:, list(target_cols)].to(torch.float32)
    norm_coords = (coords - model.coord_center) / torch.clamp(model.coord_half_range, min=1e-8)

    geom = norm_coords.unsqueeze(0).to(device)
    queries = norm_coords.unsqueeze(0).to(device)
    geom_feats = torch.empty((1, coords.shape[0], 0), dtype=torch.float32, device=device)

    with torch.inference_mode():
        pred_z = model.model(geom, queries, geom_feats)

    pred_raw = pred_z.squeeze(0).cpu() * model.target_std + model.target_mean
    return true_raw.numpy(), pred_raw.numpy()


def metric_row(true_life: np.ndarray, pred_life: np.ndarray, threshold: float | None) -> Dict[str, float]:
    mask = np.ones(true_life.shape[0], dtype=bool) if threshold is None else true_life < threshold
    n = int(mask.sum())
    if n == 0:
        return {"n_nodes": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}

    error = pred_life[mask] - true_life[mask]
    return {
        "n_nodes": n,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def all_bands() -> Iterable[Tuple[str, float | None]]:
    yield "all", None
    for threshold in THRESHOLDS:
        yield f"log_life_lt_{threshold:g}", threshold


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(values.mean())
    if n_bootstrap <= 0 or values.size < 2:
        return mean, np.nan, np.nan

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return mean, float(lo), float(hi)


def plot_pooled_metrics(rows: List[Dict], outdir: Path) -> None:
    labels = ["<2", "<3", "<4", "<5", "<6", "all"]
    band_order = ["log_life_lt_2", "log_life_lt_3", "log_life_lt_4", "log_life_lt_5", "log_life_lt_6", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)

    for ax, metric, title in zip(axes, ("mae", "rmse"), ("Log-life MAE", "Log-life RMSE")):
        for key, marker in zip(MODEL_KEYS, ("o", "s")):
            lookup = {row["band"]: row for row in rows if row["model"] == key}
            ax.plot(
                labels,
                [lookup[band][metric] for band in band_order],
                marker=marker,
                linewidth=2,
                label=key,
            )
        ax.set_xlabel("Validation subset: true log-life")
        ax.set_ylabel(f"{metric.upper()} [decades]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.savefig(outdir / "pooled_loglife_metrics_ffm12_vs_baseline.png", dpi=220)
    plt.close(fig)


def plot_geometry_deltas(per_geom_rows: List[Dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)

    for ax, band, title in zip(
        axes,
        ("log_life_lt_2", "log_life_lt_3"),
        ("Low-life <2", "Low-life <3"),
    ):
        by_id: Dict[int, Dict[str, Dict]] = {}
        for row in per_geom_rows:
            if row["band"] == band and np.isfinite(row["mae"]):
                by_id.setdefault(int(row["geometry_index"]), {})[row["model"]] = row

        baseline_values = []
        ffm12_values = []
        for entries in by_id.values():
            if all(key in entries for key in MODEL_KEYS):
                baseline_values.append(entries["MATCH_250K"]["mae"])
                ffm12_values.append(entries["MATCH_250K_FFM12"]["mae"])

        if baseline_values:
            baseline_a = np.asarray(baseline_values)
            ffm12_a = np.asarray(ffm12_values)
            limit = max(np.max(baseline_a), np.max(ffm12_a)) * 1.05
            ax.scatter(baseline_a, ffm12_a, alpha=0.65, s=22)
            ax.plot([0, limit], [0, limit], "k--", linewidth=1)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_xlabel("MATCH_250K per-geometry MAE [decades]")
            ax.set_ylabel("MATCH_250K_FFM12 per-geometry MAE [decades]")
            ax.text(0.04, 0.94, f"n geometries = {len(baseline_a)}", transform=ax.transAxes, va="top")
        else:
            ax.text(0.5, 0.5, "No geometries contain this subset", ha="center", va="center")

        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.savefig(outdir / "paired_geometry_low_life_mae_ffm12_vs_baseline.png", dpi=220)
    plt.close(fig)


def decision_text(
    pooled_rows: List[Dict],
    per_geom_rows: List[Dict],
    n_bootstrap: int,
    seed: int,
) -> str:
    lines = [
        "GINOT-A Fourier-frequency sensitivity summary",
        "=" * 46,
        "Comparison: MATCH_250K baseline versus MATCH_250K_FFM12.",
        "The intended architecture change is coordinate Fourier frequency only.",
        "Decision priority: log-life MAE/RMSE in true_loglife <2, then <3, then pooled error.",
        "This is a validation-split comparison, not an independent physical test.",
        "",
    ]

    pooled = {(row["model"], row["band"]): row for row in pooled_rows}
    for band in ("log_life_lt_2", "log_life_lt_3", "all"):
        baseline = pooled[("MATCH_250K", band)]
        ffm12 = pooled[("MATCH_250K_FFM12", band)]
        lines.append(
            f"{band}: n={baseline['n_nodes']}; "
            f"MAE baseline={baseline['mae']:.6f}, FFM12={ffm12['mae']:.6f}; "
            f"RMSE baseline={baseline['rmse']:.6f}, FFM12={ffm12['rmse']:.6f}."
        )

    lines.append("")
    for band in ("log_life_lt_2", "log_life_lt_3"):
        by_geom: Dict[int, Dict[str, Dict]] = {}
        for row in per_geom_rows:
            if row["band"] == band and np.isfinite(row["mae"]):
                by_geom.setdefault(int(row["geometry_index"]), {})[row["model"]] = row

        deltas = np.asarray(
            [
                entries["MATCH_250K_FFM12"]["mae"] - entries["MATCH_250K"]["mae"]
                for entries in by_geom.values()
                if all(key in entries for key in MODEL_KEYS)
            ]
        )
        mean, lo, hi = bootstrap_ci(deltas, n_bootstrap, seed)
        wins = int(np.sum(deltas < 0)) if deltas.size else 0
        lines.append(
            f"{band}, paired geometry-level MAE delta (FFM12 - baseline): "
            f"mean={mean:.6f}; 95% bootstrap CI=[{lo:.6f}, {hi:.6f}]; "
            f"FFM12 wins {wins}/{deltas.size} geometries."
        )

    low2_base = pooled[("MATCH_250K", "log_life_lt_2")]
    low2_ffm12 = pooled[("MATCH_250K_FFM12", "log_life_lt_2")]
    low3_base = pooled[("MATCH_250K", "log_life_lt_3")]
    low3_ffm12 = pooled[("MATCH_250K_FFM12", "log_life_lt_3")]

    if low2_base["n_nodes"] == 0:
        recommendation = "No <2 validation nodes exist; select using <3 and pooled metrics."
    elif low2_ffm12["mae"] < low2_base["mae"] and low2_ffm12["rmse"] < low2_base["rmse"]:
        recommendation = (
            "FFM12 improves both <2 MAE and RMSE. Inspect <3 and per-geometry results "
            "before deciding whether higher Fourier bandwidth should replace the baseline."
        )
    elif low2_ffm12["mae"] > low2_base["mae"] and low2_ffm12["rmse"] > low2_base["rmse"]:
        recommendation = "FFM12 worsens both <2 MAE and RMSE; the six-frequency baseline is preferred."
    elif low3_ffm12["mae"] < low3_base["mae"] and low3_ffm12["rmse"] < low3_base["rmse"]:
        recommendation = (
            "The <2 result is mixed, but FFM12 improves both <3 MAE and RMSE. "
            "Inspect paired geometry-level results before selecting a primary configuration."
        )
    else:
        recommendation = (
            "Fourier-frequency results are mixed. Do not select automatically; inspect low-life metrics, "
            "per-geometry deltas, parameter counts, and computational cost."
        )

    lines.extend(["", recommendation])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = checkpoint_paths(args)
    loaded = {key: load_model(key, path, device) for key, path in paths.items()}
    validate_ffm12_control(loaded["MATCH_250K"], loaded["MATCH_250K_FFM12"])

    val_tensors, val_indices, h5_path = validation_tensors()
    print(f"\nDevice: {device}")
    print(f"HDF5: {h5_path}")
    print(f"Validation geometries: {len(val_tensors)}; split: test_size=0.2, random_state=42")
    for key in MODEL_KEYS:
        info = loaded[key]
        print(f"{key}: {info.path.name} | {info.parameter_count:,} trainable parameters")

    predictions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {key: [] for key in MODEL_KEYS}
    for geom_no, tensor in enumerate(val_tensors, start=1):
        for key in MODEL_KEYS:
            predictions[key].append(predict(loaded[key], tensor, device))
        if geom_no % 50 == 0 or geom_no == len(val_tensors):
            print(f"Evaluated {geom_no}/{len(val_tensors)} validation geometries")

    pooled_rows: List[Dict] = []
    per_geom_rows: List[Dict] = []
    for key in MODEL_KEYS:
        all_true = np.concatenate([pair[0][:, 1] for pair in predictions[key]])
        all_pred = np.concatenate([pair[1][:, 1] for pair in predictions[key]])

        for band, threshold in all_bands():
            row = metric_row(all_true, all_pred, threshold)
            pooled_rows.append(
                {
                    "model": key,
                    "band": band,
                    "threshold": "all" if threshold is None else threshold,
                    **row,
                }
            )

        for local_i, (true, pred) in enumerate(predictions[key]):
            geometry_index = int(val_indices[local_i])
            for band, threshold in all_bands():
                row = metric_row(true[:, 1], pred[:, 1], threshold)
                per_geom_rows.append(
                    {
                        "model": key,
                        "geometry_index": geometry_index,
                        "band": band,
                        "threshold": "all" if threshold is None else threshold,
                        **row,
                    }
                )

    write_csv(args.outdir / "pooled_loglife_metrics_ffm12_vs_baseline.csv", pooled_rows)
    write_csv(args.outdir / "per_geometry_loglife_metrics_ffm12_vs_baseline.csv", per_geom_rows)
    plot_pooled_metrics(pooled_rows, args.outdir)
    plot_geometry_deltas(per_geom_rows, args.outdir)

    summary = decision_text(pooled_rows, per_geom_rows, args.bootstrap, args.seed)
    (args.outdir / "decision_summary_ffm12_vs_baseline.txt").write_text(summary, encoding="utf-8")

    manifest = {
        "comparison_type": "Fourier-frequency sensitivity: MATCH_250K vs MATCH_250K_FFM12",
        "hdf5_path": str(h5_path),
        "split": {"test_size": 0.2, "random_state": 42, "n_validation_geometries": len(val_tensors)},
        "checkpoints": {
            key: {
                "path": str(loaded[key].path),
                "parameter_count": loaded[key].parameter_count,
                "ginot_cfg": loaded[key].arch["ginot_cfg"],
            }
            for key in MODEL_KEYS
        },
        "thresholds": list(THRESHOLDS),
        "selection_rule": (
            "Prioritize <2 MAE/RMSE, then <3, then pooled metrics; inspect paired geometry results "
            "and verify only Fourier frequencies changed."
        ),
    }
    (args.outdir / "run_manifest_ffm12_vs_baseline.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + summary)
    print(f"Saved CSV tables, figures, manifest, and decision summary to: {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FFM12 comparison failed: {exc}", file=sys.stderr)
        raise
