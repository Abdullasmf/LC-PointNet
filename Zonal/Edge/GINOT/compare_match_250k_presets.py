"""Compare trained GINOT-A MATCH_250K and MATCH_250K_HIRES checkpoints.

Place this file in Zonal/Edge/GINOT and run from that directory, for example:
    python compare_match_250k_presets.py
    python compare_match_250k_presets.py --device cuda --bootstrap 2000

It performs evaluation only. It never trains, writes checkpoints, or changes
source data. Results are written under preset_comparison_results/.
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
DEFAULT_OUTDIR = ROOT / "preset_comparison_results"
THRESHOLDS: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)
MODEL_KEYS = ("MATCH_250K", "MATCH_250K_HIRES")


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
        description="Evaluate MATCH_250K vs MATCH_250K_HIRES on the deterministic validation split."
    )
    parser.add_argument("--standard-ckpt", type=Path, default=None)
    parser.add_argument("--hires-ckpt", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of paired geometry-level bootstrap resamples; 0 disables CIs.")
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


def find_one(pattern: str) -> Path:
    matches = sorted(CHECKPOINT_DIR.glob(pattern))
    if len(matches) != 1:
        rendered = "\n  ".join(str(x) for x in matches) if matches else "<none>"
        raise RuntimeError(
            f"Expected exactly one checkpoint matching '{pattern}' in {CHECKPOINT_DIR}; found "
            f"{len(matches)}:\n  {rendered}\nUse --standard-ckpt or --hires-ckpt to choose explicitly."
        )
    return matches[0]


def checkpoint_paths(args: argparse.Namespace) -> Dict[str, Path]:
    if args.standard_ckpt is not None:
        standard = args.standard_ckpt
    else:
        standard_candidates = [
            path
            for path in sorted(CHECKPOINT_DIR.glob("ginot_a_match_250k_*.pt"))
            if "_hires_" not in path.stem.lower()
        ]

        if len(standard_candidates) != 1:
            rendered = (
                "\n  ".join(str(path) for path in standard_candidates)
                if standard_candidates
                else "<none>"
            )
            raise RuntimeError(
                "Expected exactly one non-HiRes MATCH_250K checkpoint in "
                f"{CHECKPOINT_DIR}; found {len(standard_candidates)}:\n"
                f"  {rendered}\n"
                "Use --standard-ckpt to choose explicitly."
            )

        standard = standard_candidates[0]

    hires = args.hires_ckpt or find_one("ginot_a_match_250k_hires_*.pt")

    if standard.resolve() == hires.resolve():
        raise RuntimeError(
            "Standard and HiRes checkpoint paths resolve to the same file."
        )

    for path in (standard, hires):
        if not path.is_file():
            raise FileNotFoundError(path)

    return {
        "MATCH_250K": standard,
        "MATCH_250K_HIRES": hires,
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
            f"This evaluator is for coordinate-only Zonal/Edge GINOT checkpoints, but {path.name} "
            f"declares in_channels={in_channels}."
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


def validation_tensors() -> Tuple[List[torch.Tensor], List[int], Path]:
    h5_path = ROOT.parents[2] / "Data_gen" / "output" / ts.H5_FILENAME
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Validation HDF5 file was not found: {h5_path}. This script must be run from the "
            "Zonal/Edge/GINOT folder in the repository containing Data_gen/output/."
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
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


def all_bands() -> Iterable[Tuple[str, float | None]]:
    yield "all", None
    for threshold in THRESHOLDS:
        yield f"log_life_lt_{threshold:g}", threshold


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
            lookup = {r["band"]: r for r in rows if r["model"] == key}
            vals = [lookup[band][metric] for band in band_order]
            ax.plot(labels, vals, marker=marker, linewidth=2, label=key)
        ax.set_xlabel("Validation subset: true log-life")
        ax.set_ylabel(f"{metric.upper()} [decades]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(outdir / "pooled_loglife_metrics_by_threshold.png", dpi=220)
    plt.close(fig)


def plot_geometry_deltas(per_geom_rows: List[Dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, band, title in zip(axes, ("log_life_lt_2", "log_life_lt_3"), ("Low-life <2", "Low-life <3")):
        by_id: Dict[int, Dict[str, Dict]] = {}
        for row in per_geom_rows:
            if row["band"] == band and np.isfinite(row["mae"]):
                by_id.setdefault(int(row["geometry_index"]), {})[row["model"]] = row
        standard = []
        hires = []
        for entries in by_id.values():
            if all(key in entries for key in MODEL_KEYS):
                standard.append(entries["MATCH_250K"]["mae"])
                hires.append(entries["MATCH_250K_HIRES"]["mae"])
        if standard:
            standard_a = np.asarray(standard)
            hires_a = np.asarray(hires)
            lim = max(np.max(standard_a), np.max(hires_a)) * 1.05
            ax.scatter(standard_a, hires_a, alpha=0.65, s=22)
            ax.plot([0, lim], [0, lim], "k--", linewidth=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("MATCH_250K per-geometry MAE [decades]")
            ax.set_ylabel("MATCH_250K_HIRES per-geometry MAE [decades]")
            ax.text(0.04, 0.94, f"n geometries = {len(standard)}", transform=ax.transAxes, va="top")
        else:
            ax.text(0.5, 0.5, "No geometries contain this subset", ha="center", va="center")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.savefig(outdir / "paired_geometry_low_life_mae.png", dpi=220)
    plt.close(fig)


def decision_text(pooled_rows: List[Dict], per_geom_rows: List[Dict], n_bootstrap: int, seed: int) -> str:
    lines = [
        "GINOT-A preset decision summary",
        "=" * 34,
        "Decision priority: log-life MAE/RMSE in true_loglife <2, then <3, then pooled error.",
        "The result is a validation-split comparison, not an independent physical test.",
        "",
    ]
    pooled = {(r["model"], r["band"]): r for r in pooled_rows}
    for band in ("log_life_lt_2", "log_life_lt_3", "all"):
        a = pooled[("MATCH_250K", band)]
        b = pooled[("MATCH_250K_HIRES", band)]
        lines.append(
            f"{band}: n={a['n_nodes']}; MAE standard={a['mae']:.6f}, hires={b['mae']:.6f}; "
            f"RMSE standard={a['rmse']:.6f}, hires={b['rmse']:.6f}."
        )
    lines.append("")
    for band in ("log_life_lt_2", "log_life_lt_3"):
        by_geom: Dict[int, Dict[str, Dict]] = {}
        for row in per_geom_rows:
            if row["band"] == band and np.isfinite(row["mae"]):
                by_geom.setdefault(int(row["geometry_index"]), {})[row["model"]] = row
        deltas = np.asarray([
            entries["MATCH_250K_HIRES"]["mae"] - entries["MATCH_250K"]["mae"]
            for entries in by_geom.values()
            if all(key in entries for key in MODEL_KEYS)
        ])
        mean, lo, hi = bootstrap_ci(deltas, n_bootstrap, seed)
        wins = int(np.sum(deltas < 0)) if deltas.size else 0
        lines.append(
            f"{band}, paired geometry-level MAE delta (HIRES - standard): mean={mean:.6f}; "
            f"95% bootstrap CI=[{lo:.6f}, {hi:.6f}]; HIRES wins {wins}/{deltas.size} geometries."
        )
    low2_a = pooled[("MATCH_250K", "log_life_lt_2")]
    low2_b = pooled[("MATCH_250K_HIRES", "log_life_lt_2")]
    low3_a = pooled[("MATCH_250K", "log_life_lt_3")]
    low3_b = pooled[("MATCH_250K_HIRES", "log_life_lt_3")]
    if low2_a["n_nodes"] == 0:
        recommendation = "No <2 validation nodes exist; select using <3 and pooled metrics, and state the limitation."
    elif low2_b["mae"] < low2_a["mae"] and low2_b["rmse"] < low2_a["rmse"]:
        recommendation = "Preliminary recommendation: use MATCH_250K_HIRES for the ablation, because it improves both <2 MAE and RMSE."
    elif low2_b["mae"] > low2_a["mae"] and low2_b["rmse"] > low2_a["rmse"]:
        recommendation = "Preliminary recommendation: use MATCH_250K, because the higher token resolution worsens both <2 MAE and RMSE."
    elif low3_b["mae"] < low3_a["mae"] and low3_b["rmse"] < low3_a["rmse"]:
        recommendation = "<2 metrics are mixed; provisional choice is MATCH_250K_HIRES based on consistent <3 improvement. Review paired geometry plots before finalizing."
    else:
        recommendation = "Metrics are mixed. Do not select automatically; inspect the figures and paired geometry-level deltas before choosing a primary preset."
    lines.extend(["", recommendation])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(args)
    loaded = {key: load_model(key, path, device) for key, path in paths.items()}
    val_tensors, val_indices, h5_path = validation_tensors()
    print(f"Device: {device}")
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
        all_true = np.concatenate([x[0][:, 1] for x in predictions[key]])
        all_pred = np.concatenate([x[1][:, 1] for x in predictions[key]])
        for band, threshold in all_bands():
            row = metric_row(all_true, all_pred, threshold)
            pooled_rows.append({"model": key, "band": band, "threshold": "all" if threshold is None else threshold, **row})
        for local_i, (true, pred) in enumerate(predictions[key]):
            geometry_index = int(val_indices[local_i])
            true_life = true[:, 1]
            pred_life = pred[:, 1]
            for band, threshold in all_bands():
                row = metric_row(true_life, pred_life, threshold)
                per_geom_rows.append({
                    "model": key,
                    "geometry_index": geometry_index,
                    "band": band,
                    "threshold": "all" if threshold is None else threshold,
                    **row,
                })

    write_csv(args.outdir / "pooled_loglife_metrics.csv", pooled_rows)
    write_csv(args.outdir / "per_geometry_loglife_metrics.csv", per_geom_rows)
    plot_pooled_metrics(pooled_rows, args.outdir)
    plot_geometry_deltas(per_geom_rows, args.outdir)
    summary = decision_text(pooled_rows, per_geom_rows, args.bootstrap, args.seed)
    (args.outdir / "decision_summary.txt").write_text(summary, encoding="utf-8")

    manifest = {
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
        "selection_rule": "Prioritize <2 MAE/RMSE, then <3, then pooled metrics; inspect paired geometry results.",
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + summary)
    print(f"Saved CSV tables, figures, manifest, and decision summary to: {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preset comparison failed: {exc}", file=sys.stderr)
        raise
