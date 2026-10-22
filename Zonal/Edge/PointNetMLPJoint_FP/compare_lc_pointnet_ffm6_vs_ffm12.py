"""Compare LC-PointNet Fourier-frequency variants on the Zonal/Edge validation split.

Place this file in:
    Zonal/Edge/PointNetMLPJoint_FP/

It compares the 6-frequency LC-PointNet checkpoint against the existing
12-frequency checkpoint, reconstructs the deterministic geometry-level validation
split (20%, seed 42), evaluates every boundary-node query, and writes evaluation
artifacts under lc_pointnet_ffm_comparison_results/.

The script performs evaluation only. It never trains models, modifies checkpoints,
or changes source data.
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
from pn_models import PointNetMLPJoint_FP
import Training_script as ts
from pn_models import PointNetMLPJoint_FP

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "Trained_models"
DEFAULT_OUTDIR = ROOT / "lc_pointnet_ffm_comparison_results"
THRESHOLDS: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)
MODEL_KEYS = ("LC-PointNet FFM6", "LC-PointNet FFM12")


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
            "Evaluate LC-PointNet FFM6 versus FFM12 on the deterministic "
            "geometry-level validation split."
        )
    )
    parser.add_argument(
        "--ffm6-ckpt",
        type=Path,
        default=None,
        help="LC-PointNet checkpoint trained with six encoder/query Fourier frequencies.",
    )
    parser.add_argument(
        "--ffm12-ckpt",
        type=Path,
        default=None,
        help="LC-PointNet checkpoint trained with twelve encoder/query Fourier frequencies.",
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


def _select_one(paths: List[Path], description: str, flag: str) -> Path:
    if len(paths) == 1:
        return paths[0]
    rendered = "\n  ".join(str(path) for path in paths) if paths else "<none>"
    raise RuntimeError(
        f"Expected exactly one {description} checkpoint in {CHECKPOINT_DIR}; found "
        f"{len(paths)}:\n  {rendered}\nUse {flag} to choose explicitly."
    )


def checkpoint_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Resolve the two checkpoints; explicit CLI paths are recommended."""
    all_pt = sorted(CHECKPOINT_DIR.glob("*.pt"))

    ffm6 = args.ffm6_ckpt
    if ffm6 is None:
        # Conservative discovery: use only names that explicitly advertise six frequencies.
        ffm6 = _select_one(
            [path for path in all_pt if "ffm6" in path.stem.lower() or "pos6" in path.stem.lower()],
            "LC-PointNet FFM6",
            "--ffm6-ckpt",
        )

    ffm12 = args.ffm12_ckpt
    if ffm12 is None:
        # Legacy primary checkpoint may be named pnmlp_*_12.pt, but provide a safe
        # auto-discovery attempt and ask for an explicit path if ambiguous.
        ffm12 = _select_one(
            [
                path for path in all_pt
                if "ffm12" in path.stem.lower()
                or "pos12" in path.stem.lower()
                or path.stem.lower().endswith("_12")
            ],
            "LC-PointNet FFM12",
            "--ffm12-ckpt",
        )

    if ffm6.resolve() == ffm12.resolve():
        raise RuntimeError("FFM6 and FFM12 checkpoint paths resolve to the same file.")

    for path in (ffm6, ffm12):
        if not path.is_file():
            raise FileNotFoundError(path)

    return {"LC-PointNet FFM6": ffm6, "LC-PointNet FFM12": ffm12}


def _checkpoint_target_names(checkpoint: Dict) -> List[str]:
    names = checkpoint.get("target_names")
    return list(names) if names is not None else list(ts.TARGET_NAMES)


def load_model(key: str, path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("arch")
    if not isinstance(arch, dict):
        raise RuntimeError(f"{path.name} has no usable 'arch' dictionary.")

    encoder_cfg = arch.get("encoder_cfg")
    if not isinstance(encoder_cfg, dict):
        raise RuntimeError(f"{path.name} has no usable arch['encoder_cfg'] dictionary.")

    out_dim = int(arch.get("out_dim", len(checkpoint.get("target_mean", []))))
    in_channels = int(encoder_cfg.get("inchannels", arch.get("in_channels", 0)))
    head_hidden = list(arch.get("head_hidden", [192, 192, 96]))
    latent_dim = int(encoder_cfg.get("latent_dim", 128))

    if in_channels != 0:
        raise RuntimeError(
            f"This evaluator is for the coordinate-only Zonal/Edge LC-PointNet, but "
            f"{path.name} declares encoder extra channels={in_channels}."
        )

    model = PointNetMLPJoint_FP(
        latent_dim=latent_dim,
        mlp_hidden=head_hidden,
        out_dim=out_dim,
        encoder_cfg=dict(encoder_cfg),
        in_channels=in_channels,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    target_names = _checkpoint_target_names(checkpoint)
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


def _freq_pair(model: LoadedModel) -> Tuple[int, int]:
    cfg = model.arch["encoder_cfg"]
    enc = int((cfg.get("posenc") or {}).get("n_freqs", 0))
    query = int((cfg.get("head_posenc") or {}).get("n_freqs", 0))
    return enc, query


def validate_frequency_control(ffm6: LoadedModel, ffm12: LoadedModel) -> None:
    """Verify the expected 6-versus-12 Fourier-frequency comparison."""
    cfg6 = dict(ffm6.arch["encoder_cfg"])
    cfg12 = dict(ffm12.arch["encoder_cfg"])

    freq6 = _freq_pair(ffm6)
    freq12 = _freq_pair(ffm12)
    if freq6 != (6, 6):
        raise RuntimeError(
            f"Selected FFM6 checkpoint does not use 6/6 encoder/query frequencies: {freq6}."
        )
    if freq12 != (12, 12):
        raise RuntimeError(
            f"Selected FFM12 checkpoint does not use 12/12 encoder/query frequencies: {freq12}."
        )

    # These must match for this to be a clean coordinate-bandwidth control.
    keys_expected_same = [
        "latent_dim",
        "pre_hidden",
        "sa_blocks",
        "gf_hidden",
        "norm",
        "num_groups",
        "pool",
        "head_norm",
        "head_dropout",
        "fp",
    ]
    mismatches = [
        f"{key}: FFM6={cfg6.get(key)!r}, FFM12={cfg12.get(key)!r}"
        for key in keys_expected_same
        if cfg6.get(key) != cfg12.get(key)
    ]

    if list(ffm6.arch.get("head_hidden", [])) != list(ffm12.arch.get("head_hidden", [])):
        mismatches.append(
            "head_hidden: "
            f"FFM6={ffm6.arch.get('head_hidden')!r}, "
            f"FFM12={ffm12.arch.get('head_hidden')!r}"
        )

    if int(ffm6.arch.get("out_dim", -1)) != int(ffm12.arch.get("out_dim", -1)):
        mismatches.append(
            f"out_dim: FFM6={ffm6.arch.get('out_dim')!r}, "
            f"FFM12={ffm12.arch.get('out_dim')!r}"
        )

    if mismatches:
        raise RuntimeError(
            "The selected LC-PointNet checkpoints differ in settings other than Fourier "
            "frequency. This is not a clean FFM sensitivity control:\n  "
            + "\n  ".join(mismatches)
        )

    print("\nLC-PointNet Fourier-frequency control verified:")
    print(
        f"  FFM6: encoder={freq6[0]}, query={freq6[1]}, "
        f"parameters={ffm6.parameter_count:,}"
    )
    print(
        f"  FFM12: encoder={freq12[0]}, query={freq12[1]}, "
        f"parameters={ffm12.parameter_count:,}"
    )


def validation_tensors() -> Tuple[List[torch.Tensor], List[int], Path]:
    h5_path = ROOT.parents[2] / "Data_gen" / "output" / ts.H5_FILENAME
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Validation HDF5 file was not found: {h5_path}. This script must be run "
            "from Zonal/Edge/PointNetMLPJoint_FP in the repository containing Data_gen/output/."
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

    fig.savefig(outdir / "pooled_loglife_metrics_lc_pointnet_ffm6_vs_ffm12.png", dpi=220)
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

        ffm6_values = []
        ffm12_values = []
        for entries in by_id.values():
            if all(key in entries for key in MODEL_KEYS):
                ffm6_values.append(entries["LC-PointNet FFM6"]["mae"])
                ffm12_values.append(entries["LC-PointNet FFM12"]["mae"])

        if ffm6_values:
            ffm6_a = np.asarray(ffm6_values)
            ffm12_a = np.asarray(ffm12_values)
            limit = max(np.max(ffm6_a), np.max(ffm12_a)) * 1.05
            ax.scatter(ffm6_a, ffm12_a, alpha=0.65, s=22)
            ax.plot([0, limit], [0, limit], "k--", linewidth=1)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_xlabel("LC-PointNet FFM6 per-geometry MAE [decades]")
            ax.set_ylabel("LC-PointNet FFM12 per-geometry MAE [decades]")
            ax.text(0.04, 0.94, f"n geometries = {len(ffm6_a)}", transform=ax.transAxes, va="top")
        else:
            ax.text(0.5, 0.5, "No geometries contain this subset", ha="center", va="center")

        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.savefig(outdir / "paired_geometry_low_life_mae_lc_pointnet_ffm6_vs_ffm12.png", dpi=220)
    plt.close(fig)


def decision_text(
    pooled_rows: List[Dict],
    per_geom_rows: List[Dict],
    n_bootstrap: int,
    seed: int,
) -> str:
    lines = [
        "LC-PointNet Fourier-frequency sensitivity summary",
        "=" * 49,
        "Comparison: LC-PointNet FFM6 versus LC-PointNet FFM12.",
        "The intended architecture change is encoder and query Fourier frequency only.",
        "Decision priority: log-life MAE/RMSE in true_loglife <2, then <3, then pooled error.",
        "This is a validation-split comparison, not an independent physical test.",
        "",
    ]

    pooled = {(row["model"], row["band"]): row for row in pooled_rows}
    for band in ("log_life_lt_2", "log_life_lt_3", "all"):
        ffm6 = pooled[("LC-PointNet FFM6", band)]
        ffm12 = pooled[("LC-PointNet FFM12", band)]
        lines.append(
            f"{band}: n={ffm6['n_nodes']}; "
            f"MAE FFM6={ffm6['mae']:.6f}, FFM12={ffm12['mae']:.6f}; "
            f"RMSE FFM6={ffm6['rmse']:.6f}, FFM12={ffm12['rmse']:.6f}."
        )

    lines.append("")
    for band in ("log_life_lt_2", "log_life_lt_3"):
        by_geom: Dict[int, Dict[str, Dict]] = {}
        for row in per_geom_rows:
            if row["band"] == band and np.isfinite(row["mae"]):
                by_geom.setdefault(int(row["geometry_index"]), {})[row["model"]] = row

        deltas = np.asarray(
            [
                entries["LC-PointNet FFM6"]["mae"] - entries["LC-PointNet FFM12"]["mae"]
                for entries in by_geom.values()
                if all(key in entries for key in MODEL_KEYS)
            ]
        )
        mean, lo, hi = bootstrap_ci(deltas, n_bootstrap, seed)
        wins = int(np.sum(deltas < 0)) if deltas.size else 0
        lines.append(
            f"{band}, paired geometry-level MAE delta (FFM6 - FFM12): "
            f"mean={mean:.6f}; 95% bootstrap CI=[{lo:.6f}, {hi:.6f}]; "
            f"FFM6 wins {wins}/{deltas.size} geometries."
        )

    low2_6 = pooled[("LC-PointNet FFM6", "log_life_lt_2")]
    low2_12 = pooled[("LC-PointNet FFM12", "log_life_lt_2")]
    low3_6 = pooled[("LC-PointNet FFM6", "log_life_lt_3")]
    low3_12 = pooled[("LC-PointNet FFM12", "log_life_lt_3")]

    if low2_6["n_nodes"] == 0:
        recommendation = "No <2 validation nodes exist; select using <3 and pooled metrics."
    elif low2_6["mae"] < low2_12["mae"] and low2_6["rmse"] < low2_12["rmse"]:
        recommendation = "FFM6 improves both <2 MAE and RMSE; it is preferred for this benchmark."
    elif low2_6["mae"] > low2_12["mae"] and low2_6["rmse"] > low2_12["rmse"]:
        recommendation = "FFM12 improves both <2 MAE and RMSE; retain the original 12-frequency model."
    elif low3_6["mae"] < low3_12["mae"] and low3_6["rmse"] < low3_12["rmse"]:
        recommendation = (
            "The <2 result is mixed, but FFM6 improves both <3 MAE and RMSE. "
            "Inspect per-geometry results before selecting the primary configuration."
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
    validate_frequency_control(loaded["LC-PointNet FFM6"], loaded["LC-PointNet FFM12"])

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

    write_csv(args.outdir / "pooled_loglife_metrics_lc_pointnet_ffm6_vs_ffm12.csv", pooled_rows)
    write_csv(args.outdir / "per_geometry_loglife_metrics_lc_pointnet_ffm6_vs_ffm12.csv", per_geom_rows)
    plot_pooled_metrics(pooled_rows, args.outdir)
    plot_geometry_deltas(per_geom_rows, args.outdir)

    summary = decision_text(pooled_rows, per_geom_rows, args.bootstrap, args.seed)
    (args.outdir / "decision_summary_lc_pointnet_ffm6_vs_ffm12.txt").write_text(
        summary,
        encoding="utf-8",
    )

    manifest = {
        "comparison_type": "LC-PointNet Fourier-frequency sensitivity: FFM6 vs FFM12",
        "hdf5_path": str(h5_path),
        "split": {"test_size": 0.2, "random_state": 42, "n_validation_geometries": len(val_tensors)},
        "checkpoints": {
            key: {
                "path": str(loaded[key].path),
                "parameter_count": loaded[key].parameter_count,
                "encoder_cfg": loaded[key].arch["encoder_cfg"],
                "head_hidden": loaded[key].arch.get("head_hidden"),
            }
            for key in MODEL_KEYS
        },
        "thresholds": list(THRESHOLDS),
        "selection_rule": (
            "Prioritize <2 MAE/RMSE, then <3, then pooled metrics; "
            "verify that only Fourier frequencies changed."
        ),
    }
    (args.outdir / "run_manifest_lc_pointnet_ffm6_vs_ffm12.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + summary)
    print(f"Saved CSV tables, figures, manifest, and decision summary to: {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"LC-PointNet FFM comparison failed: {exc}", file=sys.stderr)
        raise
