"""Analysis/plotting helpers for model_ablation_comparison.ipynb.

This module contains no model-loading, checkpoint, or torch inference code —
that logic stays in the notebook for auditability. Everything here operates on
plain pandas/numpy data produced by the notebook's inference loop.

Expected "node_results" DataFrame schema (one row per evaluated query node):
    regime, ablation, model_family, sample_key, sample_id, node_idx,
    x_mm, r_mm, zone_id, zone_name, subzone_id, subzone_name, arc_length_mm,
    true_stress, pred_stress, true_loglife, pred_loglife
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ZONE_ID_TO_NAME: Dict[int, str] = {
    0: "bore", 1: "lower_transition", 2: "web", 3: "upper_transition", 4: "rim",
}
SUBZONE_ID_TO_NAME: Dict[int, str] = {
    0: "bore", 1: "lower_transition", 2: "web", 3: "upper_transition", 4: "rim_main",
    5: "front_face", 6: "front_cgroove", 7: "rear_arm_neck", 8: "rear_arm_land",
    9: "rear_arm_corner", 10: "rear_arm_end_face",
}
GROUPED_REGION_DEFS: Dict[str, List[str]] = {
    "Critical lower transition": ["lower_transition"],
    "Rim features": ["rim_main", "front_cgroove", "rear_arm_neck", "rear_arm_land", "rear_arm_corner"],
    "Remaining contour": ["bore", "web", "upper_transition"],
}

MIN_BIN_NODES = 30  # below this a bin's aggregate is flagged unstable
MIN_ZONE_NODES = 20

GROUP_COLS = ["regime", "ablation", "model_family"]

# ---------------------------------------------------------------------------
# Structured publication-model identity
# ---------------------------------------------------------------------------
#
# Internal model-family identifiers (and their checkpoint/hash aliases) are
# intentionally kept separate from the names used in paper figures and
# notebook display tables. Rather than a flat string-replacement table, each
# verified internal id is decomposed into its architecture, engineered-
# geometric-feature status, and training condition, then formatted centrally
# by ``format_publication_model_label``. This keeps the mapping auditable
# (one row per verified identity, see class docstring for evidence) and
# prevents ad-hoc labels from being typed by hand at plotting call sites.


@dataclass(frozen=True)
class PublicationModelIdentity:
    """Verified composition of a paper-facing model label.

    Each field must be established from the model's own checkpoint/config
    metadata (see ``resolve_publication_label`` docstring), never guessed
    from the raw internal id alone.
    """

    architecture: Literal["GC-PointNet", "LC-PointNet", "ArGEnT-A", "GINOT-A"]
    has_geometric_features: bool = False
    training_condition: Literal["standard", "weighted_loss"] = "standard"
    supervision: Literal["joint_stress_life", "life_only", "not_applicable"] = "not_applicable"


def format_publication_model_label(identity: PublicationModelIdentity) -> str:
    """Format a verified :class:`PublicationModelIdentity` into paper text."""
    label = identity.architecture
    if identity.has_geometric_features:
        label += " + GF"
    if identity.training_condition == "weighted_loss":
        label += " (weighted loss)"
    return label


# Verified internal model-family id -> structured publication identity.
#
# Evidence for each entry (see notebook ``reconstruct``/``predict_dispatch``
# helpers in 01_fp_vs_argent.ipynb, 02_engineered_geometric_features.ipynb,
# 03_data_efficiency.ipynb, 04_joint_stress_supervision.ipynb):
#   - "PointNetMLPJoint*" families are all built via ``pn.build_model_from_arch``
#     (non-FP) or ``pn.build_fp_model_from_arch`` (FP), i.e. GC-PointNet vs
#     LC-PointNet respectively; the ``_headfeat`` suffix is the only thing
#     that switches on engineered-geometric-feature consumption.
#   - "PointNetMLPJoint_weighted" is loaded with the exact same
#     ``pn.build_model_from_arch`` path (no FP propagation, no engineered
#     geometric features) as plain "PointNetMLPJoint" -- see the shared
#     ``elif family in ('PointNetMLPJoint', 'PointNetMLPJoint_weighted', ...)``
#     branch in every notebook's ``reconstruct``/model-build cell -- and only
#     differs in its (weighted) training loss. It is therefore the GC-PointNet
#     architecture trained with the weighted-loss ablation.
#   - "ArGEnT_self_att_noSDF" is the ArGEnT-A architecture; its engineered-
#     geometric-feature status is ablation-directory dependent (the same
#     family id is reused for both the plain and GF-augmented checkpoints),
#     so it is resolved via the ``geometric_features`` override argument
#     rather than a distinct family id.
MODEL_IDENTITIES: Dict[str, PublicationModelIdentity] = {
    "PointNetMLPJoint": PublicationModelIdentity("GC-PointNet"),
    "PointNetMLPJoint_headfeat": PublicationModelIdentity("GC-PointNet", has_geometric_features=True),
    "PointNetMLPJoint_FP": PublicationModelIdentity("LC-PointNet"),
    "PointNetMLPJoint_FP_headfeat": PublicationModelIdentity("LC-PointNet", has_geometric_features=True),
    "PointNetMLPJoint_weighted": PublicationModelIdentity("GC-PointNet", training_condition="weighted_loss"),
    "ArGEnT_self_att_noSDF": PublicationModelIdentity("ArGEnT-A"),
    "GINOT": PublicationModelIdentity("GINOT-A"),
    "GINOT-A": PublicationModelIdentity("GINOT-A"),
    "GINOT_A": PublicationModelIdentity("GINOT-A"),
}

# Derived flat id -> label table, kept for readability at call sites that
# only need the resolved string (e.g. error messages, ordering).
DISPLAY_MODEL_NAMES: Dict[str, str] = {
    model_id: format_publication_model_label(identity) for model_id, identity in MODEL_IDENTITIES.items()
}
DISPLAY_MODEL_ORDER: List[str] = [
    "GC-PointNet",
    "GC-PointNet + GF",
    "GC-PointNet (weighted loss)",
    "LC-PointNet",
    "LC-PointNet + GF",
    "LC-PointNet (weighted loss)",
    "ArGEnT-A",
    "ArGEnT-A + GF",
    "GINOT-A",
    "GINOT-A + GF",
]


class UnresolvedPublicationLabelError(ValueError):
    """Raised when a plotted model has no verified paper-facing label.

    This is intentionally a hard failure: silently falling back to a raw
    ``model_family``/checkpoint identifier in a paper-facing figure is the
    exact failure mode this module exists to prevent (see
    ``resolve_publication_label`` docstring).
    """


def canonicalize_model_family(
    model_family: Any,
    *,
    model_class: Optional[Any] = None,
    model_config_identity: Optional[Any] = None,
) -> str:
    """Canonicalize verified legacy model identifiers used in checkpoints.

    This keeps notebook result tables on a single internal identifier even when
    older folder names or metadata aliases appear during discovery.
    """
    family = None if model_family is None else str(model_family)
    model_class = None if model_class is None else str(model_class)
    model_config_identity = None if model_config_identity is None else str(model_config_identity)
    if family in {"GINOT", "GINOT-A", "GINOT_A"}:
        return "GINOT-A"
    if model_class == "GINOT_A" or model_config_identity == "GINOT_A":
        return "GINOT-A"
    return "" if family is None else family


def resolve_publication_label(
    model_internal_id: str,
    *,
    geometric_features: Optional[bool] = None,
    checkpoint_id: Optional[str] = None,
) -> str:
    """Single authoritative internal-id -> paper-facing label resolver.

    Every paper-facing plotting function/notebook cell must call this
    resolver (directly or via :func:`display_model_name`) instead of typing
    a label by hand or falling back to a raw identifier. It never silently
    returns a raw id: an unresolved identity raises
    :class:`UnresolvedPublicationLabelError` and stops figure generation.

    Parameters
    ----------
    model_internal_id:
        The verified, unmodified ``model_family`` value from the results
        DataFrame/checkpoint metadata (never renamed on disk).
    geometric_features:
        Set ``True``/``False`` only when the caller has verified, from the
        loaded checkpoint's own configuration (e.g. ``extra_feat_cols`` /
        ``Training_script.py`` ``INPUT_COLS``), whether *this specific*
        checkpoint consumes engineered geometric features. This is required
        for ``ArGEnT_self_att_noSDF``, whose engineered-feature status is
        ablation-directory dependent (the same family id is used for both
        the plain and GF-augmented ArGEnT-A checkpoints). It is ignored for
        the PointNet families, whose GF status is already encoded in the
        family id itself (``*_headfeat`` suffix).
    checkpoint_id:
        Optional checkpoint path/identifier, included only in the error
        message to help diagnose an unresolved mapping.
    """
    identity = MODEL_IDENTITIES.get(str(model_internal_id))
    if identity is None:
        raise UnresolvedPublicationLabelError(
            "Unresolved publication label for a plotted model. "
            f"model_internal_id={model_internal_id!r}, checkpoint_id={checkpoint_id!r}. "
            f"Known verified identities: {sorted(MODEL_IDENTITIES)}. "
            "Add a verified PublicationModelIdentity to MODEL_IDENTITIES instead of "
            "falling back to the raw identifier in a paper-facing figure."
        )
    if str(model_internal_id) == "ArGEnT_self_att_noSDF" and geometric_features:
        identity = PublicationModelIdentity(
            identity.architecture,
            has_geometric_features=True,
            training_condition=identity.training_condition,
            supervision=identity.supervision,
        )
    return format_publication_model_label(identity)


def display_model_name(model_family: str) -> str:
    """Strict paper-facing name resolver (thin wrapper over
    :func:`resolve_publication_label`) that never changes an internal
    identifier and never silently falls back to it either.
    """
    return resolve_publication_label(model_family)


def ordered_model_families(families: Sequence[str]) -> List[str]:
    """Order internal identifiers by their paper-facing model names."""
    return sorted(
        families,
        key=lambda family: (
            DISPLAY_MODEL_ORDER.index(display_model_name(family))
            if display_model_name(family) in DISPLAY_MODEL_ORDER
            else len(DISPLAY_MODEL_ORDER),
            display_model_name(family),
        ),
    )


# Alias kept for readability at call sites that only need publication
# ordering (identical behaviour to ``ordered_model_families``).
apply_publication_model_order = ordered_model_families


SUPERVISION_MODE_LABELS: Dict[str, str] = {
    "joint": "joint stress–life",
    "life_only": "life only",
}


def format_supervision_label(publication_label: str, stress_supervision_mode: str) -> str:
    """Append a verified supervision condition to a publication model label.

    ``stress_supervision_mode`` must be ``"joint"`` (joint stress + log-life
    supervision) or ``"life_only"`` (log-life-only supervision). Never pass
    a checkpoint id or raw folder name here.
    """
    if stress_supervision_mode not in SUPERVISION_MODE_LABELS:
        raise ValueError(
            f"Unknown stress_supervision_mode={stress_supervision_mode!r}; "
            f"expected one of {sorted(SUPERVISION_MODE_LABELS)}"
        )
    return f"{publication_label} ({SUPERVISION_MODE_LABELS[stress_supervision_mode]})"


def format_gf_delta_label(gf_publication_label: str, base_publication_label: str) -> str:
    """Build an explicit ``"<+GF label> − <base label>"`` delta label.

    Use this for every engineered-geometric-feature delta plot instead of
    typing ``headfeat − baseline``/``EF − baseline`` by hand.
    """
    return f"{gf_publication_label} \u2212 {base_publication_label}"


def presentation_table(df: pd.DataFrame, label_fn: Optional[Any] = None) -> pd.DataFrame:
    """Copy a result table and substitute display names only for notebook output.

    ``label_fn`` defaults to the module-level :func:`display_model_name`. Pass a
    notebook-local resolver when a raw ``model_family`` id is genuinely ambiguous
    across notebooks (for example an ArGEnT checkpoint that is feature-augmented
    in one ablation directory but not in another) so the correct paper-facing
    label (e.g. ``ArGEnT-A + GF``) is used for that notebook's rendered tables.
    """
    resolver = label_fn if label_fn is not None else display_model_name
    display_df = df.copy()
    for column in ("model_family", "left_family", "right_family",
                   "model_family_baseline", "model_family_ablation"):
        if column in display_df:
            display_df[column] = display_df[column].map(resolver)
    return display_df


# ---------------------------------------------------------------------------
# Pooled metrics (reproduces the notebook's original pooled-metric behaviour)
# ---------------------------------------------------------------------------

def _mse(t, p):
    t, p = np.asarray(t, np.float64), np.asarray(p, np.float64)
    return float(np.mean((t - p) ** 2))


def _rmse(t, p):
    return float(np.sqrt(_mse(t, p)))


def _mae(t, p):
    t, p = np.asarray(t, np.float64), np.asarray(p, np.float64)
    return float(np.mean(np.abs(t - p)))


def _r2(t, p):
    t, p = np.asarray(t, np.float64), np.asarray(p, np.float64)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan


def loglife_to_raw_life(loglife) -> np.ndarray:
    x = np.asarray(loglife, dtype=np.float64)
    return np.power(10.0, np.clip(x, -30.0, 30.0))


def percentage_metrics(y_true_raw, y_pred_raw) -> Dict[str, float]:
    y_true_raw = np.asarray(y_true_raw, np.float64)
    y_pred_raw = np.asarray(y_pred_raw, np.float64)
    denom = np.clip(y_true_raw, 1e-12, None)
    pe = (y_pred_raw - y_true_raw) / denom * 100.0
    ape = np.abs(pe)
    return {"MAPE (%)": float(np.mean(ape)), "MPE (%)": float(np.mean(pe)), "Max_PE (%)": float(np.max(ape))}


def pooled_metrics_from_nodes(node_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled Stress / LogLife metrics per (regime, ablation, model_family).

    Raw-life MAPE/MPE/Max_PE are retained as SECONDARY diagnostics only.
    """
    if node_df.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for keys, g in node_df.groupby(GROUP_COLS):
        regime, ablation, model_family = keys
        for target, tcol, pcol in [("Stress", "true_stress", "pred_stress"), ("LogLife", "true_loglife", "pred_loglife")]:
            t, p = g[tcol].to_numpy(), g[pcol].to_numpy()
            rec: Dict[str, Any] = {
                "regime": regime, "ablation": ablation, "model_family": model_family, "target": target,
                "MSE": _mse(t, p), "RMSE": _rmse(t, p), "MAE": _mae(t, p), "R2 (log)": _r2(t, p),
                "MAPE (%)": np.nan, "MPE (%)": np.nan, "Max_PE (%)": np.nan,
                "RMSE_raw": np.nan, "MSE_raw_life": np.nan, "R2_raw_life": np.nan,
            }
            if target == "LogLife":
                t_raw, p_raw = loglife_to_raw_life(t), loglife_to_raw_life(p)
                raw_mse = _mse(t_raw, p_raw)
                rec.update(percentage_metrics(t_raw, p_raw))
                rec["MSE_raw_life"] = raw_mse
                rec["RMSE_raw"] = float(np.sqrt(raw_mse))
                rec["R2_raw_life"] = _r2(t_raw, p_raw)
            records.append(rec)
    return pd.DataFrame(records).sort_values(GROUP_COLS + ["target"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LogLife criticality bins
# ---------------------------------------------------------------------------

FULL_TEST_SET_LABEL = "Full test set"
BIN_DEFS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("log_life < 2", None, 2.0),
    ("2 <= log_life < 3", 2.0, 3.0),
    ("3 <= log_life < 4", 3.0, 4.0),
    ("4 <= log_life < 6", 4.0, 6.0),
    ("log_life >= 6", 6.0, None),
]
PHYSICAL_LIFE_BIN_DEFS: List[Tuple[str, Optional[float], Optional[float]]] = list(BIN_DEFS)
PHYSICAL_LIFE_BIN_ORDER: List[str] = [label for label, _, _ in PHYSICAL_LIFE_BIN_DEFS]
ALL_LIFE_BIN_DEFS: List[Tuple[str, Optional[float], Optional[float]]] = [
    (FULL_TEST_SET_LABEL, None, None),
    *PHYSICAL_LIFE_BIN_DEFS,
]


def life_bin_metrics_by_groups(
    node_df: pd.DataFrame,
    group_cols: Sequence[str],
    include_full_set: bool = True,
) -> pd.DataFrame:
    if node_df.empty:
        return pd.DataFrame(columns=list(group_cols) + ["life_bin", "n_samples", "mae_loglife", "rmse_loglife", "unstable"])
    bin_defs: List[Tuple[str, Optional[float], Optional[float]]] = (
        ALL_LIFE_BIN_DEFS if include_full_set else PHYSICAL_LIFE_BIN_DEFS
    )
    records: List[Dict[str, Any]] = []
    for keys, g in node_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        t_all = g["true_loglife"].to_numpy()
        p_all = g["pred_loglife"].to_numpy()
        for bin_name, lo, hi in bin_defs:
            mask = np.ones_like(t_all, dtype=bool)
            if lo is not None:
                mask &= (t_all >= lo)
            if hi is not None:
                mask &= (t_all < hi)
            n = int(mask.sum())
            err = p_all[mask] - t_all[mask]
            unstable = n < MIN_BIN_NODES
            mae_loglife = float(np.mean(np.abs(err))) if n else np.nan
            rmse_loglife = float(np.sqrt(np.mean(err ** 2))) if n else np.nan
            signed_mean_error_loglife = float(np.mean(err)) if n else np.nan
            median_abs_error_loglife = float(np.median(np.abs(err))) if n else np.nan
            records.append({
                **base,
                "life_bin": bin_name, "bin": bin_name,
                "n_samples": n, "n_nodes": n,
                "mae_loglife": mae_loglife,
                "rmse_loglife": rmse_loglife,
                "signed_mean_error_loglife": signed_mean_error_loglife,
                "median_abs_error_loglife": median_abs_error_loglife,
                "MAE": mae_loglife,
                "RMSE": rmse_loglife,
                "signed_mean_error": signed_mean_error_loglife,
                "median_abs_error": median_abs_error_loglife,
                "unstable": bool(unstable),
            })
    return pd.DataFrame(records)


def loglife_bin_metrics(node_df: pd.DataFrame) -> pd.DataFrame:
    return life_bin_metrics_by_groups(node_df, GROUP_COLS, include_full_set=True)


def split_life_bin_metrics(
    metrics_df: pd.DataFrame,
    bin_col: str = "life_bin",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty or bin_col not in metrics_df.columns:
        return metrics_df.iloc[0:0].copy(), metrics_df.iloc[0:0].copy()
    per_bin_df = metrics_df[metrics_df[bin_col].isin(PHYSICAL_LIFE_BIN_ORDER)].copy()
    full_set_df = metrics_df[metrics_df[bin_col].eq(FULL_TEST_SET_LABEL)].copy()
    if not per_bin_df.empty:
        per_bin_df[bin_col] = pd.Categorical(
            per_bin_df[bin_col],
            categories=PHYSICAL_LIFE_BIN_ORDER,
            ordered=True,
        )
    return per_bin_df, full_set_df


def assert_non_empty_plot_df(
    plot_df: pd.DataFrame,
    *,
    intended_plot: str,
    reference_df: Optional[pd.DataFrame] = None,
    metric_cols: Sequence[str] = ("mae_loglife", "rmse_loglife"),
) -> None:
    if not plot_df.empty:
        return
    source = reference_df if reference_df is not None else plot_df
    model_variants = sorted(source["model_family"].dropna().astype(str).unique().tolist()) if "model_family" in source.columns else []
    training_fracs = []
    if "training_fraction" in source.columns:
        training_fracs = sorted(pd.to_numeric(source["training_fraction"], errors="coerce").dropna().unique().tolist())
    life_bins = sorted(source["life_bin"].dropna().astype(str).unique().tolist()) if "life_bin" in source.columns else []
    counts = (
        source.groupby(["life_bin", "training_fraction"]).size().to_dict()
        if all(c in source.columns for c in ["life_bin", "training_fraction"]) else {}
    )
    non_null = {c: int(source[c].notna().sum()) for c in metric_cols if c in source.columns}
    raise ValueError(
        f"No rows available for {intended_plot}. "
        f"Report available model variants={model_variants}, "
        f"training fractions={training_fracs}, life bins={life_bins}, "
        f"row counts by life bin and training fraction={counts}, "
        f"non-null metrics={non_null}."
    )


# ---------------------------------------------------------------------------
# Zone / subzone metrics
# ---------------------------------------------------------------------------

def zone_metrics_from_nodes(node_df: pd.DataFrame, label_col: str = "subzone_name",
                             required_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Compatibility wrapper that now returns grouped physical-region metrics.

    `label_col` and `required_labels` are ignored to preserve old call sites.
    """
    _ = label_col, required_labels
    return grouped_region_metrics_from_nodes(node_df)


def grouped_region_metrics_from_nodes(node_df: pd.DataFrame) -> pd.DataFrame:
    if node_df.empty:
        return pd.DataFrame()
    subzone_to_group = {
        subzone: group
        for group, subzones in GROUPED_REGION_DEFS.items()
        for subzone in subzones
    }
    node_df = node_df.copy()
    node_df["grouped_region"] = node_df["subzone_name"].map(subzone_to_group)
    records: List[Dict[str, Any]] = []
    for keys, g in node_df.groupby(GROUP_COLS):
        regime, ablation, model_family = keys
        for label, subzones in GROUPED_REGION_DEFS.items():
            sub = g[g["subzone_name"].isin(subzones)]
            if sub.empty:
                records.append({
                    "regime": regime, "ablation": ablation, "model_family": model_family,
                    "grouped_region": label, "n_nodes": 0,
                    "LogLife_MAE": np.nan, "LogLife_RMSE": np.nan,
                    "signed_mean_LogLife_error": np.nan, "median_abs_LogLife_error": np.nan,
                    "Stress_MAE": np.nan, "Stress_RMSE": np.nan, "status": "missing",
                })
                continue
            tl, pl = sub["true_loglife"].to_numpy(), sub["pred_loglife"].to_numpy()
            ts, ps = sub["true_stress"].to_numpy(), sub["pred_stress"].to_numpy()
            err_l = pl - tl
            finite_stress = np.isfinite(ts) & np.isfinite(ps)
            records.append({
                "regime": regime, "ablation": ablation, "model_family": model_family,
                "grouped_region": label, "n_nodes": int(len(sub)),
                "LogLife_MAE": _mae(tl, pl), "LogLife_RMSE": _rmse(tl, pl),
                "signed_mean_LogLife_error": float(np.mean(err_l)),
                "median_abs_LogLife_error": float(np.median(np.abs(err_l))),
                "Stress_MAE": _mae(ts[finite_stress], ps[finite_stress]) if finite_stress.any() else np.nan,
                "Stress_RMSE": _rmse(ts[finite_stress], ps[finite_stress]) if finite_stress.any() else np.nan,
                "status": "ok" if len(sub) >= MIN_ZONE_NODES else "unstable_low_count",
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Geometry-level (per-sample) metrics
# ---------------------------------------------------------------------------

def geometry_level_metrics(node_df: pd.DataFrame) -> pd.DataFrame:
    if node_df.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for keys, g in node_df.groupby(GROUP_COLS + ["sample_id"]):
        regime, ablation, model_family, sample_id = keys
        g = g.reset_index(drop=True)
        true_life = g["true_loglife"].to_numpy()
        pred_life = g["pred_loglife"].to_numpy()
        true_stress = g["true_stress"].to_numpy()
        pred_stress = g["pred_stress"].to_numpy()

        true_crit_idx = int(np.argmin(true_life))
        pred_crit_idx = int(np.argmin(pred_life))
        true_max_s_idx = int(np.argmax(true_stress))
        pred_max_s_idx = int(np.argmax(pred_stress))

        dx = g["x_mm"].iloc[pred_crit_idx] - g["x_mm"].iloc[true_crit_idx]
        dr = g["r_mm"].iloc[pred_crit_idx] - g["r_mm"].iloc[true_crit_idx]
        crit_dist_mm = float(np.hypot(dx, dr))

        same_zone = bool(g["zone_id"].iloc[pred_crit_idx] == g["zone_id"].iloc[true_crit_idx])
        same_subzone = bool(g["subzone_id"].iloc[pred_crit_idx] == g["subzone_id"].iloc[true_crit_idx]) \
            if g["subzone_id"].notna().any() else None

        min_life_err = float(pred_life[true_crit_idx] - true_life[true_crit_idx])
        records.append({
            "regime": regime, "ablation": ablation, "model_family": model_family, "sample_id": sample_id,
            "whole_geometry_loglife_mae": _mae(true_life, pred_life),
            "whole_geometry_stress_mae": _mae(true_stress, pred_stress),
            "true_min_loglife": float(true_life[true_crit_idx]),
            "pred_min_loglife": float(pred_life[pred_crit_idx]),
            "min_loglife_error_decades": min_life_err,
            "abs_min_loglife_error_decades": float(abs(min_life_err)),
            "true_crit_x_mm": float(g["x_mm"].iloc[true_crit_idx]), "true_crit_r_mm": float(g["r_mm"].iloc[true_crit_idx]),
            "pred_crit_x_mm": float(g["x_mm"].iloc[pred_crit_idx]), "pred_crit_r_mm": float(g["r_mm"].iloc[pred_crit_idx]),
            "crit_node_distance_mm": crit_dist_mm,
            "true_max_stress": float(true_stress[true_max_s_idx]),
            "pred_max_stress": float(pred_stress[pred_max_s_idx]),
            "max_stress_error": float(pred_stress[pred_max_s_idx] - true_stress[true_max_s_idx]),
            "abs_max_stress_error": float(abs(pred_stress[pred_max_s_idx] - true_stress[true_max_s_idx])),
            "same_zone_critical": same_zone,
            "same_subzone_critical": same_subzone,
            "true_crit_zone": ZONE_ID_TO_NAME.get(int(g["zone_id"].iloc[true_crit_idx])),
            "true_crit_subzone": g["subzone_name"].iloc[true_crit_idx] if g["subzone_id"].notna().any() else None,
        })
    return pd.DataFrame(records)


def geometry_metrics_summary(geom_df: pd.DataFrame) -> pd.DataFrame:
    """Median / p90 / worst-case summary across geometries, per (regime, ablation, model)."""
    if geom_df.empty:
        return pd.DataFrame()
    cols = ["min_loglife_error_decades", "crit_node_distance_mm", "max_stress_error"]
    records = []
    for keys, g in geom_df.groupby(GROUP_COLS):
        regime, ablation, model_family = keys
        rec = {"regime": regime, "ablation": ablation, "model_family": model_family, "n_geometries": len(g)}
        for c in cols:
            vals = g[c].abs().to_numpy()
            rec[f"{c}_median_abs"] = float(np.median(vals))
            rec[f"{c}_p90_abs"] = float(np.percentile(vals, 90))
            rec[f"{c}_worst_abs"] = float(np.max(vals))
        rec["frac_same_zone_critical"] = float(g["same_zone_critical"].mean())
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Representative geometry selection
# ---------------------------------------------------------------------------

def select_representative_geometries(geom_df: pd.DataFrame, regime: str, ablation: str,
                                      argent_family: str = "ArGEnT_self_att_noSDF",
                                      disagreement_family: str = "PointNetMLPJoint_FP") -> Dict[str, Any]:
    """Deterministically pick median / critical-life / model-disagreement sample_ids.

    Uses only geometries evaluated for all requested model families (shared IDs).
    """
    sub = geom_df[(geom_df["regime"] == regime) & (geom_df["ablation"] == ablation)]
    if sub.empty:
        return {}
    families = sorted(sub["model_family"].unique().tolist())
    shared_ids = set(sub["sample_id"].unique())
    for fam in families:
        shared_ids &= set(sub.loc[sub["model_family"] == fam, "sample_id"].unique())
    shared_ids = sorted(shared_ids)
    if not shared_ids:
        return {}
    sub = sub[sub["sample_id"].isin(shared_ids)]

    # Composite joint error per geometry, averaged across model families present.
    per_geom = sub.groupby("sample_id").apply(
        lambda g: float(np.mean(np.abs(g["min_loglife_error_decades"]).to_numpy()
                                 / max(np.abs(g["min_loglife_error_decades"]).to_numpy().std() or 1.0, 1e-6)
                                 + np.abs(g["max_stress_error"]).to_numpy()
                                 / max(np.abs(g["max_stress_error"]).to_numpy().std() or 1.0, 1e-6)))
    )
    median_val = per_geom.median()
    median_id = int((per_geom - median_val).abs().idxmin())

    true_min_by_geom = sub.groupby("sample_id")["true_min_loglife"].first()
    critical_id = int(true_min_by_geom.idxmin())

    disagreement_id = None
    if argent_family in families and disagreement_family in families:
        a = sub[sub["model_family"] == argent_family].set_index("sample_id")["min_loglife_error_decades"].abs()
        f = sub[sub["model_family"] == disagreement_family].set_index("sample_id")["min_loglife_error_decades"].abs()
        common = a.index.intersection(f.index)
        if len(common):
            diff = (a.loc[common] - f.loc[common]).abs()
            disagreement_id = int(diff.idxmax())

    return {"median": median_id, "critical_life": critical_id, "model_disagreement": disagreement_id}


def paired_argent_fp_diff(geom_df: pd.DataFrame,
                           argent_family: str = "ArGEnT_self_att_noSDF",
                           fp_family: str = "PointNetMLPJoint_FP") -> pd.DataFrame:
    """Per-geometry ArGEnT-minus-FP |min-LogLife error| difference.

    Sign convention: positive => FP has smaller absolute critical-life error than ArGEnT
    (i.e. FP is better for that geometry). Negative => ArGEnT is better.
    """
    records = []
    for (regime, ablation), g in geom_df.groupby(["regime", "ablation"]):
        a = g[g["model_family"] == argent_family].set_index("sample_id")["min_loglife_error_decades"].abs()
        f = g[g["model_family"] == fp_family].set_index("sample_id")["min_loglife_error_decades"].abs()
        common = a.index.intersection(f.index)
        for sid in common:
            records.append({
                "regime": regime, "ablation": ablation, "sample_id": sid,
                "argent_abs_error": float(a.loc[sid]), "fp_abs_error": float(f.loc[sid]),
                "argent_minus_fp": float(a.loc[sid] - f.loc[sid]),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Raw tokens that must never survive into a paper-facing figure. Matched
# case-insensitively as substrings, except ``EF`` which is matched as a
# case-sensitive standalone token (see ``_EF_TOKEN_PATTERN``) so that it does
# not spuriously fire on ordinary English words.
PROHIBITED_PUBLICATION_TOKENS: List[str] = [
    "pnmlp_",
    "argent_self_",
    "pointnetmlpjoint_fp_headfeat",
    "pointnetmlpjoint_headfeat",
    "pointnetmlpjoint_fp",
    "pointnetmlpjoint",
    "pointnetfp",
    "headfeat",
    "adapted argent-inspired attention operator",
    "argent_self_att_nosdf",
    "with stress",
    "no stress",
    "with-stress",
    "no-stress",
    "baseline",
    "regular",
    "_arc_feat",
]
_EF_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])EF(?![A-Za-z])")


def _iter_figure_text(fig) -> List[Tuple[str, str]]:
    """Collect ``(text, location)`` pairs for every visible text element."""
    found: List[Tuple[str, str]] = []

    def _add(value: Optional[str], location: str) -> None:
        if value:
            found.append((str(value), location))

    suptitle = getattr(fig, "_suptitle", None)
    _add(suptitle.get_text() if suptitle is not None else None, "figure.suptitle")

    for ax_idx, ax in enumerate(fig.get_axes()):
        _add(ax.get_title(), f"axes[{ax_idx}].title")
        _add(ax.get_xlabel(), f"axes[{ax_idx}].xlabel")
        _add(ax.get_ylabel(), f"axes[{ax_idx}].ylabel")
        for tick in ax.get_xticklabels():
            _add(tick.get_text(), f"axes[{ax_idx}].xticklabel")
        for tick in ax.get_yticklabels():
            _add(tick.get_text(), f"axes[{ax_idx}].yticklabel")
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                _add(text.get_text(), f"axes[{ax_idx}].legend")
        for text in ax.texts:
            _add(text.get_text(), f"axes[{ax_idx}].annotation")

    for leg_idx, legend in enumerate(getattr(fig, "legends", []) or []):
        for text in legend.get_texts():
            _add(text.get_text(), f"figure.legend[{leg_idx}]")

    return found


def validate_no_raw_publication_labels(fig, *, context: str = "") -> None:
    """Fail figure generation if any visible text contains a raw/internal token.

    Inspects the figure suptitle, every axes title/x-y label/tick label/
    legend/annotation, and any figure-level legends. Raises ``ValueError``
    (rather than silently saving) so a figure can never reach the paper with
    a checkpoint hash, ``headfeat``, ``EF``, ``with/no stress``, etc.
    """
    offenders: List[Tuple[str, str, str]] = []
    for text, location in _iter_figure_text(fig):
        lowered = text.lower()
        for token in PROHIBITED_PUBLICATION_TOKENS:
            if token in lowered:
                offenders.append((text, location, token))
        if _EF_TOKEN_PATTERN.search(text):
            offenders.append((text, location, "EF"))
    if offenders:
        header = "Prohibited raw token found in paper-facing figure"
        if context:
            header += f" ({context})"
        lines = [header + ":"]
        for text, location, token in offenders:
            lines.append(f"  - text={text!r} location={location} matched_token={token!r}")
        lines.append(
            "Use resolve_publication_label()/format_supervision_label()/"
            "format_gf_delta_label() to build reader-facing text instead."
        )
        raise ValueError("\n".join(lines))


def make_wrapped_life_bin_grid(
    n_panels: int,
    ncols: int = 2,
    panel_width: float = 6.5,
    panel_height: float = 4.5,
    **subplot_kwargs: Any,
):
    """Create a compact ``ncols``-wide wrapped grid for per-life-bin panels.

    Unused trailing axes (when ``n_panels`` does not evenly fill the grid)
    are hidden, not left visually empty. Panels must be filled in reading
    order (left to right, then top to bottom) by the caller so ordered
    physical life bins stay ordered low-to-high log-life.

    Returns ``(fig, active_axes, all_axes_flat)`` where ``active_axes`` has
    exactly ``n_panels`` entries to plot into and ``all_axes_flat`` includes
    the hidden ones (useful for ``add_external_figure_legend``, which only
    reads legend handles from currently-visible axes anyway).
    """
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        **subplot_kwargs,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes_flat[:n_panels], axes_flat


def add_external_figure_legend(
    fig,
    axes,
    *,
    title: Optional[str] = None,
    bbox_to_anchor: Tuple[float, float] = (1.01, 0.5),
    loc: str = "center left",
):
    """Add one deduplicated figure-level legend outside the plotting grid.

    Reads handles/labels from every visible axes, de-duplicates by label
    (so a series repeated across panels appears once), and places the
    legend outside the axes so it never overlaps the suptitle, a subplot
    title, or plotted data. Callers must reserve legend space, e.g. via
    ``fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.93))``.
    """
    axes_list = [ax for ax in np.atleast_1d(axes).ravel().tolist() if ax is not None and ax.get_visible()]
    handles: List[Any] = []
    labels: List[str] = []
    seen = set()
    for ax in axes_list:
        h_list, l_list = ax.get_legend_handles_labels()
        for handle, label in zip(h_list, l_list):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if not handles:
        return None
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=True, title=title)


def _save_fig(fig, out_dir: Path, name: str) -> None:
    validate_no_raw_publication_labels(fig, context=name)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    try:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    except Exception:
        pass


def robust_signed_limit(err_arrays: Sequence[np.ndarray], pct: float = 99.0, floor: float = 1e-6) -> float:
    all_abs = np.concatenate([np.abs(np.asarray(a)) for a in err_arrays if len(a)])
    if all_abs.size == 0:
        return floor
    return float(max(np.percentile(all_abs, pct), floor))


def plot_field_comparison(geom_nodes_by_model: Dict[str, pd.DataFrame], field_true_col: str, field_pred_col: str,
                           error_unit: str, title_prefix: str, fixed_error_limit: Optional[float] = None,
                           mark_extrema: str = "min", out_dir: Optional[Path] = None, filename: Optional[str] = None,
                           label_fn: Optional[Any] = None):
    """Common true/pred colour limits and one common symmetric error limit across models.

    ``label_fn`` defaults to the module-level :func:`display_model_name`. Pass a
    notebook-local resolver when a raw ``model_family`` id needs a notebook-specific
    paper-facing label (for example a feature-augmented ArGEnT checkpoint that
    should render as ``ArGEnT-A + GF``).
    """
    resolve_label = label_fn if label_fn is not None else display_model_name
    models = list(geom_nodes_by_model.keys())
    all_true = np.concatenate([geom_nodes_by_model[m][field_true_col].to_numpy() for m in models])
    all_pred = np.concatenate([geom_nodes_by_model[m][field_pred_col].to_numpy() for m in models])
    vmin, vmax = float(min(all_true.min(), all_pred.min())), float(max(all_true.max(), all_pred.max()))

    errs = [geom_nodes_by_model[m][field_pred_col].to_numpy() - geom_nodes_by_model[m][field_true_col].to_numpy() for m in models]
    err_limit = fixed_error_limit if fixed_error_limit is not None else robust_signed_limit(errs)
    enorm = TwoSlopeNorm(vmin=-err_limit, vcenter=0.0, vmax=err_limit)

    fig, axes = plt.subplots(3, len(models), figsize=(5.2 * len(models), 13.5), squeeze=False)
    for j, m in enumerate(models):
        d = geom_nodes_by_model[m]
        x, r = d["x_mm"].to_numpy(), d["r_mm"].to_numpy()
        t, p = d[field_true_col].to_numpy(), d[field_pred_col].to_numpy()
        err = p - t

        ax = axes[0, j]
        sc = ax.scatter(x, r, c=t, vmin=vmin, vmax=vmax, cmap="viridis", s=10)
        ax.set_title(f"{resolve_label(m)}\nTrue"); fig.colorbar(sc, ax=ax)

        ax = axes[1, j]
        sc = ax.scatter(x, r, c=p, vmin=vmin, vmax=vmax, cmap="viridis", s=10)
        ax.set_title("Predicted"); fig.colorbar(sc, ax=ax)

        ax = axes[2, j]
        sc = ax.scatter(x, r, c=err, cmap="RdBu_r", norm=enorm, s=10)
        ax.set_title(f"Signed error ({error_unit})"); fig.colorbar(sc, ax=ax)

        if mark_extrema == "min":
            ti, pi = int(np.argmin(t)), int(np.argmin(p))
        else:
            ti, pi = int(np.argmax(t)), int(np.argmax(p))
        # Compute equal-aspect bounds from the coordinate data so the disc
        # reads as a disc rather than an I-beam-like elongated shape.
        x_c = float(np.nanmean([x.min(), x.max()]))
        r_c = float(np.nanmean([r.min(), r.max()]))
        half_range = max(
            float(np.nanmax(np.abs(x - x_c))),
            float(np.nanmax(np.abs(r - r_c))),
        ) * 1.05 + 1e-6
        for ax in axes[:, j]:
            ax.scatter([x[ti]], [r[ti]], marker="*", s=160, edgecolor="k", facecolor="lime", label="true extremum", zorder=5)
            ax.scatter([x[pi]], [r[pi]], marker="X", s=120, edgecolor="k", facecolor="orange", label="pred extremum", zorder=5)
            ax.set_xlabel("x (mm)"); ax.set_ylabel("r (mm)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_c - half_range, x_c + half_range)
            ax.set_ylim(r_c - half_range, r_c + half_range)
        axes[0, j].legend(loc="upper right", fontsize=7)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    if out_dir is not None and filename:
        _save_fig(fig, out_dir, filename)
    return fig


def plot_arc_length_error(geom_nodes_by_model: Dict[str, pd.DataFrame], title_prefix: str,
                           out_dir: Optional[Path] = None, filename: Optional[str] = None):
    models = list(geom_nodes_by_model.keys())
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for m in models:
        d = geom_nodes_by_model[m].sort_values("arc_length_mm")
        if d["arc_length_mm"].isna().all():
            continue
        s = d["arc_length_mm"].to_numpy()
        label = display_model_name(m)
        axes[0].plot(s, d["pred_stress"] - d["true_stress"], label=label, alpha=0.8)
        axes[1].plot(s, d["pred_loglife"] - d["true_loglife"], label=label, alpha=0.8)
        axes[2].plot(s, d["true_loglife"], label=label, alpha=0.8)

    # zone boundaries from the first model's data (zones are geometry properties, model-independent)
    ref = geom_nodes_by_model[models[0]].sort_values("arc_length_mm")
    if ref["arc_length_mm"].notna().any():
        boundaries = ref.groupby("subzone_name")["arc_length_mm"].agg(["min", "max"]).dropna()
        for name, row in boundaries.iterrows():
            for ax in axes:
                ax.axvspan(row["min"], row["max"], alpha=0.04)

    axes[0].set_ylabel("Signed Stress error (MPa)")
    axes[1].set_ylabel("Signed LogLife error (decades)")
    axes[2].set_ylabel("True LogLife")
    axes[2].set_xlabel("Arc length (mm)")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    if out_dir is not None and filename:
        _save_fig(fig, out_dir, filename)
    return fig


def plot_zone_bar(zone_df: pd.DataFrame, title: str, out_dir: Optional[Path] = None, filename: Optional[str] = None,
                   label_fn: Optional[Any] = None):
    """``label_fn`` defaults to the module-level :func:`display_model_name`."""
    resolve_label = label_fn if label_fn is not None else display_model_name
    d = zone_df.copy()
    d = d[d["status"] != "missing"]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(13, 5))
    labels = sorted(d["subzone_name"].unique(), key=lambda z: PRINCIPAL_SUBZONES.index(z) if z in PRINCIPAL_SUBZONES else 99)
    families = ordered_model_families(d["model_family"].unique())
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(families))
    for i, fam in enumerate(families):
        sub = d[d["model_family"] == fam].set_index("subzone_name")
        vals = [sub.loc[l, "MAE"] if l in sub.index else np.nan for l in labels]
        counts = [sub.loc[l, "n_nodes"] if l in sub.index else 0 for l in labels]
        bars = ax.bar(x + (i - (len(families) - 1) / 2) * width, vals, width=width,
                      label=resolve_label(fam))
        for b, n in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, (b.get_height() or 0), f"n={int(n)}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("LogLife MAE"); ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    if out_dir is not None and filename:
        _save_fig(fig, out_dir, filename)
    return fig


def plot_bin_bar(bin_df: pd.DataFrame, title: str, out_dir: Optional[Path] = None, filename: Optional[str] = None,
                  label_fn: Optional[Any] = None):
    """``label_fn`` defaults to the module-level :func:`display_model_name`.

    Pass a notebook-local resolver (built on top of
    :func:`resolve_publication_label`) when a raw ``model_family`` id in this
    call needs a notebook-specific verified label, e.g. an ArGEnT checkpoint
    that is engineered-feature-augmented in this notebook's ablation
    directory but not in another notebook's.

    ``title`` remains in the signature for existing callers, but life-band
    plots intentionally omit a figure title so pooled and binned values share
    the same metric scale.
    """
    resolve_label = label_fn if label_fn is not None else display_model_name
    if bin_df.empty:
        return None
    bin_col = "life_bin" if "life_bin" in bin_df.columns else "bin"
    per_bin_df, full_set_df = split_life_bin_metrics(bin_df, bin_col=bin_col)
    if per_bin_df.empty and full_set_df.empty:
        return None
    families = ordered_model_families(bin_df["model_family"].unique())
    bins = PHYSICAL_LIFE_BIN_ORDER
    bin_labels = {
        "log_life < 2": "y < 2",
        "2 <= log_life < 3": "2 ≤ y < 3",
        "3 <= log_life < 4": "3 ≤ y < 4",
        "4 <= log_life < 6": "4 ≤ y < 6",
        "log_life >= 6": "y ≥ 6",
    }
    group_labels = ["Pooled", *(bin_labels.get(bin_name, bin_name) for bin_name in bins)]
    x = np.arange(len(group_labels), dtype=float)
    width = 0.8 / max(1, len(families))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    family_colors = {fam: colors[i % len(colors)] for i, fam in enumerate(families)}

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharey=True)
    for panel_label, metric, ax in zip(["(a)", "(b)"], ["MAE", "RMSE"], axes):
        metric_col = metric if metric in bin_df.columns else metric.lower() + "_loglife"
        ax.text(0.015, 0.97, panel_label, transform=ax.transAxes, ha="left", va="top",
                fontsize=14)
        for i, fam in enumerate(families):
            fam_full = full_set_df[full_set_df["model_family"] == fam]
            pooled_value = (
                float(fam_full.iloc[0][metric_col])
                if not fam_full.empty and metric_col in fam_full.columns
                else np.nan
            )
            sub = per_bin_df[per_bin_df["model_family"] == fam].set_index(bin_col)
            vals = [
                pooled_value,
                *(sub.loc[b, metric_col] if b in sub.index else np.nan for b in bins),
            ]
            unstable = [sub.loc[b, "unstable"] if b in sub.index else True for b in bins]
            bars = ax.bar(
                x + (i - (len(families) - 1) / 2) * width,
                vals,
                width=width,
                color=family_colors[fam],
                label=resolve_label(fam),
            )
            for bar, is_unstable in zip(bars[1:], unstable):
                if is_unstable and np.isfinite(bar.get_height()) and bar.get_height() > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        "unstable",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="red",
                        rotation=90,
                    )
        ax.axvline(0.5, color="black", linewidth=0.8)
        ax.set_ylabel(f"{metric} of log-life (decades)")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=20, ha="right")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    if out_dir is not None and filename:
        _save_fig(fig, out_dir, filename)
    return fig


DATA_EFFICIENCY_SERIES: List[Tuple[str, str, str]] = [
    ("PointNetMLPJoint_FP", "LC-PointNet", "#ff7f0e"),
    ("ArGEnT_self_att_noSDF", "ArGEnT-A", "#2ca02c"),
    ("GINOT-A", "GINOT-A", "#d62728"),
]
DATA_EFFICIENCY_COLUMNS: List[Tuple[str, str]] = [
    (FULL_TEST_SET_LABEL, "Pooled"),
    ("log_life < 2", "y < 2"),
    ("log_life >= 6", "y ≥ 6"),
]


def plot_data_efficiency_grid(csv_path: Path, out_dir: Path, filename: str, regime: str = "Zonal"):
    """Single 3 x 2 single-column data-efficiency figure (3.5 x 6.5 in).

    Rows are pooled, y < 2 and y >= 6; columns are MAE and RMSE. Reads the saved data-efficiency table, keeps only ``regime`` rows, and
    writes ``filename`` as PDF and PNG (300 dpi) into ``out_dir``.
    """
    df = pd.read_csv(csv_path)
    df = df[df["regime"] == regime]
    geometries = sorted(int(g) for g in df["training_geometries"].dropna().unique())
    letters = iter("abcdef")

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
    }):
        fig, axes = plt.subplots(3, 2, figsize=(3.5, 6.0), sharex=True)
        for row, (life_bin, header) in enumerate(DATA_EFFICIENCY_COLUMNS):
            for col, metric in enumerate(["MAE", "RMSE"]):
                ax = axes[row, col]
                for family, label, color in DATA_EFFICIENCY_SERIES:
                    sub = df[(df["model_family"] == family) & (df["life_bin"] == life_bin)]
                    sub = sub.dropna(subset=[metric]).sort_values("training_geometries")
                    ax.plot(sub["training_geometries"], sub[metric], marker="o", markersize=4,
                            linewidth=1.2, color=color, label=label)
                ax.set_xticks(geometries)
                ax.set_xlim(min(geometries) - 150, max(geometries) + 150)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
                ax.text(0.04, 0.96, f"({next(letters)}) {header}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, fontweight="bold")
                ax.tick_params(axis="x", labelrotation=45, pad=1, length=2)
                ax.tick_params(axis="y", pad=1, length=2)
                if row == 0:
                    ax.set_title(f"{metric} of log-life (decades)", fontsize=7.5, pad=3)
                if row == len(DATA_EFFICIENCY_COLUMNS) - 1:
                    ax.set_xlabel("Training geometries", labelpad=1)
        axes[0, 0].legend(fontsize=6, loc="upper right", borderaxespad=0.3,
                          handlelength=1.2, labelspacing=0.2, borderpad=0.3)
        fig.set_layout_engine("constrained", w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.04)
        validate_no_raw_publication_labels(fig, context=filename)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{filename}.pdf", dpi=300)
        fig.savefig(out_dir / f"{filename}.png", dpi=300)
    return fig


def plot_geometry_scatter(geom_df: pd.DataFrame, out_dir: Optional[Path] = None, filename_prefix: Optional[str] = None):
    figs = []
    families = ordered_model_families(geom_df["model_family"].unique())

    fig, ax = plt.subplots(figsize=(6, 6))
    for fam in families:
        d = geom_df[geom_df["model_family"] == fam]
        ax.scatter(d["true_min_loglife"], d["pred_min_loglife"], s=14, alpha=0.6,
                   label=display_model_name(fam))
    lims = [geom_df["true_min_loglife"].min(), geom_df["true_min_loglife"].max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("True minimum LogLife"); ax.set_ylabel("Predicted minimum LogLife")
    ax.legend(fontsize=7); fig.tight_layout()
    if out_dir is not None and filename_prefix:
        _save_fig(fig, out_dir, f"{filename_prefix}_min_loglife_scatter")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fam in families:
        d = geom_df[geom_df["model_family"] == fam]
        ax.scatter(d["true_max_stress"], d["pred_max_stress"], s=14, alpha=0.6,
                   label=display_model_name(fam))
    lims = [geom_df["true_max_stress"].min(), geom_df["true_max_stress"].max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("True maximum stress (MPa)"); ax.set_ylabel("Predicted maximum stress (MPa)")
    ax.legend(fontsize=7); fig.tight_layout()
    if out_dir is not None and filename_prefix:
        _save_fig(fig, out_dir, f"{filename_prefix}_max_stress_scatter")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for fam in families:
        d = geom_df[geom_df["model_family"] == fam]
        ax.hist(d["min_loglife_error_decades"], bins=20, alpha=0.5, label=display_model_name(fam))
    ax.set_xlabel("Per-geometry min-LogLife error (decades)"); ax.set_ylabel("Count")
    ax.legend(fontsize=7); ax.axvline(0, color="k", lw=0.8); fig.tight_layout()
    if out_dir is not None and filename_prefix:
        _save_fig(fig, out_dir, f"{filename_prefix}_min_loglife_error_hist")
    figs.append(fig)
    return figs


def plot_geometry_error_distributions(
    geom_df: pd.DataFrame,
    title: str,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
):
    if geom_df.empty:
        return None
    families = ordered_model_families(geom_df["model_family"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fam in families:
        d = geom_df[geom_df["model_family"] == fam]
        axes[0].hist(d["whole_geometry_loglife_mae"], bins=20, alpha=0.45,
                     label=display_model_name(fam))
        axes[1].hist(d["abs_min_loglife_error_decades"], bins=20, alpha=0.45,
                     label=display_model_name(fam))
    axes[0].set_title("A. Per-geometry whole-field LogLife MAE")
    axes[0].set_xlabel("MAE (decades)")
    axes[0].set_ylabel("Count")
    axes[1].set_title("B. Per-geometry absolute minimum-LogLife error")
    axes[1].set_xlabel("|error| (decades)")
    axes[1].set_ylabel("Count")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    if out_dir is not None and filename:
        _save_fig(fig, out_dir, filename)
    return fig


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)


def save_json(obj: Any, out_dir: Path, name: str) -> None:
    """Save JSON with non-finite numeric values normalized to JSON null."""

    def _json_ready(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_ready(v) for v in value]
        if isinstance(value, np.ndarray):
            return [_json_ready(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return None if not math.isfinite(float(value)) else float(value)
        if isinstance(value, bool):
            return value
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, Path):
            return str(value)
        return value

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(obj), f, indent=2, default=str, allow_nan=False)
