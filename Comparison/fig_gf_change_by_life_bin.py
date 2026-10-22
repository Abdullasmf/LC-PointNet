from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CSV = HERE / "results" / "02_engineered_geometric_features" / "paired_headfeat_vs_baseline_by_life_bin_with_ginot.csv"
OUT = HERE / "results" / "02_engineered_geometric_features"

MODELS = [
    ("PointNetMLPJoint", "GC-PointNet", "C0"),
    ("PointNetMLPJoint_FP", "LC-PointNet", "C1"),
    ("ArGEnT_self_att_noSDF", "ArGEnT-A", "C2"),
    ("GINOT-A", "GINOT-A", "C3"),
]
BINS = [
    ("Full test set", "Pooled"),
    ("log_life < 2", r"$\mathrm{y}<2$"),
    ("2 <= log_life < 3", r"$2\leq \mathrm{y}<3$"),
    ("3 <= log_life < 4", r"$3\leq \mathrm{y}<4$"),
    ("4 <= log_life < 6", r"$4\leq \mathrm{y}<6$"),
    ("log_life >= 6", r"$\mathrm{y}\geq 6$"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.labelsize": 9,
})

df = pd.read_csv(CSV)
df = df[df["regime"] == "Zonal"]

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
x = np.arange(len(BINS))
w = 0.2
for ax, metric, label, tag in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"], ["(a)", "(b)"]):
    for i, (key, name, color) in enumerate(MODELS):
        vals = []
        for b, _ in BINS:
            r = df[(df["model_family_baseline"] == key) & (df["life_bin"] == b)]
            assert len(r) == 1, (key, b, len(r))
            r = r.iloc[0]
            base, abl = r[f"baseline_{metric}_loglife"], r[f"ablation_{metric}_loglife"]
            vals.append((abl - base) / base * 100)
        ax.bar(x + (i - 1.5) * w, vals, w, color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0.5, color="k", lw=0.6)
    ax.set_xlim(-0.6, len(BINS) - 0.4)
    ax.set_ylim(-95, 15)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in BINS], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"Change in {label} with GF (%)")
    ax.text(0.02, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontsize=9)
axes[0].legend(loc="lower right", ncol=1, frameon=False, handlelength=1.2, labelspacing=0.3)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_gf_change_by_life_bin.{ext}", dpi=300)
