#!/usr/bin/env python
"""TUSZ 60s hyperedge sweep — single PDF, two side-by-side panels.

Left:  AUROC vs E_h (circle marker).
Right: F1    vs E_h (square marker).

E_h ∈ {1, 3, 5, 7, 9}. E_h=1/3 from the main results table (3-seed mean);
E_h=5/7/9 from sweep_tusz60_E{5,7,9}_s123 (single seed).

Horizontal reference lines: GRU-GCN and EvoBrain (TUSZ 60s, paper table),
drawn in two slightly different blues with different dash styles.

A single legend sits centered above both panels.
"""
import os, sys, glob, warnings, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, "/storage/project/r-nimam6-0/hkim3239/disease/src")
from utils import thresh_max_f1
from sklearn.metrics import roc_auc_score, f1_score

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["cmr10", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.labelsize": 30,
    "xtick.labelsize": 26, "ytick.labelsize": 26,
    "legend.fontsize": 22, "legend.frameon": False,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.4, "lines.markersize": 10,
    "figure.dpi": 120,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "/storage/project/r-nimam6-0/hkim3239/disease/figures/analysis"
os.makedirs(OUT, exist_ok=True)

OURS_COLOR    = "#a6243a"   # red
GRUGCN_COLOR  = "#3a7ab8"   # lighter blue
EVO_COLOR     = "#143a6b"   # darker navy


def _metrics(run, clip_len=60):
    rd = glob.glob(os.path.join(run, f"TUSZ/detection/{clip_len}/*"))
    if not rd: return None
    d = rd[0]
    tr = os.path.join(d, "test_results.npz"); dv = os.path.join(d, "dev_results.npz")
    if not (os.path.exists(tr) and os.path.exists(dv)): return None
    T = np.load(tr); D = np.load(dv)
    yt_t = T["y_true_clip"] if "y_true_clip" in T.files else T["y_true"]
    yp_t = T["y_prob_clip"] if "y_prob_clip" in T.files else T["y_prob"]
    yt_d = D["y_true_clip"] if "y_true_clip" in D.files else D["y_true"]
    yp_d = D["y_prob_clip"] if "y_prob_clip" in D.files else D["y_prob"]
    tau  = thresh_max_f1(y_true=yt_d, y_prob=yp_d)
    return (roc_auc_score(yt_t, yp_t),
            f1_score(yt_t, (yp_t > tau).astype(int), average="binary",
                     zero_division=0))


def _agg(pattern, clip_len=60):
    runs = sorted(glob.glob(pattern))
    M = [_metrics(r, clip_len) for r in runs]
    M = [m for m in M if m is not None]
    if not M: return None
    a = np.array([x[0] for x in M]); f = np.array([x[1] for x in M])
    return float(a.mean()), float(f.mean())


# E_h=1, 3 — paper main-results-table values (3-seed mean).
AUTH = {1: (0.877, 0.569), 3: (0.877, 0.537)}

points = []
for eh in [1, 3, 5, 7, 9]:
    if eh in AUTH:
        a, f = AUTH[eh]
    else:
        out = _agg(f"/storage/scratch1/3/hkim3239/eeg/runs/sweep_tusz60_E{eh}_s*")
        if out is None:
            raise SystemExit(f"missing runs for E_h={eh}")
        a, f = out
    points.append((eh, a, f))
    print(f"  E_h={eh}  AUROC={a:.3f}  F1={f:.3f}")

Eh    = np.array([p[0] for p in points])
mu_a  = np.array([p[1] for p in points])
mu_f  = np.array([p[2] for p in points])

# Paper-table TUSZ 60s baselines.
GRUGCN  = dict(name="GRU-GCN",  auroc=0.822, f1=0.438,
               color=GRUGCN_COLOR, ls=(0, (4, 2)))     # dashed
EVOBRAIN = dict(name="EvoBrain", auroc=0.865, f1=0.483,
                color=EVO_COLOR,    ls=(0, (1, 1.5)))  # dotted


def _draw_panel(ax, ys, ylabel, ours_marker, baselines_key, ylim, yticks):
    lG = ax.axhline(GRUGCN[baselines_key], color=GRUGCN["color"],
                    linestyle=GRUGCN["ls"], linewidth=2.4,
                    label=GRUGCN["name"])
    lE = ax.axhline(EVOBRAIN[baselines_key], color=EVOBRAIN["color"],
                    linestyle=EVOBRAIN["ls"], linewidth=2.4,
                    label=EVOBRAIN["name"])
    (lO,) = ax.plot(Eh, ys, color=OURS_COLOR, linestyle="-",
                    marker=ours_marker, markerfacecolor="white",
                    markeredgewidth=2.0,
                    label="LightSTHyper (Ours)")
    ax.set_xlabel(r"$E_h$ (number of hyperedges)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(Eh)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return lO, lG, lE


fig, (axA, axF) = plt.subplots(1, 2, figsize=(13.0, 4.4),
                               constrained_layout=False)

lO, lG, lE = _draw_panel(
    axA, mu_a, "AUROC", ours_marker="o", baselines_key="auroc",
    ylim=(0.80, 0.90),
    yticks=[0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
)
_draw_panel(
    axF, mu_f, "F1", ours_marker="s", baselines_key="f1",
    ylim=(0.40, 0.60),
    yticks=[0.40, 0.45, 0.50, 0.55, 0.60],
)

# Single legend centered above both subplots.
fig.legend(handles=[lO, lG, lE],
           loc="upper center", bbox_to_anchor=(0.5, 1.02),
           ncol=3, columnspacing=1.8, handlelength=2.4,
           borderaxespad=0.0)

# Tight layout but leave headroom for the legend.
fig.subplots_adjust(left=0.085, right=0.965, top=0.80, bottom=0.22,
                   wspace=0.32)

out_path = os.path.join(OUT, "tusz60_hyperedge_sweep_split.pdf")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("wrote", out_path)
