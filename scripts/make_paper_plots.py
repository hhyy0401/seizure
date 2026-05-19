#!/usr/bin/env python
"""Make ablation + hyperedge-sweep PDFs for the paper.

Style: Computer-Modern (cmr10) serif, no grid, no error bars,
no title, single axes with twin y (AUROC left / F1 right),
distinct markers + solid/dashed line styles.
"""
import os, sys, glob, warnings, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
sys.path.insert(0, "/storage/project/r-nimam6-0/hkim3239/disease/src")
from utils import thresh_max_f1
from sklearn.metrics import roc_auc_score, f1_score

# ---- Roman / Computer-Modern look ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["cmr10", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "legend.frameon": False,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.7, "lines.markersize": 8,
    "figure.dpi": 120,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "/storage/project/r-nimam6-0/hkim3239/disease/figures"
os.makedirs(OUT, exist_ok=True)

COL_A = "#1f4e79"   # AUROC — deep blue
COL_F = "#a6243a"   # F1    — deep red

# ---- metrics helpers ----
def metrics_for(run, ds, clip_len=12):
    rd = glob.glob(os.path.join(run, f"{ds}/detection/{clip_len}/*"))
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
    return roc_auc_score(yt_t, yp_t), f1_score(yt_t, (yp_t > tau).astype(int),
                                               average="binary", zero_division=0)

def agg(runs, ds, clip_len=12):
    M = [metrics_for(r, ds, clip_len) for r in runs]; M = [m for m in M if m]
    if not M: return None
    a = np.array([x[0] for x in M]); f = np.array([x[1] for x in M])
    return a.mean(), f.mean(), len(M)

def agg_with_std(runs, ds, clip_len):
    """Return (auroc_mean, auroc_std, f1_mean, f1_std, n) or None."""
    M = [metrics_for(r, ds, clip_len) for r in runs]; M = [m for m in M if m]
    if not M: return None
    a = np.array([x[0] for x in M]); f = np.array([x[1] for x in M])
    return a.mean(), a.std(), f.mean(), f.std(), len(M)

# Authoritative numbers from the main results table — used to override the
# Full ablation row and the E_h in {1,2,3} sweep points for cross-table
# consistency (mu_AUROC, sd_AUROC, mu_F1, sd_F1).
AUTH = {
    (12, 1): (0.898, 0.003, 0.519, 0.019),
    (12, 2): (0.892, 0.006, 0.440, 0.023),
    (12, 3): (0.889, 0.006, 0.504, 0.029),
    (60, 1): (0.877, 0.017, 0.569, 0.017),
    (60, 2): (0.848, 0.019, 0.463, 0.040),
    (60, 3): (0.877, 0.005, 0.537, 0.039),
}

def dual_axis_plot(x, mu_a, mu_f, xlabel, xticks, out_path,
                   ylim_a=None, ylim_f=None,
                   yticks_a=None, yticklabels_a=None,
                   yticks_f=None, yticklabels_f=None,
                   legend_loc="best",
                   figsize=(3.3, 2.5), markersize=None):
    fig, axA = plt.subplots(figsize=figsize, constrained_layout=True)
    axF = axA.twinx()

    mk = {} if markersize is None else {"markersize": markersize}
    lA, = axA.plot(x, mu_a, color=COL_A, linestyle="-", marker="o",
                   markerfacecolor="white", markeredgewidth=1.4, label="AUROC", **mk)
    lF, = axF.plot(x, mu_f, color=COL_F, linestyle="--", marker="s",
                   markerfacecolor="white", markeredgewidth=1.4, label="F1", **mk)

    axA.set_xlabel(xlabel)
    axA.set_ylabel("AUROC", color=COL_A)
    axF.set_ylabel("F1",    color=COL_F)
    axA.tick_params(axis="y", colors=COL_A)
    axF.tick_params(axis="y", colors=COL_F)
    axA.spines["left"].set_color(COL_A)
    axF.spines["right"].set_color(COL_F)
    axA.spines["right"].set_visible(False)
    axF.spines["left"].set_visible(False)
    axA.spines["top"].set_visible(False)
    axF.spines["top"].set_visible(False)

    if xticks is not None:
        axA.set_xticks(xticks)
    if ylim_a is not None: axA.set_ylim(*ylim_a)
    if ylim_f is not None: axF.set_ylim(*ylim_f)
    if yticks_a is not None:
        axA.set_yticks(yticks_a)
        if yticklabels_a is not None:
            axA.set_yticklabels(yticklabels_a)
    if yticks_f is not None:
        axF.set_yticks(yticks_f)
        if yticklabels_f is not None:
            axF.set_yticklabels(yticklabels_f)

    axA.legend(handles=[lA, lF], loc=legend_loc)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)

# ==========================================================================
# Plot 1 — TUSZ12 hyperedge sweep, E_h ∈ {1, 3, 5, 7}
# E_h=1, 3 from the paper table; E_h=5, 7 recomputed with same protocol.
# E_h=2 dropped per request.
# ==========================================================================
a5, f5, _ = agg(sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/eh_tusz12_E5_s*")), "TUSZ")
a7, f7, _ = agg(sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/eh_tusz12_E7_s*")), "TUSZ")

Eh   = np.array([1, 3, 5, 7])
mu_a = np.array([0.898, 0.889, a5, a7])
mu_f = np.array([0.519, 0.504, f5, f7])

dual_axis_plot(
    x=Eh, mu_a=mu_a, mu_f=mu_f,
    xlabel=r"$E_h$ (number of hyperedges)",
    xticks=Eh,
    out_path=os.path.join(OUT, "tusz12_hyperedge_sweep.pdf"),
    ylim_a=(0.840, 0.905),
    ylim_f=(0.34, 0.55),
    yticks_a=[0.84, 0.86, 0.88, 0.90],
    yticklabels_a=["0.84", "0.86", "0.88", "0.90"],
    yticks_f=[0.35, 0.40, 0.45, 0.50, 0.55],
    yticklabels_f=["0.35", "0.40", "0.45", "0.50", "0.55"],
    legend_loc="lower left",
    figsize=(4.0, 2.5),
    markersize=5,
)

# ==========================================================================
# Plot 2 — TUSZ12 ablation (run after sbatch completes; placeholder names).
# ==========================================================================
abl_tusz12 = [
    ("Full",                       "/storage/scratch1/3/hkim3239/eeg/runs/final_tusz12_E1_s*"),
    (r"$(-)$ node" "\n" "embedding","/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz12v2_no_node_emb_s*"),
    (r"$(-)$ Mamba",               "/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz12v2_no_mamba_s*"),
    (r"$(-)$ ST-" "\n" "hyperedge", "/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz12v2_pair_s*"),
]
have_all = True
rows = []
for lbl, pat in abl_tusz12:
    runs = sorted(glob.glob(pat))
    out = agg(runs, "TUSZ") if runs else None
    if out is None:
        have_all = False
        print(f"  skip ablation plot — missing: {lbl}  pattern={pat}")
        continue
    am, fm, n = out
    if lbl == "Full":
        mu_a, _, mu_f, _ = AUTH[(12, 1)]
        am, fm = mu_a, mu_f
    rows.append((lbl, am, fm, n))
    print(f"  {lbl.replace(chr(10),' '):40s}  AUROC={am:.3f}  F1={fm:.3f}  n={n}")

if have_all and len(rows) == len(abl_tusz12):
    labels = [r[0] for r in rows]
    am     = np.array([r[1] for r in rows])
    fm     = np.array([r[2] for r in rows])

    fig, axA = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    axF = axA.twinx()

    x = np.arange(len(rows))
    lA, = axA.plot(x, am, color=COL_A, linestyle="-",  marker="o",
                   markerfacecolor="white", markeredgewidth=1.5, label="AUROC")
    lF, = axF.plot(x, fm, color=COL_F, linestyle="--", marker="s",
                   markerfacecolor="white", markeredgewidth=1.5, label="F1")

    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=11)
    axA.set_ylabel("AUROC", color=COL_A); axF.set_ylabel("F1", color=COL_F)
    axA.tick_params(axis="y", colors=COL_A); axF.tick_params(axis="y", colors=COL_F)
    axA.spines["left"].set_color(COL_A);    axF.spines["right"].set_color(COL_F)
    axA.spines["right"].set_visible(False); axF.spines["left"].set_visible(False)
    axA.spines["top"].set_visible(False);   axF.spines["top"].set_visible(False)

    # Clean y-tick formatting: fewer decimals, explicit nice locations
    axA.set_ylim(0.872, 0.900)
    axA.set_yticks([0.875, 0.880, 0.885, 0.890, 0.895])
    axA.set_yticklabels(["0.875", "0.880", "0.885", "0.890", "0.895"])
    axF.set_ylim(0.28, 0.52)
    axF.set_yticks([0.30, 0.35, 0.40, 0.45, 0.50])
    axF.set_yticklabels(["0.30", "0.35", "0.40", "0.45", "0.50"])

    axA.legend(handles=[lA, lF], loc="lower left")

    out_p = os.path.join(OUT, "tusz12_ablation.pdf")
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_p)
else:
    print("(ablation PDF will be generated after the TUSZ ablation sbatch jobs finish)")

# ==========================================================================
# Plot 3 — TUSZ60 hyperedge sweep, E_h in {1, 2, 3, 5, 7, 9}
# E_h=1/2/3 from final_tusz60_* (3 seeds each); E_h=5/7/9 from
# sweep_tusz60_E579_* (single seed, 123).
# ==========================================================================
sw60 = {
    1: sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/final_tusz60_E1_s*")),
    3: sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/final_tusz60_E3_s*")),
    5: sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/sweep_tusz60_E5_s*")),
    7: sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/sweep_tusz60_E7_s*")),
    9: sorted(glob.glob("/storage/scratch1/3/hkim3239/eeg/runs/sweep_tusz60_E9_s*")),
}
rows60 = []
for eh, runs in sw60.items():
    out = agg(runs, "TUSZ", clip_len=60) if runs else None
    if out is None:
        print(f"  TUSZ60 E_h={eh}: NO RUNS YET")
        continue
    am, fm, n = out
    # Override E_h ∈ {1,2,3} with main-results-table numbers for consistency.
    src = "raw"
    if (60, eh) in AUTH:
        mu_a, _, mu_f, _ = AUTH[(60, eh)]
        am, fm = mu_a, mu_f
        src = "AUTH"
    print(f"  TUSZ60 E_h={eh}: AUROC={am:.3f}  F1={fm:.3f}  n={n}  [{src}]")
    rows60.append((eh, am, fm, n))

if len(rows60) >= 4:
    Eh60   = np.array([r[0] for r in rows60])
    mu_a60 = np.array([r[1] for r in rows60])
    mu_f60 = np.array([r[2] for r in rows60])
    dual_axis_plot(
        x=Eh60, mu_a=mu_a60, mu_f=mu_f60,
        xlabel=r"$E_h$ (number of hyperedges)",
        xticks=Eh60,
        out_path=os.path.join(OUT, "tusz60_hyperedge_sweep.pdf"),
        ylim_a=(0.840, 0.920), ylim_f=(0.45, 0.65),
        yticks_a=[0.84, 0.86, 0.88, 0.90, 0.92],
        yticklabels_a=["0.84", "0.86", "0.88", "0.90", "0.92"],
        yticks_f=[0.45, 0.50, 0.55, 0.60, 0.65],
        yticklabels_f=["0.45", "0.50", "0.55", "0.60", "0.65"],
        legend_loc="lower left",
    )
else:
    print("(TUSZ60 sweep PDF will be generated once E_h=5/7/9 jobs finish)")

# ==========================================================================
# Plot 4 — TUSZ60 ablation (Full E_h=3 / -node_emb / -Mamba / -ST-hyperedge).
# ==========================================================================
abl_tusz60 = [
    ("Full",                       "/storage/scratch1/3/hkim3239/eeg/runs/final_tusz60_E3_s*"),
    (r"$(-)$ node" "\n" "embedding","/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz60_no_node_emb_s*"),
    (r"$(-)$ Mamba",               "/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz60_no_mamba_s*"),
    (r"$(-)$ ST-" "\n" "hyperedge", "/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz60_pair_s*"),
]
rows60a = []
have_all60a = True
for lbl, pat in abl_tusz60:
    runs = sorted(glob.glob(pat))
    out = agg(runs, "TUSZ", clip_len=60) if runs else None
    if out is None:
        have_all60a = False
        print(f"  skip TUSZ60 ablation -- missing: {lbl}  pattern={pat}")
        continue
    am, fm, n = out
    if lbl == "Full":
        mu_a, _, mu_f, _ = AUTH[(60, 3)]
        am, fm = mu_a, mu_f
    rows60a.append((lbl, am, fm, n))
    print(f"  TUSZ60 {lbl.replace(chr(10),' '):40s}  AUROC={am:.3f}  F1={fm:.3f}  n={n}")

if have_all60a and len(rows60a) == len(abl_tusz60):
    labels = [r[0] for r in rows60a]
    am     = np.array([r[1] for r in rows60a])
    fm     = np.array([r[2] for r in rows60a])

    fig, axA = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    axF = axA.twinx()
    x = np.arange(len(rows60a))
    lA, = axA.plot(x, am, color=COL_A, linestyle="-",  marker="o",
                   markerfacecolor="white", markeredgewidth=1.5, label="AUROC")
    lF, = axF.plot(x, fm, color=COL_F, linestyle="--", marker="s",
                   markerfacecolor="white", markeredgewidth=1.5, label="F1")
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=11)
    axA.set_ylabel("AUROC", color=COL_A); axF.set_ylabel("F1", color=COL_F)
    axA.tick_params(axis="y", colors=COL_A); axF.tick_params(axis="y", colors=COL_F)
    axA.spines["left"].set_color(COL_A);    axF.spines["right"].set_color(COL_F)
    axA.spines["right"].set_visible(False); axF.spines["left"].set_visible(False)
    axA.spines["top"].set_visible(False);   axF.spines["top"].set_visible(False)
    axA.legend(handles=[lA, lF], loc="best")

    out_p = os.path.join(OUT, "tusz60_ablation.pdf")
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_p)
else:
    print("(TUSZ60 ablation PDF will be generated after the abl_tusz60 jobs finish)")

# ==========================================================================
# LaTeX ablation tables (TUSZ12, TUSZ60, and side-by-side combined).
# Drops .tex fragments into paper/ ; \input{} from the main manuscript.
# ==========================================================================
PAPER_DIR = "/storage/project/r-nimam6-0/hkim3239/disease/paper"
os.makedirs(PAPER_DIR, exist_ok=True)

def fmt_cell(mu, sd):
    return f"{mu:.3f}$\\pm${sd:.3f}"

# Variant rows shared by both regimes (label, glob pattern template — {ds_tag}
# gets replaced per clip length below).
_ABL_VARIANTS = [
    ("Full",                            "{full}"),
    ("$(-)$ node embedding",            "{base}_no_node_emb_s*"),
    ("$(-)$ Mamba backbone",            "{base}_no_mamba_s*"),
    ("$(-)$ ST-hyperedge block",        "{base}_pair_s*"),
]
_BASELINES = {
    12: dict(base="/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz12v2",
             full="/storage/scratch1/3/hkim3239/eeg/runs/final_tusz12_E1_s*"),
    60: dict(base="/storage/scratch1/3/hkim3239/eeg/runs/abl_tusz60",
             full="/storage/scratch1/3/hkim3239/eeg/runs/final_tusz60_E3_s*"),
}
_FULL_EH = {12: 1, 60: 3}

def collect_ablation_rows(clip_len):
    """Returns list of (label, am, asd, fm, fsd, n) or None if any row missing.
    The Full row is overridden by AUTH (main-results-table numbers) so that
    ablation tables stay consistent with the headline benchmark table."""
    tmpl = _BASELINES[clip_len]
    out = []
    for lbl, pat_tmpl in _ABL_VARIANTS:
        pat = pat_tmpl.format(**tmpl)
        runs = sorted(glob.glob(pat))
        agg = agg_with_std(runs, "TUSZ", clip_len) if runs else None
        if agg is None: return None
        am, asd, fm, fsd, n = agg
        if lbl == "Full":
            mu_a, sd_a, mu_f, sd_f = AUTH[(clip_len, _FULL_EH[clip_len])]
            am, asd, fm, fsd = mu_a, sd_a, mu_f, sd_f
        out.append((lbl, am, asd, fm, fsd, n))
    return out

def write_single_table(rows, clip_len, out_path):
    n_seeds = rows[0][5]
    lines = [
        f"% Auto-generated by scripts/make_paper_plots.py — TUSZ {clip_len}s ablation",
        f"% Baseline = uni-Mamba + no-aux + node\\_emb + $E_h{{=}}{_FULL_EH[clip_len]}$ "
        f"(matches final\\_tusz{clip_len}\\_E{_FULL_EH[clip_len]}). 3 seeds, mean$\\pm$std.",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Variant & AUROC & F1 \\\\",
        "\\midrule",
    ]
    for lbl, am, asd, fm, fsd, n in rows:
        prefix = "\\textbf{" + lbl + "}" if lbl == "Full" else lbl
        lines.append(f"{prefix} & {fmt_cell(am, asd)} & {fmt_cell(fm, fsd)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", out_path)

def write_combined_table(rows12, rows60, out_path):
    n12, n60 = rows12[0][5], rows60[0][5]
    lines = [
        "% Auto-generated by scripts/make_paper_plots.py",
        "% Combined ablation, TUSZ 12s and 60s detection. Mean$\\pm$std over "
        f"{n12} (12s) / {n60} (60s) seeds.",
        "% TUSZ 12s baseline = final\\_tusz12\\_E1 (uni-Mamba + no-aux + "
        f"node\\_emb, $E_h{{=}}{_FULL_EH[12]}$).",
        "% TUSZ 60s baseline = final\\_tusz60\\_E3 (uni-Mamba + no-aux + "
        f"node\\_emb, $E_h{{=}}{_FULL_EH[60]}$).",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{TUSZ 12\\,s} & \\multicolumn{2}{c}{TUSZ 60\\,s} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Variant & AUROC & F1 & AUROC & F1 \\\\",
        "\\midrule",
    ]
    for (lbl, a12, sa12, f12, sf12, _), (_, a60, sa60, f60, sf60, _) \
            in zip(rows12, rows60):
        prefix = "\\textbf{" + lbl + "}" if lbl == "Full" else lbl
        lines.append(f"{prefix} & {fmt_cell(a12, sa12)} & {fmt_cell(f12, sf12)} "
                     f"& {fmt_cell(a60, sa60)} & {fmt_cell(f60, sf60)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", out_path)

rows12 = collect_ablation_rows(12)
rows60 = collect_ablation_rows(60)
if rows12 is not None:
    write_single_table(rows12, 12, os.path.join(PAPER_DIR, "ablation_tusz12.tex"))
else:
    print("(skip ablation_tusz12.tex -- some TUSZ12 ablation runs missing)")
if rows60 is not None:
    write_single_table(rows60, 60, os.path.join(PAPER_DIR, "ablation_tusz60.tex"))
else:
    print("(skip ablation_tusz60.tex -- some TUSZ60 ablation runs missing)")
if rows12 is not None and rows60 is not None:
    write_combined_table(rows12, rows60,
                         os.path.join(PAPER_DIR, "ablation_combined.tex"))
else:
    print("(skip ablation_combined.tex -- need both regimes)")
