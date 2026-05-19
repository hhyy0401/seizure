"""F1 / AUROC training curves for the CHB-MIT 12s ablation + hyperedge sweep.

Reads the TensorBoard event files written by main.py, aggregates over seeds,
and produces paper-style plots: AUROC (solid) + F1 (dashed) on the same axis
(twin y-axis), one figure per config + a combined comparison figure.

Usage:
    python scripts/plot_ablation_curves.py \
        --runs_root /storage/scratch1/3/hkim3239/eeg/runs \
        --out_dir   figures/abl_chb12

Expects run dirs matching:
    {final_chb12_E{1,2,3}, sweep_chb12_E{5,7,9}, abl_chb12_{no_node_emb,
     no_mamba, pair}}_s{123,456,789}_*
each containing CHBMIT/detection/12/<run>/events.out.tfevents.*
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CONFIGS = [
    # (group_key, glob prefix, display name, color)
    ("final_E1", "final_chb12_E1_s",          "Full (E=1)",         "#1f77b4"),
    ("final_E2", "final_chb12_E2_s",          "Full (E=2)",         "#aec7e8"),
    ("final_E3", "final_chb12_E3_s",          "Full (E=3)",         "#7fb3d5"),
    ("sweep_E5", "sweep_chb12_E5_s",          "Full (E=5)",         "#5dade2"),
    ("sweep_E7", "sweep_chb12_E7_s",          "Full (E=7)",         "#3498db"),
    ("sweep_E9", "sweep_chb12_E9_s",          "Full (E=9)",         "#2874a6"),
    ("no_node",  "abl_chb12_no_node_emb_s",   "− node emb",         "#ff7f0e"),
    ("no_mamba", "abl_chb12_no_mamba_s",      "− Mamba (linear)",   "#2ca02c"),
    ("pair",     "abl_chb12_pair_s",          "Pairwise attn",      "#d62728"),
]
SEED_RE = re.compile(r"_s(\d+)_")


def find_event_dir(run_root: str, prefix: str):
    """Return list of (seed, event-dir) for every run whose name starts with prefix."""
    out = []
    for top in sorted(glob.glob(os.path.join(run_root, prefix + "*"))):
        m = SEED_RE.search(os.path.basename(top))
        if not m:
            continue
        seed = int(m.group(1))
        ev = glob.glob(os.path.join(top, "CHBMIT/detection/12/*"))
        ev = [e for e in ev if os.path.isdir(e) and
              any(f.startswith("events.out") for f in os.listdir(e))]
        if ev:
            out.append((seed, ev[0]))
    return out


def load_curve(ev_dir: str, tag: str):
    """Return (steps, values) for one tag, or None if missing."""
    ea = EventAccumulator(ev_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    evs = ea.Scalars(tag)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])


def interp_to_grid(curves, n=100):
    """curves: list of (steps, values). Returns (grid, mean, std) on common grid."""
    if not curves:
        return None
    all_steps = np.concatenate([c[0] for c in curves])
    grid = np.linspace(all_steps.min(), all_steps.max(), n)
    interp = np.stack([np.interp(grid, s, v) for s, v in curves])
    return grid, interp.mean(0), interp.std(0)


def plot_one(ax_l, ax_r, label, color, auroc, f1):
    """auroc / f1: (grid, mean, std) or None."""
    if auroc is not None:
        g, m, s = auroc
        ax_l.plot(g, m, color=color, linestyle="-", linewidth=1.6, label=label)
        ax_l.fill_between(g, m - s, m + s, color=color, alpha=0.12)
    if f1 is not None:
        g, m, s = f1
        ax_r.plot(g, m, color=color, linestyle="--", linewidth=1.4)
        ax_r.fill_between(g, m - s, m + s, color=color, alpha=0.08)


def make_per_config_plot(out_path, label, auroc, f1):
    fig, ax_l = plt.subplots(figsize=(5.0, 3.4), dpi=140)
    ax_r = ax_l.twinx()
    plot_one(ax_l, ax_r, label, "#1f77b4", auroc, f1)
    ax_l.set_xlabel("training step")
    ax_l.set_ylabel("AUROC  (solid)", color="#1f77b4")
    ax_r.set_ylabel("F1  (dashed)", color="#777777")
    ax_l.tick_params(axis="y", colors="#1f77b4")
    ax_l.set_ylim(0.5, 1.0)
    ax_r.set_ylim(0.0, 0.7)
    ax_l.grid(True, alpha=0.25)
    ax_l.set_title(label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_combined_plot(out_path, series):
    """series: [(label, color, auroc_tuple, f1_tuple), ...]"""
    fig, ax_l = plt.subplots(figsize=(7.5, 4.4), dpi=140)
    ax_r = ax_l.twinx()
    for label, color, au, f in series:
        plot_one(ax_l, ax_r, label, color, au, f)
    ax_l.set_xlabel("training step")
    ax_l.set_ylabel("AUROC  (solid)")
    ax_r.set_ylabel("F1  (dashed)")
    ax_l.set_ylim(0.5, 1.0)
    ax_r.set_ylim(0.0, 0.7)
    ax_l.grid(True, alpha=0.25)
    ax_l.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax_l.set_title("CHB-MIT 12s — eval AUROC (solid) / F1 (dashed)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="/storage/scratch1/3/hkim3239/eeg/runs")
    ap.add_argument("--out_dir",   default="figures/abl_chb12")
    ap.add_argument("--auroc_tag", default="eval/auroc")
    ap.add_argument("--f1_tag",    default="eval/F1")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    series = []

    for key, prefix, label, color in CONFIGS:
        ev_dirs = find_event_dir(args.runs_root, prefix)
        if not ev_dirs:
            print(f"[skip] {key}: no runs found under {args.runs_root}/{prefix}*")
            continue
        au_curves, f1_curves = [], []
        for seed, ev in ev_dirs:
            au = load_curve(ev, args.auroc_tag)
            f1 = load_curve(ev, args.f1_tag)
            if au is not None:
                au_curves.append(au)
            if f1 is not None:
                f1_curves.append(f1)
        au = interp_to_grid(au_curves)
        f1 = interp_to_grid(f1_curves)
        print(f"[ok]   {key}: {len(ev_dirs)} runs, "
              f"AUROC end ~ {au[1][-1]:.3f}, F1 end ~ {f1[1][-1]:.3f}"
              if au is not None and f1 is not None
              else f"[partial] {key}: AUROC={au is not None}, F1={f1 is not None}")
        make_per_config_plot(
            os.path.join(args.out_dir, f"curve_{key}.pdf"), label, au, f1)
        series.append((label, color, au, f1))

    if series:
        make_combined_plot(os.path.join(args.out_dir, "curves_combined.pdf"), series)
        print(f"wrote {len(series)+1} figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
