"""Parse training logs and plot loss curves for all 6 models (paper-style)."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_DIR = Path("/home/hkim3239/eeg/evobrain/logs")
OUT_PNG = Path("/home/hkim3239/eeg/evobrain/loss_curves.png")

RUNS = [
    ("evobrain",              "evobrain_20260512_212356",              "EvoBrain"),
    ("light_attention",       "light_attention_20260513_002041",       "Light-Attn"),
    ("light_mamba_band_plv",  "light_mamba_band_plv_20260512_212356",  "PLV"),
    ("light_dyn_hyper",       "light_dyn_hyper_20260513_002041",       "Dyn-Hyper"),
    ("light_static_hyper",    "light_static_hyper_20260513_002041",    "Static-Hyper"),
    ("light_attn_band_gated", "light_attn_band_gated_20260513_002041", "no-PLV"),
]

COLORS = {
    "evobrain":              "#555555",   # dark gray (baseline)
    "light_attention":       "#b0b0b0",   # light gray (baseline)
    "light_mamba_band_plv":  "#d62728",   # red
    "light_dyn_hyper":       "#2ca02c",   # green
    "light_static_hyper":    "#9467bd",   # purple
    "light_attn_band_gated": "#ff7f0e",   # orange
}
LINESTYLES = {
    "evobrain":              "--",        # baseline = dashed
    "light_attention":       "--",        # baseline = dashed
    "light_mamba_band_plv":  "-",
    "light_dyn_hyper":       "-",
    "light_static_hyper":    "-",
    "light_attn_band_gated": "-",
}

PAT = re.compile(r"epoch=(\d+),\s*loss=([0-9.eE+\-]+)")


def parse_log(path: Path):
    losses_per_epoch = {}
    with open(path, "r") as fh:
        text = fh.read()
    parts = text.replace("\r", "\n").split("\n")
    for ln in parts:
        m = PAT.search(ln)
        if m:
            ep = int(m.group(1))
            l = float(m.group(2))
            losses_per_epoch.setdefault(ep, []).append(l)
    return losses_per_epoch


def epoch_summary(losses_per_epoch):
    eps = sorted(losses_per_epoch.keys())
    means = [sum(losses_per_epoch[e]) / len(losses_per_epoch[e]) for e in eps]
    return eps, means


def main():
    # Paper-style: larger fonts, compact figure
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "axes.linewidth": 1.6,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=140)

    for name, prefix, label in RUNS:
        path = LOG_DIR / f"{prefix}.log"
        if not path.exists():
            print(f"  WARN: missing log {path}", file=sys.stderr)
            continue
        d = parse_log(path)
        if not d:
            print(f"  WARN: no losses parsed from {path}", file=sys.stderr)
            continue
        eps, means = epoch_summary(d)
        ax.plot(eps, means, label=label,
                color=COLORS.get(name, None),
                ls=LINESTYLES.get(name, "-"),
                lw=2.6)
        print(f"{label:25s}: epochs {eps[0]}..{eps[-1]}, final mean loss={means[-1]:.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("CHB-MIT Seizure Detection — Training Loss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
