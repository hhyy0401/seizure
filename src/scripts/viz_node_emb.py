"""t-SNE / cosine plots of learnable node embeddings.

Usage:
    python scripts/viz_node_emb.py <ckpt_path> <out_dir>
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


def _darken(hex_color, factor=0.55):
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c * factor for c in rgb)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         18,
    "axes.titlesize":    22,
    "axes.labelsize":    22,
    "xtick.labelsize":   18,
    "ytick.labelsize":   18,
    "legend.fontsize":   18,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox":      "tight",
    "savefig.dpi":       240,
})


CH_TUSZ = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
           "F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"]

REGION_TUSZ = {
    "FP1":"frontal","FP2":"frontal","F3":"frontal","F4":"frontal",
    "F7":"frontal","F8":"frontal","FZ":"frontal",
    "C3":"central","C4":"central","CZ":"central",
    "P3":"parietal","P4":"parietal","PZ":"parietal",
    "T3":"temporal","T4":"temporal","T5":"temporal","T6":"temporal",
    "O1":"occipital","O2":"occipital",
}

# Seaborn "deep" — user-preferred palette.
REGION_COLORS = {
    "frontal":   "#4C72B0",  # blue
    "central":   "#55A868",  # green
    "parietal":  "#8172B2",  # purple
    "temporal":  "#DD8452",  # orange
    "occipital": "#C44E52",  # red
}
REGION_ORDER = ["frontal", "central", "parietal", "temporal", "occipital"]


def load_node_emb(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state", state.get("state_dict", state))
    matches = [k for k in sd if k.endswith("node_emb") or "node_emb" in k]
    if not matches:
        sys.exit(f"node_emb not found in ckpt keys: {list(sd.keys())[:10]}")
    return sd[matches[0]].detach().cpu().numpy()


def plot_tsne(emb2, title, out_path, ch_names=CH_TUSZ, regions=REGION_TUSZ):
    # Compact figure for paper single-column render; fonts oversized at source.
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    # Colored beads with darker rim for subtle raised look.
    fc = [REGION_COLORS.get(regions.get(c, "other"), "#888") for c in ch_names]
    ec = [_darken(c, 0.55) for c in fc]
    ax.scatter(emb2[:, 0], emb2[:, 1], s=440, c=fc,
               edgecolors=ec, linewidths=1.4, zorder=3)

    for i, c in enumerate(ch_names):
        ax.text(emb2[i, 0], emb2[i, 1], c, fontsize=10, fontweight="bold",
                ha="center", va="center", color="white", zorder=4)

    handles = [Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=REGION_COLORS[r],
                       markeredgecolor=_darken(REGION_COLORS[r], 0.55),
                       markeredgewidth=0.9, markersize=7, label=r)
               for r in REGION_ORDER]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handletextpad=0.35, borderpad=0.2, labelspacing=0.25,
              fontsize=11)
    ax.set_title(title, pad=8, fontsize=20)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.tick_params(axis="both", which="major", labelsize=16)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    emb = load_node_emb(ckpt_path)
    print(f"node_emb shape {emb.shape}")
    N = emb.shape[0]
    ch_names = CH_TUSZ if N == 19 else [f"ch{i}" for i in range(N)]
    regions = REGION_TUSZ if N == 19 else {c:"other" for c in ch_names}

    # t-SNE
    from sklearn.manifold import TSNE
    perp = max(2, min(5, N // 3))
    ts = TSNE(n_components=2, perplexity=perp, init="pca",
              random_state=0).fit_transform(emb)
    plot_tsne(ts, "t-SNE of learned node embeddings",
              os.path.join(out_dir, "node_emb_tsne.png"), ch_names, regions)

    # Cosine similarity heatmap
    nrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    cos = nrm @ nrm.T
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels(ch_names, rotation=90)
    ax.set_yticks(range(N)); ax.set_yticklabels(ch_names)
    ax.set_title("Cosine similarity of learned node embeddings", pad=10)
    # Color tick labels by region to match the t-SNE legend / topomap.
    for axis_labels, names in [(ax.get_xticklabels(), ch_names),
                               (ax.get_yticklabels(), ch_names)]:
        for lbl, name in zip(axis_labels, names):
            lbl.set_color(REGION_COLORS.get(regions.get(name, "other"), "black"))
            lbl.set_fontweight("bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Cosine similarity", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "node_emb_cosine.png"))
    plt.close(fig)

    # Drop legacy PCA figure if present.
    p = os.path.join(out_dir, "node_emb_pca.png")
    if os.path.exists(p): os.remove(p)

    print(f"saved 2 figures under {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python viz_node_emb.py <ckpt_path> <out_dir>")
    main(sys.argv[1], sys.argv[2])
