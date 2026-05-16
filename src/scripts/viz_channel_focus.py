"""Channel-centric hyperedge membership figures (avg over edges).

The 3 hyperedges learn essentially redundant weightings, so we report a
single representative pattern: mean M across edges, broken down by
channel × class. Paper-quality styling — no harsh marker borders, larger
ticks, restrained palette.

Usage:
    python scripts/viz_channel_focus.py <dump.npz> <out_dir>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _lum(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_colors(vals, cmap, vmin, vmax):
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return ["white" if _lum(cmap_obj(norm(v))) < 0.55 else "#1a1a1a"
            for v in vals]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         16,
    "axes.titlesize":    18,
    "axes.labelsize":    17,
    "xtick.labelsize":   15,
    "ytick.labelsize":   15,
    "legend.fontsize":   14,
    "figure.titlesize":  18,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox":      "tight",
    "savefig.dpi":       220,
})


CH_TUSZ = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
           "F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"]

POS_TUSZ = {
    "FP1": (-0.31, 0.95), "FP2": (0.31, 0.95),
    "F7":  (-0.81, 0.59), "F3": (-0.51, 0.59), "FZ": (0, 0.51),
    "F4":  ( 0.51, 0.59), "F8": ( 0.81, 0.59),
    "T3":  (-0.99, 0),    "C3": (-0.51, 0),    "CZ": (0, 0),
    "C4":  ( 0.51, 0),    "T4": ( 0.99, 0),
    "T5":  (-0.81,-0.59), "P3": (-0.51,-0.59), "PZ": (0,-0.51),
    "P4":  ( 0.51,-0.59), "T6": ( 0.81,-0.59),
    "O1":  (-0.31,-0.95), "O2": ( 0.31,-0.95),
}

# Region mapping — matches viz_node_emb so axis tick labels stay consistent.
REGION_TUSZ = {
    "FP1":"frontal","FP2":"frontal","F3":"frontal","F4":"frontal",
    "F7":"frontal","F8":"frontal","FZ":"frontal",
    "C3":"central","C4":"central","CZ":"central",
    "P3":"parietal","P4":"parietal","PZ":"parietal",
    "T3":"temporal","T4":"temporal","T5":"temporal","T6":"temporal",
    "O1":"occipital","O2":"occipital",
}
REGION_COLORS = {
    "frontal":   "#4C72B0",
    "central":   "#55A868",
    "parietal":  "#8172B2",
    "temporal":  "#DD8452",
    "occipital": "#C44E52",
}


def _color_tick_labels(ax, axis, ch_names):
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lbl, name in zip(labels, ch_names):
        lbl.set_color(REGION_COLORS.get(REGION_TUSZ.get(name, "other"), "black"))
        lbl.set_fontweight("bold")


def head(ax):
    ax.add_patch(plt.Circle((0, 0), 1.05, fill=False, lw=1.2, color="#444"))
    ax.plot([-0.1, 0, 0.1], [1.0, 1.18, 1.0], color="#444", lw=1.0)
    ax.plot([-1.05, -1.12, -1.05], [0.1, 0, -0.1], color="#444", lw=1.0)
    ax.plot([ 1.05,  1.12,  1.05], [0.1, 0, -0.1], color="#444", lw=1.0)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")


def topomap(ax, vals, ch_names, pos, title, cmap, vmin, vmax, text="auto"):
    head(ax)
    xs = [pos[c][0] for c in ch_names]; ys = [pos[c][1] for c in ch_names]
    sc = ax.scatter(xs, ys, c=vals, s=1500, cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0, zorder=3)
    tcs = (_text_colors(vals, cmap, vmin, vmax) if text == "auto"
           else [text] * len(vals))
    for c, x, y, tc in zip(ch_names, xs, ys, tcs):
        ax.text(x, y, c, fontsize=15, fontweight="bold",
                ha="center", va="center", color=tc, zorder=4)
    ax.set_title(title, pad=10)
    return sc


def fig_channel_topo(M, y, out_dir):
    """Edge-averaged channel membership, seizure vs non-seizure."""
    pos = (y == 1); neg = ~pos
    M_e = M.astype(np.float32).mean(axis=-1)
    m_pos = M_e[pos].mean(axis=(0, 1))
    m_neg = M_e[neg].mean(axis=(0, 1))
    vmin = min(m_pos.min(), m_neg.min()); vmax = max(m_pos.max(), m_neg.max())
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2))
    sc = topomap(axes[0], m_neg, CH_TUSZ, POS_TUSZ,
                 "Non-seizure", "viridis", vmin, vmax)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=13)
    sc = topomap(axes[1], m_pos, CH_TUSZ, POS_TUSZ,
                 "Seizure", "viridis", vmin, vmax)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=13)
    d = m_pos - m_neg
    v = max(abs(d).max(), 1e-6)
    sc = topomap(axes[2], d, CH_TUSZ, POS_TUSZ,
                 "Δ (seizure − non-seizure)", "RdBu_r", -v, v,
                 text="white")
    cb = fig.colorbar(sc, ax=axes[2], fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "channel_topo.png"))
    plt.close(fig)


def fig_channel_temporal(M, y, out_dir):
    """For seizure clips, mean M[t, n] → (N, T) heatmap."""
    pos = (y == 1)
    if not pos.any(): return
    M_e = M.astype(np.float32).mean(axis=-1)
    mat = M_e[pos].mean(axis=0).T
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    im = ax.imshow(mat, aspect="auto", cmap="rocket" if "rocket" in plt.colormaps() else "magma")
    ax.set_yticks(range(len(CH_TUSZ))); ax.set_yticklabels(CH_TUSZ)
    ax.set_xticks(range(0, mat.shape[1])); ax.set_xticklabels(range(0, mat.shape[1]))
    ax.set_xlabel("Time step (s)"); ax.set_ylabel("Channel")
    ax.set_title("Per-channel membership over time — seizure clips", pad=10)
    _color_tick_labels(ax, "y", CH_TUSZ)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Mean hyperedge membership", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "channel_temporal.png"))
    plt.close(fig)


def main(dump_path, out_dir):
    d = np.load(dump_path)
    M = d["M_last"]
    y = d["y_true"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"M {M.shape}  pos={y.sum()}/{y.size}")
    fig_channel_topo(M, y, out_dir)
    fig_channel_temporal(M, y, out_dir)
    # Delete legacy figures that we no longer keep.
    for stale in ("channel_bar.png", "channel_boxplot.png"):
        p = os.path.join(out_dir, stale)
        if os.path.exists(p): os.remove(p)
    print(f"saved 2 figures under {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python viz_channel_focus.py <dump.npz> <out_dir>")
    main(sys.argv[1], sys.argv[2])
