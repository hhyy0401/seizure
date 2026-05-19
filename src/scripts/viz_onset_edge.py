"""Onset-anchored 2-panel figure for one hyperedge.

Reads `dump_membership.py` output (with --dense, so dense_y is present),
picks a representative seizure clip whose seizure onset falls inside the
clip window, and draws:

    LEFT  : head topomap, spatial pattern at onset ± window (mean over t)
    RIGHT : (T, N) heatmap of M[:, :, edge_idx] for the full clip,
            with a vertical dashed line at the onset second.

Usage:
    python scripts/viz_onset_edge.py <dump.npz> <out_dir> \
        [--edge_idx 0] [--win 2] [--clip_idx -1]

`--clip_idx -1` (default) → auto-pick by sharpness; else use the given index.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_channel_focus import (
    CH_TUSZ, POS_TUSZ, REGION_TUSZ, REGION_COLORS,
    head, topomap, _color_tick_labels, _text_colors,
)

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         16,
    "axes.titlesize":    18,
    "axes.labelsize":    17,
    "xtick.labelsize":   15,
    "ytick.labelsize":   15,
    "figure.titlesize":  18,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox":      "tight",
    "savefig.dpi":       220,
})


def find_onset(dy_row):
    """First 0→1 transition (or first ictal second); -1 if never seizure."""
    idx = np.where(dy_row > 0)[0]
    return int(idx[0]) if len(idx) else -1


def find_offset(dy_row):
    """Last 1→0 transition (post-transition index); -1 if seizure never ends
    within the clip (e.g. spans entire clip or ends still ictal)."""
    diffs = np.diff(dy_row.astype(int))
    ends = np.where(diffs == -1)[0] + 1
    return int(ends[-1]) if len(ends) else -1


def find_transitions(dy_row):
    """Return arrays of 0→1 and 1→0 transition indices (post-transition idx)."""
    diffs = np.diff(dy_row.astype(int))
    starts = np.where(diffs ==  1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1
    return starts, ends


def categorize_clip(dy_row):
    """Return one of: 'spans', 'onset_only', 'offset_only', 'both', 'multi', 'none'."""
    starts, ends = find_transitions(dy_row)
    if dy_row.sum() == 0:                    return "none"
    if dy_row.all():                         return "spans"
    if len(starts) > 1 or len(ends) > 1:     return "multi"
    has_s, has_e = len(starts) > 0, len(ends) > 0
    if has_s and has_e: return "both"
    if has_s:           return "onset_only"
    return "offset_only"


def pick_transition_rep(M, y, y_prob, dense_y, edge_idx, kind, prob_min=0.5):
    """Pick best representative seizure clip in category `kind`.
    Score = model prob + edge attention sharpness."""
    sz = np.where(y == 1)[0]
    cands = [s for s in sz
             if categorize_clip(dense_y[s]) == kind and y_prob[s] > prob_min]
    if not cands: return None
    scores = []
    for s in cands:
        m = M[s, :, :, edge_idx].astype(np.float32)
        scores.append((s, float(y_prob[s]) + (m.max() - m.mean())))
    scores.sort(key=lambda x: -x[1])
    return int(scores[0][0])


def pick_clip(M, y, y_prob, dense_y, edge_idx, win, n_ch_expected):
    """Filter for clean onset-in-window clips, then rank by attention sharpness."""
    S, T, N, _ = M.shape
    if N != n_ch_expected:
        raise ValueError(f"M has {N} channels, expected {n_ch_expected}")

    onsets = np.array([find_onset(dense_y[s]) for s in range(S)])
    durations = dense_y.sum(axis=1)

    # Representative criteria:
    #  - clip-level seizure (y==1)
    #  - onset detected inside the window AND with room on both sides for win=±2
    #  - seizure does not start at the very first second (no pre-onset visible)
    #  - seizure does not span the whole clip (we want to see ramp-up)
    #  - 10 <= seizure duration <= 30 (out of 60s)
    #  - model got it right (y_prob > 0.5)
    mask = (
        (y == 1)
        & (onsets >= win)
        & (onsets <= T - win - 1)
        & (durations >= 10)
        & (durations <= 30)
        & (y_prob > 0.5)
    )
    cands = np.where(mask)[0]
    if len(cands) == 0:
        # Relax: drop duration bound, keep onset-in-window
        mask = (y == 1) & (onsets >= 1) & (onsets <= T - 2) & (y_prob > 0.5)
        cands = np.where(mask)[0]
        if len(cands) == 0:
            raise SystemExit("No clip satisfies onset filter; try a different ckpt.")

    # Rank by attention sharpness: peak − baseline of M[:, :, edge_idx]
    peak     = M[cands, :, :, edge_idx].astype(np.float32).max(axis=(1, 2))
    baseline = M[cands, :, :, edge_idx].astype(np.float32).mean(axis=(1, 2))
    sharpness = peak - baseline
    chosen = int(cands[np.argsort(-sharpness)[0]])
    print(f"chose clip {chosen}  (onset={onsets[chosen]}s, "
          f"duration={int(durations[chosen])}s, p={y_prob[chosen]:.3f}, "
          f"sharpness={sharpness[np.argsort(-sharpness)[0]]:.3f}; "
          f"out of {len(cands)} candidates)")
    return chosen, int(onsets[chosen])


def _topomap_on_dot(ax, vals, ch_names, pos, title, cmap, vmin, vmax,
                    dot_size=1500, label_size=15, title_size=20):
    """Topomap with channel labels ON the dot (adaptive text color)."""
    from viz_channel_focus import head as _head
    _head(ax)
    xs = [pos[c][0] for c in ch_names]
    ys = [pos[c][1] for c in ch_names]
    sc = ax.scatter(xs, ys, c=vals, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=1.2, edgecolors="white", zorder=3)
    tcs = _text_colors(vals, cmap, vmin, vmax)
    for c, x, y, tc in zip(ch_names, xs, ys, tcs):
        ax.text(x, y, c, fontsize=label_size, fontweight="bold",
                ha="center", va="center", color=tc, zorder=4)
    ax.set_title(title, pad=14, fontsize=title_size)
    return sc


def _stretch(arr, lo=5, hi=95):
    """Percentile-based vmin/vmax for low-variance data."""
    a = arr[np.isfinite(arr)]
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def _onset_window_mean(M, y, dense_y, edge_idx, win):
    """Mean of M[..., edge_idx] across all seizure clips, time-sliced to
    onset ± win seconds. Returns (N,)."""
    S, T, N, _ = M.shape
    onsets = np.array([find_onset(dense_y[s]) for s in range(S)])
    sz_idx = np.where((y == 1) & (onsets >= 0))[0]
    chunks = []
    for s in sz_idx:
        on = int(onsets[s])
        t0, t1 = max(0, on - win), min(T, on + win + 1)
        chunks.append(M[s, t0:t1, :, edge_idx].astype(np.float32).mean(axis=0))
    return np.stack(chunks).mean(axis=0), len(sz_idx)


def make_spatial_figure(M, y, dense_y, edge_idx, win, ch_names, ch_pos,
                        out_path):
    """3-panel topomap for one edge: non-seizure / seizure (onset±win) / Δ.
    Suptitle deliberately omits the edge index — caller encodes it in filename."""
    M = M.astype(np.float32)
    S, T, N, _ = M.shape
    nz_mask = (y == 0)
    spatial_nz = M[nz_mask, :, :, edge_idx].mean(axis=(0, 1))            # (N,)
    spatial_sz, n_sz = _onset_window_mean(M, y, dense_y, edge_idx, win)

    delta = spatial_sz - spatial_nz
    vmin = float(min(spatial_sz.min(), spatial_nz.min()))
    vmax = float(max(spatial_sz.max(), spatial_nz.max()))
    if vmax - vmin < 1e-4:
        vmin, vmax = vmin - 0.01, vmax + 0.01
    v = max(abs(delta.max()), abs(delta.min()), 1e-4)

    fig = plt.figure(figsize=(19, 8.0))
    gs = fig.add_gridspec(1, 3, wspace=0.30, top=0.86)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    DOT, LBL, TIT = 1050, 13, 22

    sc0 = _topomap_on_dot(
        ax0, spatial_nz, ch_names, ch_pos,
        f"Non-seizure  (n={int(nz_mask.sum())} clips)",
        "cividis", vmin, vmax,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    cb0 = fig.colorbar(sc0, ax=ax0, fraction=0.045, pad=0.04)
    cb0.set_label("Mean membership", fontsize=20)
    cb0.ax.tick_params(labelsize=18)

    sc1 = _topomap_on_dot(
        ax1, spatial_sz, ch_names, ch_pos,
        f"Seizure  (±{win}s of onset, n={n_sz})",
        "cividis", vmin, vmax,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.045, pad=0.04)
    cb1.set_label("Mean membership", fontsize=20)
    cb1.ax.tick_params(labelsize=18)

    sc2 = _topomap_on_dot(
        ax2, delta, ch_names, ch_pos,
        "Δ (seizure − non-seizure)", "RdBu_r", -v, v,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.045, pad=0.04)
    cb2.set_label("Δ membership", fontsize=20)
    cb2.ax.tick_params(labelsize=18)

    fig.suptitle("Hyperedge channel pattern: seizure vs non-seizure",
                 y=0.97, fontsize=22, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_edge_comparison_figure(M, y, dense_y, win, ch_names, ch_pos,
                                out_path):
    """Three head topomaps side-by-side, one per hyperedge, all showing the
    seizure-onset spatial pattern. If the three look different, hyperedges
    learned distinct spatial primitives → 'hyperedge effect' evidence."""
    M = M.astype(np.float32)
    E_h = M.shape[-1]
    spatials, n_sz = [], 0
    for e in range(E_h):
        s, n = _onset_window_mean(M, y, dense_y, e, win)
        spatials.append(s); n_sz = n
    # Shared color scale across edges so panels are directly comparable.
    vmin = float(min(s.min() for s in spatials))
    vmax = float(max(s.max() for s in spatials))
    if vmax - vmin < 1e-4:
        vmin, vmax = vmin - 0.01, vmax + 0.01

    fig = plt.figure(figsize=(6.5 * E_h, 8.0))
    gs = fig.add_gridspec(1, E_h, wspace=0.30, top=0.86)
    DOT, LBL, TIT = 1050, 13, 22
    for e in range(E_h):
        ax = fig.add_subplot(gs[0, e])
        sc = _topomap_on_dot(
            ax, spatials[e], ch_names, ch_pos,
            f"Hyperedge {e}", "cividis", vmin, vmax,
            dot_size=DOT, label_size=LBL, title_size=TIT)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cb.set_label("Mean membership", fontsize=20)
        cb.ax.tick_params(labelsize=18)
    fig.suptitle(f"Per-hyperedge spatial pattern at seizure onset  "
                 f"(±{win}s, n={n_sz})",
                 y=0.97, fontsize=22, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_edge_temporal_figure(M, y, dense_y, win_pre, win_post,
                              ch_names, out_path):
    """Stacked (E_h × 1) onset-aligned N×T heatmaps — one per hyperedge.
    Clips are time-aligned so x=0 is seizure onset. If the rows look
    different, hyperedges differ in temporal profile too."""
    M = M.astype(np.float32)
    S, T, N, E_h = M.shape
    onsets = np.array([find_onset(dense_y[s]) for s in range(S)])
    sz_idx = np.where((y == 1) & (onsets >= 0))[0]

    # Aligned window length = win_pre + 1 + win_post (relative seconds)
    L = win_pre + 1 + win_post
    rel_axis = np.arange(-win_pre, win_post + 1)
    # accumulators with valid-count for proper averaging
    sums  = np.zeros((E_h, L, N), dtype=np.float32)
    cnts  = np.zeros((E_h, L),    dtype=np.int32)
    for s in sz_idx:
        on = int(onsets[s])
        for li, rel in enumerate(rel_axis):
            t = on + rel
            if 0 <= t < T:
                sums[:, li, :] += M[s, t, :, :].transpose(1, 0)          # (E_h, N)
                cnts[:, li]   += 1
    cnts_safe = np.maximum(cnts, 1)[..., None]                            # (E_h,L,1)
    aligned = sums / cnts_safe                                            # (E_h, L, N)

    # Per-edge vmin/vmax via percentile stretch (consistent across edges)
    flat = aligned.reshape(-1)
    vmin, vmax = float(np.percentile(flat, 2)), float(np.percentile(flat, 98))
    if vmax - vmin < 1e-4:
        vmin, vmax = vmin - 0.01, vmax + 0.01

    fig, axes = plt.subplots(E_h, 1, figsize=(15, 4.4 * E_h),
                              sharex=True, sharey=True)
    if E_h == 1: axes = [axes]
    for e, ax in enumerate(axes):
        im = ax.imshow(aligned[e].T, aspect="auto", cmap="cividis",
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       extent=[rel_axis[0] - 0.5, rel_axis[-1] + 0.5,
                               N - 0.5, -0.5])
        ax.set_yticks(range(N)); ax.set_yticklabels(ch_names, fontsize=16)
        _color_tick_labels(ax, "y", ch_names)
        ax.axvline(0, color="#ff2d95", linestyle="--", lw=2.8, alpha=0.95)
        ax.set_ylabel(f"Hyperedge {e}\nchannel", fontsize=20)
        ax.tick_params(axis="x", labelsize=18)
        if e == E_h - 1:
            ax.set_xlabel("Time relative to seizure onset (s)", fontsize=22)
    fig.suptitle(f"Onset-aligned spatio-temporal membership "
                 f"(n={len(sz_idx)} seizure clips)",
                 y=0.995, fontsize=22, fontweight="bold")
    # One shared colorbar on the right
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Mean membership (p2–p98)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_heatmap_figure(clip_M, onset_t, ch_names, out_path):
    """Single-clip (N, T) heatmap with onset line."""
    T, N = clip_M.shape
    heat = clip_M.T.astype(np.float32)                                   # (N, T)
    vmin_h, vmax_h = _stretch(heat, 2, 98)
    if vmax_h - vmin_h < 1e-4:
        vmin_h, vmax_h = vmin_h - 0.01, vmax_h + 0.01

    fig = plt.figure(figsize=(15.0, 9.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(heat, aspect="auto", cmap="cividis",
                   vmin=vmin_h, vmax=vmax_h, interpolation="nearest")
    ax.set_yticks(range(N))
    ax.set_yticklabels(ch_names, fontsize=18)
    _color_tick_labels(ax, "y", ch_names)
    xtick = np.arange(0, T, max(1, T // 6))
    ax.set_xticks(xtick); ax.set_xticklabels(xtick, fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=22)
    ax.set_ylabel("Channel", fontsize=22)
    ax.set_title("Spatio-temporal membership — representative clip",
                 fontsize=22, pad=16, fontweight="bold")
    ax.axvline(onset_t, color="#ff2d95", linestyle="--", lw=2.8, alpha=0.95)
    ax.text(onset_t + 0.6, 0.6, f"onset = {onset_t}s",
            color="#ff2d95", fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="#ff2d95",
                      alpha=0.95, pad=4, linewidth=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Mean membership (p2–p98)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def _aligned_average(M, y, dense_y, edge_idx, anchor_fn, win_pre, win_post):
    """Generic anchor-aligned aggregator. Returns (aligned (L, N), rel_axis,
    n_clips_used)."""
    S, T, N, _ = M.shape
    anchors = np.array([anchor_fn(dense_y[s]) for s in range(S)])
    sz_idx = np.where((y == 1) & (anchors >= 0))[0]
    L = win_pre + 1 + win_post
    rel_axis = np.arange(-win_pre, win_post + 1)
    sums = np.zeros((L, N), dtype=np.float32)
    cnts = np.zeros((L,),   dtype=np.int32)
    for s in sz_idx:
        a = int(anchors[s])
        for li, rel in enumerate(rel_axis):
            t = a + rel
            if 0 <= t < T:
                sums[li] += M[s, t, :, edge_idx]
                cnts[li] += 1
    aligned = sums / np.maximum(cnts, 1)[..., None]                      # (L, N)
    return aligned, rel_axis, len(sz_idx)


def make_aligned_heatmap_pdf(M, y, dense_y, edge_idx, anchor_fn, anchor_label,
                              transition_label, win_pre, win_post,
                              ch_names, out_path):
    """Anchor-aligned N×T heatmap, averaged across all seizure clips with
    valid anchor. `anchor_label` ∈ {"onset", "offset"}; `transition_label`
    ∈ {"0→1", "1→0"}."""
    M = M.astype(np.float32)
    N = M.shape[2]
    aligned, rel_axis, n_used = _aligned_average(
        M, y, dense_y, edge_idx, anchor_fn, win_pre, win_post)
    heat = aligned.T                                                      # (N, L)
    vmin, vmax = _stretch(heat, 2, 98)
    if vmax - vmin < 1e-4: vmin, vmax = vmin - 0.005, vmax + 0.005

    fig = plt.figure(figsize=(15.0, 9.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(heat, aspect="auto", cmap="cividis",
                   vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=[rel_axis[0] - 0.5, rel_axis[-1] + 0.5,
                           N - 0.5, -0.5])
    ax.set_yticks(range(N))
    ax.set_yticklabels(ch_names, fontsize=18)
    _color_tick_labels(ax, "y", ch_names)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel(f"Time relative to seizure {anchor_label} (s)", fontsize=22)
    ax.set_ylabel("Channel", fontsize=22)
    ax.set_title(f"{anchor_label.capitalize()}-aligned ({transition_label}) "
                 f"spatio-temporal membership  (n={n_used})",
                 fontsize=22, pad=16, fontweight="bold")
    ax.axvline(0, color="#ff2d95", linestyle="--", lw=2.8, alpha=0.95)
    ax.text(0.5, 0.6, f"{anchor_label} (t = 0)",
            color="#ff2d95", fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="#ff2d95",
                      alpha=0.95, pad=4, linewidth=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Mean membership (p2–p98)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# Backwards-compat alias used elsewhere in this file (paper figure pipeline).
def make_single_edge_temporal_figure(M, y, dense_y, edge_idx,
                                     win_pre, win_post, ch_names, out_path):
    make_aligned_heatmap_pdf(
        M, y, dense_y, edge_idx, find_onset, "onset", "0→1",
        win_pre, win_post, ch_names, out_path)


def make_pre_post_spatial_figure(M, y, dense_y, edge_idx, win,
                                 ch_names, ch_pos, out_path):
    """4-panel topomap: non-seizure | pre-onset (-win..0) | post-onset (0..+win)
    | Δ(post − pre). Within-seizure pre/post comparison removes inter-clip
    variance and isolates the spatial shift at seizure onset."""
    M = M.astype(np.float32)
    S, T, N, _ = M.shape
    nz_mask = (y == 0)
    spatial_nz = M[nz_mask, :, :, edge_idx].mean(axis=(0, 1))            # (N,)

    onsets = np.array([find_onset(dense_y[s]) for s in range(S)])
    sz_idx = np.where((y == 1) & (onsets >= win) & (onsets <= T - win - 1))[0]

    pre_chunks, post_chunks = [], []
    for s in sz_idx:
        on = int(onsets[s])
        pre_chunks.append( M[s, max(0, on - win):on,           :, edge_idx].mean(axis=0))
        post_chunks.append(M[s, on:min(T, on + win + 1),       :, edge_idx].mean(axis=0))
    spatial_pre  = np.stack(pre_chunks).mean(axis=0)
    spatial_post = np.stack(post_chunks).mean(axis=0)
    delta = spatial_post - spatial_pre

    vmin = float(min(spatial_nz.min(), spatial_pre.min(), spatial_post.min()))
    vmax = float(max(spatial_nz.max(), spatial_pre.max(), spatial_post.max()))
    if vmax - vmin < 1e-4: vmin, vmax = vmin - 0.005, vmax + 0.005
    v = max(abs(delta.max()), abs(delta.min()), 1e-4)

    fig = plt.figure(figsize=(26, 8.0))
    gs = fig.add_gridspec(1, 4, wspace=0.32, top=0.86)
    DOT, LBL, TIT = 1050, 13, 20

    sc0 = _topomap_on_dot(
        fig.add_subplot(gs[0, 0]), spatial_nz, ch_names, ch_pos,
        f"Non-seizure  (n={int(nz_mask.sum())})", "cividis", vmin, vmax,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    fig.colorbar(sc0, ax=fig.axes[-1], fraction=0.045, pad=0.04).ax.tick_params(labelsize=16)

    sc1 = _topomap_on_dot(
        fig.add_subplot(gs[0, 1]), spatial_pre, ch_names, ch_pos,
        f"Pre-onset  (−{win}..0 s,  n={len(sz_idx)})", "cividis", vmin, vmax,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    fig.colorbar(sc1, ax=fig.axes[-1], fraction=0.045, pad=0.04).ax.tick_params(labelsize=16)

    sc2 = _topomap_on_dot(
        fig.add_subplot(gs[0, 2]), spatial_post, ch_names, ch_pos,
        f"Post-onset  (0..+{win} s,  n={len(sz_idx)})", "cividis", vmin, vmax,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    fig.colorbar(sc2, ax=fig.axes[-1], fraction=0.045, pad=0.04).ax.tick_params(labelsize=16)

    sc3 = _topomap_on_dot(
        fig.add_subplot(gs[0, 3]), delta, ch_names, ch_pos,
        "Δ (post − pre)", "RdBu_r", -v, v,
        dot_size=DOT, label_size=LBL, title_size=TIT)
    fig.colorbar(sc3, ax=fig.axes[-1], fraction=0.045, pad=0.04).ax.tick_params(labelsize=16)

    fig.suptitle("Hyperedge spatial pattern: non-seizure vs pre vs post onset",
                 y=0.97, fontsize=22, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_clip_gallery_figure(M, y, y_prob, dense_y, edge_idx,
                             ch_names, out_path, n_per_group=4,
                             clip_filter_mask=None, title_suffix=""):
    """Gallery of clips at three confidence levels:
       row 1 = easy   (y=1, p > 0.9)
       row 2 = hard   (y=1, p in [0.5, 0.7])
       row 3 = missed (y=1, p < 0.5)
    Each cell: small N×T heatmap with red(onset) + blue(offset) dashed lines.
    Pass `clip_filter_mask` (S,) bool to restrict to a transition category."""
    rng = np.random.default_rng(0)
    M = M.astype(np.float32)
    S, T, N, _ = M.shape
    sz = (y == 1)
    if clip_filter_mask is not None:
        sz = sz & clip_filter_mask

    groups = [
        ("Easy  (p>0.9)",     np.where(sz & (y_prob > 0.9))[0]),
        ("Hard  (0.5≤p≤0.7)", np.where(sz & (y_prob >= 0.5) & (y_prob <= 0.7))[0]),
        ("Missed (p<0.5)",    np.where(sz & (y_prob < 0.5))[0]),
    ]

    def pick(arr, k):
        if len(arr) <= k: return arr
        return rng.choice(arr, size=k, replace=False)

    selected = [(name, pick(idxs, n_per_group)) for name, idxs in groups]
    nrows, ncols = len(selected), n_per_group
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.6 * nrows),
                              sharey=True)
    if nrows == 1: axes = [axes]
    for r, (name, idxs) in enumerate(selected):
        for c in range(ncols):
            ax = axes[r][c] if nrows > 1 else axes[c]
            if c >= len(idxs):
                ax.axis("off"); continue
            s = int(idxs[c])
            heat = M[s, :, :, edge_idx].T
            vmin, vmax = _stretch(heat, 5, 95)
            if vmax - vmin < 1e-4: vmin, vmax = vmin - 0.005, vmax + 0.005
            ax.imshow(heat, aspect="auto", cmap="cividis",
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            starts, ends = find_transitions(dense_y[s])
            # Draw transitions: red dashed = onset (0→1), blue dashed = offset (1→0)
            for x in starts:
                ax.axvline(x, color="white", linestyle="-",  lw=2.6, alpha=0.95)
                ax.axvline(x, color="#e60000", linestyle="--", lw=1.8, alpha=1.0)
            for x in ends:
                ax.axvline(x, color="white", linestyle="-",  lw=2.6, alpha=0.95)
                ax.axvline(x, color="#0066ff", linestyle="--", lw=1.8, alpha=1.0)
            ax.set_yticks(range(N))
            ax.set_yticklabels(ch_names if c == 0 else [], fontsize=11)
            if c == 0: _color_tick_labels(ax, "y", ch_names)
            ax.tick_params(axis="x", labelsize=11)
            # Compact title with all transitions
            tlabel = " ".join(
                [f"on={int(x)}s" for x in starts] +
                [f"off={int(x)}s" for x in ends])
            ax.set_title(f"clip {s}  p={y_prob[s]:.2f}  {tlabel}", fontsize=11)
            if r == nrows - 1: ax.set_xlabel("Time (s)", fontsize=14)
            if c == 0:
                ax.set_ylabel(f"{name}\nchannel", fontsize=14, fontweight="bold")
    fig.suptitle(f"Clip gallery — model confidence sweep{title_suffix}",
                 y=1.005, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}  (n_avail per row: "
          f"{[len(idxs) for _, idxs in selected]})")


def _draw_onset_line(ax, x):
    """Bright red dashed onset line, white halo for visibility on dark cmaps."""
    ax.axvline(x, color="white", linestyle="-",  lw=5.2, alpha=0.95, zorder=5)
    ax.axvline(x, color="#e60000", linestyle="--", lw=3.6, alpha=1.0, zorder=6)


def make_clip_heatmap_pdf(clip_M, onset_t, ch_names, out_path):
    """Single-clip N×T heatmap, cividis (blue→yellow), strong percentile
    stretch (p5/p95) so the onset boundary is visible as dark→bright shift.
    Title-less; metadata lives in filename."""
    T, N = clip_M.shape
    heat = clip_M.T.astype(np.float32)                                   # (N, T)
    vmin, vmax = _stretch(heat, 5, 95)
    if vmax - vmin < 1e-4: vmin, vmax = vmin - 0.01, vmax + 0.01

    fig = plt.figure(figsize=(15.0, 9.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(heat, aspect="auto", cmap="cividis",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(N))
    ax.set_yticklabels(ch_names, fontsize=24)
    _color_tick_labels(ax, "y", ch_names)
    xtick = np.arange(0, T, max(1, T // 6))
    ax.set_xticks(xtick); ax.set_xticklabels(xtick, fontsize=26)
    ax.set_xlabel("Time (s)", fontsize=30)
    ax.set_ylabel("Channel", fontsize=30)
    _draw_onset_line(ax, onset_t)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Mean membership  (p5–p95)", fontsize=24)
    cb.ax.tick_params(labelsize=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_clip_topomap_pdf(clip_M, onset_t, win, ch_names, ch_pos, out_path,
                          nonsz_spatial):
    """3-panel head topomap for one clip:
        non-seizure (population baseline) | pre-onset | post-onset
    All three share the same color scale so panels are directly comparable.
    `nonsz_spatial` is a (N,) baseline computed once over non-sz test clips."""
    T, N = clip_M.shape
    pre  = clip_M[max(0, onset_t - win):onset_t,                  :].astype(np.float32).mean(axis=0)
    post = clip_M[onset_t:min(T, onset_t + win + 1),              :].astype(np.float32).mean(axis=0)

    panels = [
        ("(a) Non-seizure", nonsz_spatial),
        ("(b) Pre-onset",   pre),
        ("(c) Post-onset",  post),
    ]
    vmin = float(min(p.min() for _, p in panels))
    vmax = float(max(p.max() for _, p in panels))
    if vmax - vmin < 1e-4: vmin, vmax = vmin - 0.01, vmax + 0.01

    fig = plt.figure(figsize=(24, 9.0))
    gs = fig.add_gridspec(1, 3, wspace=0.30, top=0.92)
    DOT, LBL, TIT = 2000, 22, 32
    last_sc = None
    for i, (ttl, vec) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        last_sc = _topomap_on_dot(
            ax, vec, ch_names, ch_pos, ttl, "cividis", vmin, vmax,
            dot_size=DOT, label_size=LBL, title_size=TIT)
        cb = fig.colorbar(last_sc, ax=ax, fraction=0.045, pad=0.04)
        cb.set_label("Mean membership", fontsize=28)
        cb.ax.tick_params(labelsize=26)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def _seizure_segments(dy_row):
    """Yield (start, end, label) contiguous segments. label ∈ {seizure, non-seizure}."""
    T = len(dy_row)
    cur = int(dy_row[0] > 0)
    start = 0
    for t in range(1, T):
        v = int(dy_row[t] > 0)
        if v != cur:
            yield (start, t, "seizure" if cur else "non-seizure")
            cur, start = v, t
    yield (start, T, "seizure" if cur else "non-seizure")


def _annotate_segments(ax, dy_row, T, ymin):
    """Place 'seizure' / 'non-seizure' text labels at top of axes per segment.
    No background box; placed right above the data area. Skip segments
    shorter than 3s to avoid clutter."""
    seg_color = {"seizure": "#b30000", "non-seizure": "#264fa5"}
    for s_t, e_t, label in _seizure_segments(dy_row):
        if e_t - s_t < 3: continue
        mid = (s_t + e_t) / 2.0
        ax.text(mid, ymin, label, fontsize=28, fontweight="bold",
                color=seg_color[label], ha="center", va="bottom",
                zorder=10, clip_on=False)


def make_clip_heatmap_with_transitions_pdf(clip_M, dense_y_row,
                                            ch_names, out_path):
    """Single-clip heatmap with ALL visible transition lines (red=onset 0→1,
    blue=offset 1→0) AND 'seizure' / 'non-seizure' text per segment.
    cividis, p5–p95 stretch. Title-less."""
    T, N = clip_M.shape
    heat = clip_M.T.astype(np.float32)
    # Fixed range matching topo3 colorbar for cross-figure consistency.
    vmin, vmax = 0.48, 0.52

    starts, ends = find_transitions(dense_y_row)

    fig = plt.figure(figsize=(16.0, 10.5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(heat, aspect="auto", cmap="cividis",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(N)); ax.set_yticklabels(ch_names, fontsize=28)
    _color_tick_labels(ax, "y", ch_names)
    xtick = np.arange(0, T, max(1, T // 6))
    ax.set_xticks(xtick); ax.set_xticklabels(xtick, fontsize=30)
    ax.set_xlabel("Time (s)", fontsize=34)
    ax.set_ylabel("Channel", fontsize=34)
    for s in starts: _draw_onset_line(ax, s)
    for e in ends:
        ax.axvline(e, color="white",  linestyle="-",  lw=5.2, alpha=0.95, zorder=5)
        ax.axvline(e, color="#0066ff", linestyle="--", lw=3.6, alpha=1.0, zorder=6)
    # 'seizure' / 'non-seizure' segment labels right above the data area.
    _annotate_segments(ax, dense_y_row, T, ymin=-0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02,
                      shrink=0.55, aspect=14)
    cb.set_ticks([0.48, 0.50, 0.52])
    cb.set_ticklabels(["0.48", "0.50", "0.52"])
    cb.set_label("Mean membership", fontsize=28, labelpad=12)
    cb.ax.tick_params(labelsize=26, length=8, width=1.4, pad=6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_clip_topomap_around_t_pdf(clip_M, anchor_t, win, ch_names, ch_pos,
                                   out_path, nonsz_spatial, anchor_label):
    """3-panel topo: non-seizure | pre-anchor (−win..0) | post-anchor (0..+win).
    `anchor_label` is e.g. 'onset' or 'offset' (only used in panel titles).
    Time window `win` is documented in the caption / main text, not in figure.
    Single shared colorbar on the far right."""
    T, N = clip_M.shape
    pre  = clip_M[max(0, anchor_t - win):anchor_t,           :].astype(np.float32).mean(axis=0)
    post = clip_M[anchor_t:min(T, anchor_t + win + 1),       :].astype(np.float32).mean(axis=0)
    # Unified labels across all clips (matches EvoBrain Fig 6 narrative).
    panels = [
        ("(a) Normal",        nonsz_spatial),
        ("(b) Pre-seizure",   pre),
        ("(c) Post-seizure",  post),
    ]
    # Fixed colorbar range across ALL clips for visual consistency. Values
    # outside [0.48, 0.52] saturate at the colormap extremes.
    vmin, vmax = 0.48, 0.52

    fig = plt.figure(figsize=(22, 8.6))
    gs = fig.add_gridspec(1, 3, wspace=0.12, top=0.95,
                          left=0.02, right=0.92)
    DOT, LBL, TIT = 1750, 20, 30
    last_sc = None
    axes = []
    for i, (ttl, vec) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i]); axes.append(ax)
        # Empty title to topomap fn; we put the caption BELOW instead.
        last_sc = _topomap_on_dot(
            ax, vec, ch_names, ch_pos, "", "cividis", vmin, vmax,
            dot_size=DOT, label_size=LBL, title_size=TIT)
        ax.set_title(ttl, fontsize=TIT, pad=8,
                     fontweight="bold", y=-0.12)
    # Fixed 3 ticks at 0.48 / 0.50 / 0.52 (membership baseline ~0.5, signal
    # range ±0.03). The data range itself can extend slightly beyond these
    # — the ticks are labels, not clipping bounds.
    cb = fig.colorbar(last_sc, ax=axes, fraction=0.018, pad=0.025,
                       shrink=0.55, aspect=14)
    cb.set_ticks([0.48, 0.50, 0.52])
    cb.set_ticklabels(["0.48", "0.50", "0.52"])
    cb.set_label("Mean membership", fontsize=26, labelpad=12)
    cb.ax.tick_params(labelsize=26, length=8, width=1.4, pad=6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_transition_rep_figures(M, y, y_prob, dense_y, edge_idx, ch_names,
                                 ch_pos, win, out_dir, seed_tag):
    """One representative clip per transition category. Each gets heatmap +
    3-panel topomap. Anchor for topomap:
       - 'both', 'onset_only' → anchor at onset
       - 'offset_only'        → anchor at offset (no onset in clip)"""
    nonsz_mask = (y == 0)
    nonsz_spatial = (M[nonsz_mask, :, :, edge_idx]
                     .astype(np.float32).mean(axis=(0, 1)))               # (N,)

    for kind, label, anchor_fn in [
        ("both",        "0to1to0",   lambda dy: find_transitions(dy)[0][0]),
        ("onset_only",  "0to1",      lambda dy: find_transitions(dy)[0][0]),
        ("offset_only", "1to0",      lambda dy: find_transitions(dy)[1][0]),
    ]:
        cid = pick_transition_rep(M, y, y_prob, dense_y, edge_idx, kind)
        if cid is None:
            print(f"[skip] no qualifying clip for kind={kind}"); continue
        anchor_t = int(anchor_fn(dense_y[cid]))
        p = float(y_prob[cid])
        clip_M = M[cid, :, :, edge_idx].astype(np.float32)
        anchor_label = "onset" if kind != "offset_only" else "offset"

        base = f"{seed_tag}_edge{edge_idx}_{label}_clip{cid}_p{p:.2f}_anchor{anchor_t}s"
        make_clip_heatmap_with_transitions_pdf(
            clip_M, dense_y[cid], ch_names,
            os.path.join(out_dir, f"{base}_heatmap.pdf"))
        make_clip_topomap_around_t_pdf(
            clip_M, anchor_t, win, ch_names, ch_pos,
            os.path.join(out_dir, f"{base}_topo3.pdf"),
            nonsz_spatial, anchor_label)


def make_aligned_topomap_pdf(M, y, dense_y, edge_idx, ch_names, ch_pos,
                              anchor_fn, anchor_label, transition_label,
                              bins, out_path):
    """Anchor-aligned population-average head topomap, binned into time
    windows. `bins` = list of (lo, hi) second pairs (relative to anchor,
    inclusive lo, exclusive hi); each becomes one head plot."""
    M = M.astype(np.float32)
    S, T, N, _ = M.shape
    anchors = np.array([anchor_fn(dense_y[s]) for s in range(S)])
    sz_idx = np.where((y == 1) & (anchors >= 0))[0]

    panels = []
    for lo, hi in bins:
        sums = np.zeros(N, dtype=np.float32)
        cnts = 0
        for s in sz_idx:
            a = int(anchors[s])
            for rel in range(lo, hi):
                t = a + rel
                if 0 <= t < T:
                    sums += M[s, t, :, edge_idx]
                    cnts += 1
        panels.append((f"[{lo:+d}, {hi:+d}) s", sums / max(cnts, 1)))

    vmin = float(min(v.min() for _, v in panels))
    vmax = float(max(v.max() for _, v in panels))
    if vmax - vmin < 1e-4: vmin, vmax = vmin - 0.01, vmax + 0.01

    nb = len(panels)
    fig = plt.figure(figsize=(7.5 * nb, 8.6))
    gs = fig.add_gridspec(1, nb, wspace=0.30, top=0.90)
    DOT, LBL, TIT = 1800, 16, 24
    for i, (ttl, vec) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        sc = _topomap_on_dot(
            ax, vec, ch_names, ch_pos, ttl, "cividis", vmin, vmax,
            dot_size=DOT, label_size=LBL, title_size=TIT)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cb.set_label("Mean membership", fontsize=22)
        cb.ax.tick_params(labelsize=20)
    fig.suptitle(f"Population-average channel pattern around seizure "
                 f"{anchor_label} ({transition_label})  (n={len(sz_idx)})",
                 y=0.97, fontsize=24, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# Backwards-compat alias for onset.
def make_onset_aligned_topomap_pdf(M, y, dense_y, edge_idx, ch_names, ch_pos,
                                    win_pre, win_post, bins, out_path):
    make_aligned_topomap_pdf(
        M, y, dense_y, edge_idx, ch_names, ch_pos,
        find_onset, "onset", "0→1", bins, out_path)


def make_per_clip_pair(M, y, y_prob, dense_y, edge_idx, ch_names, ch_pos,
                       clip_ids, win, out_dir, seed_tag):
    """For each user-specified clip: write transition-aware heatmap (red/blue
    lines + seizure/non-seizure labels) + 3-panel topomap with the right
    anchor (onset for onset_only/both, offset for offset_only).
    Filename encodes category + p + anchor."""
    nonsz_mask = (y == 0)
    nonsz_spatial = (M[nonsz_mask, :, :, edge_idx]
                     .astype(np.float32).mean(axis=(0, 1)))               # (N,)

    for cid in clip_ids:
        kind = categorize_clip(dense_y[cid])
        if kind == "none":
            print(f"[skip] clip {cid}: no seizure in dense_y"); continue
        starts, ends = find_transitions(dense_y[cid])
        if kind == "offset_only":
            anchor_t, anchor_label, kind_tag = int(ends[0]),   "offset", "1to0"
        elif kind == "both":
            anchor_t, anchor_label, kind_tag = int(starts[0]), "onset",  "0to1to0"
        elif kind == "onset_only":
            anchor_t, anchor_label, kind_tag = int(starts[0]), "onset",  "0to1"
        elif kind == "multi":
            anchor_t, anchor_label, kind_tag = int(starts[0]), "onset",  "multi"
        else:  # 'spans' — no transition in clip; use mid as a fallback anchor
            anchor_t, anchor_label, kind_tag = M.shape[1] // 2, "mid",  "spans"

        p = float(y_prob[cid])
        clip_M = M[cid, :, :, edge_idx].astype(np.float32)
        base = (f"{seed_tag}_edge{edge_idx}_{kind_tag}_clip{cid}"
                f"_p{p:.2f}_anchor{anchor_t}s")
        make_clip_heatmap_with_transitions_pdf(
            clip_M, dense_y[cid], ch_names,
            os.path.join(out_dir, f"{base}_heatmap.pdf"))
        make_clip_topomap_around_t_pdf(
            clip_M, anchor_t, win, ch_names, ch_pos,
            os.path.join(out_dir, f"{base}_topo3.pdf"),
            nonsz_spatial, anchor_label)


def _draw_head_edges(ax, edge_w, ch_names, ch_pos, vmin, vmax, top_k=10,
                     cmap_name="cividis", node_size=1750, label_size=20,
                     edge_lw=4.5):
    """Head topomap with top-k strongest pairwise edges drawn as colored lines.
    EvoBrain Fig 6 style: blue nodes labeled with channel names, edge color
    on cividis (yellow=strong, dark=weak)."""
    import matplotlib.colors as mcolors
    from viz_channel_focus import head as _head
    _head(ax)
    xs = [ch_pos[c][0] for c in ch_names]
    ys = [ch_pos[c][1] for c in ch_names]
    N = len(ch_names)

    triu_i, triu_j = np.triu_indices(N, k=1)
    weights = edge_w[triu_i, triu_j]
    top = np.argsort(-weights)[:top_k]
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Draw weakest first so strongest sit on top
    for rank in sorted(top, key=lambda r: weights[r]):
        i, j = triu_i[rank], triu_j[rank]
        c = cmap(norm(weights[rank]))
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color=c, lw=edge_lw, zorder=2, alpha=0.95,
                solid_capstyle="round")

    # Nodes — small dark-blue circles, channel label inside.
    ax.scatter(xs, ys, s=node_size, c="#2e3a8c", linewidths=0.8,
               edgecolors="white", zorder=3)
    for c, x, y in zip(ch_names, xs, ys):
        ax.text(x, y, c, fontsize=label_size, fontweight="bold",
                ha="center", va="center", color="white", zorder=4)


def _pairwise_edges_from_membership(m_window, baseline, mode="dev"):
    """m_window: (T_win, N, E_h) membership averaged over a small time window.
    baseline: (N, E_h) non-seizure baseline. Returns (N, N) symmetric weight
    matrix (diagonal zeroed) suitable for top-k edge picking.

    mode='dev' : co-membership over baseline-subtracted positive deviation.
                 Captures abnormal co-recruitment (what differs from rest).
    """
    m = m_window.mean(axis=0)                                  # (N, E_h)
    if mode == "dev":
        dev = np.maximum(m - baseline, 0.0)
    else:
        dev = m
    # (N, E_h) → (N, N) via dev_i · dev_j summed over hyperedges
    edge_w = dev @ dev.T                                       # (N, N)
    np.fill_diagonal(edge_w, 0.0)
    return edge_w


def make_evobrain_style_figure(M_clip, baseline, snapshot_times, win,
                               ch_names, ch_pos, out_path,
                               panel_title=None, top_k=10, norm_mode="local"):
    """3 head topomaps side by side in EvoBrain Fig 6 style.
    M_clip: (T, N, E_h) membership for one clip.
    baseline: (N, E_h) non-seizure population baseline.
    snapshot_times: list of 3 time indices (seconds). Each draws the mean of
                    M over [t-win/2, t+win/2] window.
    norm_mode='local'  → each panel's top-k spans its own [weak, strong]
                          (matches EvoBrain Fig 6 — within-panel relative).
    norm_mode='shared' → one scale across the 3 snapshots (absolute strength)."""
    import matplotlib.colors as mcolors
    T, N, E_h = M_clip.shape
    edge_mats = []
    for t in snapshot_times:
        half = win // 2
        t0, t1 = max(0, t - half), min(T, t + half + 1)
        ew = _pairwise_edges_from_membership(M_clip[t0:t1], baseline, mode="dev")
        edge_mats.append(ew)

    triu_i, triu_j = np.triu_indices(N, k=1)

    if norm_mode == "shared":
        top_vals = []
        for em in edge_mats:
            w = em[triu_i, triu_j]
            top_vals.extend(sorted(w, reverse=True)[:top_k])
        gv = (float(min(top_vals)), float(max(top_vals)))
        if gv[1] - gv[0] < 1e-6: gv = (gv[0] - 1e-3, gv[1] + 1e-3)
        scales = [gv] * len(edge_mats)
    else:  # local
        scales = []
        for em in edge_mats:
            w = em[triu_i, triu_j]
            top = sorted(w, reverse=True)[:top_k]
            lo, hi = float(min(top)), float(max(top))
            if hi - lo < 1e-6: lo, hi = lo - 1e-3, hi + 1e-3
            scales.append((lo, hi))

    fig = plt.figure(figsize=(18.5, 6.4))
    gs = fig.add_gridspec(1, 3, wspace=0.10, top=0.86, left=0.02, right=0.91)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, t, em, (lo, hi) in zip(axes, snapshot_times, edge_mats, scales):
        _draw_head_edges(ax, em, ch_names, ch_pos, lo, hi, top_k=top_k,
                         node_size=380, label_size=7.5)
        ax.set_title(f"t = {t} s", fontsize=20, pad=10)

    # Right-side colorbar: shared-mode uses absolute range; local-mode is
    # symbolic Weak→Strong (no absolute number is meaningful across panels).
    lo_cbar, hi_cbar = (scales[0] if norm_mode == "shared" else (0.0, 1.0))
    sm = plt.cm.ScalarMappable(
        cmap="cividis", norm=mcolors.Normalize(vmin=lo_cbar, vmax=hi_cbar))
    cb = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.03)
    cb.set_ticks([lo_cbar, hi_cbar])
    cb.set_ticklabels(["Weak", "Strong"])
    cb.ax.tick_params(labelsize=18)

    if panel_title:
        fig.suptitle(panel_title, y=0.97, fontsize=22, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# --- EvoBrain Fig 6 (a)(b)(c) rightmost-panel edges (manually traced from
# the published Figure 6) — used by `make_evobrain_repro_figure` to reproduce
# the same graph topology on our head template / cividis colormap.
EVOBRAIN_FIG6_EDGES = {
    "(a) Normal": [
        ("F3", "FZ"),  ("FZ", "F4"), ("F4", "F8"), ("F3", "C3"),
        ("C3", "CZ"),  ("FZ", "CZ"), ("CZ", "C4"), ("CZ", "PZ"),
        ("C4", "P4"),  ("O1", "O2"),
    ],
    "(b) Pre-seizure": [
        ("C3", "CZ"), ("C3", "T5"), ("C3", "P3"), ("C3", "PZ"),
        ("T5", "P3"), ("T5", "O1"), ("P3", "O1"), ("P3", "PZ"),
        ("PZ", "P4"), ("P4", "O2"),
    ],
    "(c) Post-seizure": [
        ("F7", "F3"), ("F3", "FP1"), ("F3", "FZ"), ("F7", "T3"),
        ("F3", "T3"), ("F3", "C3"),  ("T3", "C3"), ("T3", "T5"),
        ("C3", "T5"), ("T5", "P3"),
    ],
}


def _draw_head_with_fixed_edges(ax, edges, ch_names, ch_pos,
                                node_color="#2e3a8c", edge_color="black",
                                edge_lw=4.0, node_size=1750, label_size=20):
    """Head template with explicit edge list, all edges drawn uniformly
    (same color, same thickness). Matches our scenarios figure's node
    style (navy nodes + white labels) — used for EvoBrain Fig 6 topology
    reproduction so the only visible difference between figures is the
    edge connectivity itself."""
    from viz_channel_focus import head as _head
    _head(ax)
    for a, b in edges:
        x0, y0 = ch_pos[a]; x1, y1 = ch_pos[b]
        ax.plot([x0, x1], [y0, y1], color=edge_color, lw=edge_lw,
                zorder=2, alpha=0.95, solid_capstyle="round")

    xs = [ch_pos[c][0] for c in ch_names]
    ys = [ch_pos[c][1] for c in ch_names]
    ax.scatter(xs, ys, s=node_size, c=node_color, linewidths=0.8,
               edgecolors="white", zorder=3)
    for c, x, y in zip(ch_names, xs, ys):
        ax.text(x, y, c, fontsize=label_size, fontweight="bold",
                ha="center", va="center", color="white", zorder=4)


def make_evobrain_repro_figure(panel_dict, ch_names, ch_pos, out_path,
                                node_size=1750, label_size=20):
    """Reproduce EvoBrain Fig 6's (a)(b)(c) rightmost panels on OUR head
    template — IDENTICAL figsize / gridspec / node size / label size /
    caption placement as `make_evobrain_scenarios_figure` and the topo3
    figures (DOT=1750, LBL=20, figsize=(22, 8.6)), so the only visible
    difference between the figures is the edge connectivity. Right margin
    reserved (no colorbar drawn) so panel widths match exactly."""
    labels = list(panel_dict.keys())
    nP = len(labels)
    # Match topo3 / scenarios figure exactly.
    fig = plt.figure(figsize=(22, 8.6))
    gs = fig.add_gridspec(1, nP, wspace=0.12, top=0.95, left=0.02,
                          right=0.92)
    axes = [fig.add_subplot(gs[0, i]) for i in range(nP)]
    for ax, lbl in zip(axes, labels):
        edges = panel_dict[lbl]
        _draw_head_with_fixed_edges(
            ax, edges, ch_names, ch_pos,
            node_color="#2e3a8c", edge_color="black", edge_lw=4.0,
            node_size=node_size, label_size=label_size)
        ax.set_title(lbl, fontsize=30, pad=8, fontweight="bold", y=-0.12)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_evobrain_scenarios_figure(M_all, baseline, scenarios, win,
                                    ch_names, ch_pos, out_path,
                                    top_k=10, node_size=1750, label_size=20):
    """1×N row of EvoBrain Fig 6 style mini-heads, one snapshot per scenario.
    scenarios: list of (clip_idx, t, panel_label). All panels share one color
    scale (matches EvoBrain: weak / medium / strong vs. shared 'Strong'→'Weak'
    colorbar). Matches topo3 figsize/gridspec so panel widths align."""
    import matplotlib.colors as mcolors
    N = len(ch_names)
    edge_mats = []
    for cid, t, _lbl in scenarios:
        half = win // 2
        T_clip = M_all.shape[1]
        t0, t1 = max(0, t - half), min(T_clip, t + half + 1)
        ew = _pairwise_edges_from_membership(
            M_all[cid, t0:t1].astype(np.float32), baseline, mode="dev")
        edge_mats.append(ew)

    # Shared scale across panels — EvoBrain narrative: Normal's top-k sits at
    # the weak end, Focal's top-k at the strong end, on the SAME cmap.
    triu_i, triu_j = np.triu_indices(N, k=1)
    top_vals = []
    for em in edge_mats:
        w = em[triu_i, triu_j]
        top_vals.extend(sorted(w, reverse=True)[:top_k])
    vmin, vmax = float(min(top_vals)), float(max(top_vals))
    if vmax - vmin < 1e-6: vmin, vmax = vmin - 1e-3, vmax + 1e-3

    nP = len(scenarios)
    fig = plt.figure(figsize=(22, 8.6))
    gs = fig.add_gridspec(1, nP, wspace=0.12, top=0.95, left=0.02,
                          right=0.92)
    axes = [fig.add_subplot(gs[0, i]) for i in range(nP)]
    for ax, (cid, t, lbl), em in zip(axes, scenarios, edge_mats):
        _draw_head_edges(ax, em, ch_names, ch_pos, vmin, vmax, top_k=top_k,
                         node_size=node_size, label_size=label_size,
                         edge_lw=4.5)
        ax.set_title(lbl, fontsize=30, pad=8, fontweight="bold", y=-0.12)

    # Right-side shared Weak→Strong colorbar (matches topo3 colorbar size).
    sm = plt.cm.ScalarMappable(
        cmap="cividis", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.025,
                      shrink=0.55, aspect=14)
    cb.set_ticks([vmin, vmax]); cb.set_ticklabels(["Weak", "Strong"])
    cb.ax.tick_params(labelsize=26, length=8, width=1.4, pad=6)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_figure(clip_M, onset_t, win, ch_names, ch_pos, out_dir,
                clip_idx, seed_tag, edge_idx, p_pred,
                M_all=None, y_all=None, y_prob_all=None, dense_y_all=None):
    """Writes 4 PDFs, all for a single hyperedge (`edge_idx`):
       - <tag>_pre_post_spatial.pdf : non-sz / pre / post / Δ topomap
       - <tag>_onset_aligned.pdf    : 1-row onset-aligned N×T heatmap
       - <tag>_clip<idx>_repr.pdf   : representative single clip heatmap
       - <tag>_gallery.pdf          : multi-clip easy/hard/missed gallery"""
    base = f"{seed_tag}_edge{edge_idx}"
    if M_all is not None:
        make_pre_post_spatial_figure(
            M_all, y_all, dense_y_all, edge_idx, win, ch_names, ch_pos,
            os.path.join(out_dir, f"{base}_pre_post_spatial.pdf"))
        make_single_edge_temporal_figure(
            M_all, y_all, dense_y_all, edge_idx,
            win_pre=15, win_post=20, ch_names=ch_names,
            out_path=os.path.join(out_dir, f"{base}_onset_aligned.pdf"))
        if y_prob_all is not None:
            make_clip_gallery_figure(
                M_all, y_all, y_prob_all, dense_y_all, edge_idx,
                ch_names,
                os.path.join(out_dir, f"{base}_gallery.pdf"))
    make_heatmap_figure(
        clip_M, onset_t, ch_names,
        os.path.join(out_dir, f"{base}_clip{clip_idx}_repr.pdf"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_npz")
    p.add_argument("out_dir")
    p.add_argument("--edge_idx", type=int, default=0)
    p.add_argument("--win",      type=int, default=2,
                   help="Onset window half-width in seconds for spatial mean.")
    p.add_argument("--clip_idx", type=int, default=-1,
                   help="-1 → auto-pick; else use this clip index (default mode).")
    p.add_argument("--clip_ids", type=str, default="",
                   help="Comma-separated clip indices for per-clip mode.")
    p.add_argument("--transitions", action="store_true", default=False,
                   help="Auto-pick 1 representative per transition category "
                        "(0→1→0 / 0→1 / 1→0); write heatmap + 3-panel topomap.")
    p.add_argument("--onset_topo", action="store_true", default=False,
                   help="Population-average ONSET-aligned heatmap + 4-bin topomap.")
    p.add_argument("--offset_topo", action="store_true", default=False,
                   help="Population-average OFFSET-aligned heatmap + 4-bin topomap.")
    p.add_argument("--transition_galleries", action="store_true", default=False,
                   help="One gallery PDF per transition category "
                        "(0→1→0 / 0→1 / 1→0), each easy/hard/missed × 4 clips.")
    p.add_argument("--evobrain", type=str, default="",
                   help="EvoBrain Fig-6 style 3-snapshot edge figure. "
                        "Format: 'CLIP_ID,T1,T2,T3[,TITLE]' "
                        "(times in seconds; uses ALL hyperedges and "
                        "baseline-subtracted outer-product for edges).")
    p.add_argument("--evobrain_win", type=int, default=3,
                   help="Window size (s) for each snapshot averaging.")
    p.add_argument("--evobrain_topk", type=int, default=10,
                   help="Number of strongest edges to draw per panel.")
    p.add_argument("--evobrain_norm", type=str, default="local",
                   choices=["local", "shared"],
                   help="Per-panel color scale: 'local' (default, EvoBrain "
                        "Fig 6 style — each panel's top-k spans weak→strong) "
                        "or 'shared' (absolute strength across the 3 panels).")
    p.add_argument("--evobrain_scenarios", type=str, default="",
                   help="EvoBrain Fig 6 (a)(b)(c)(d) style 1×N row, one "
                        "single-snapshot mini-head per scenario. Format: "
                        "'CLIP1:T1:LABEL1;CLIP2:T2:LABEL2;...' "
                        "(LABEL is the panel title, e.g. '(a) Normal').")
    p.add_argument("--evobrain_repro", action="store_true", default=False,
                   help="Reproduce EvoBrain Fig 6 (a)(b)(c) rightmost panels "
                        "on OUR head template + cividis colorbar (gray nodes, "
                        "topology only). Edge lists are hardcoded.")
    p.add_argument("--seed_tag", type=str, default="s123")
    a = p.parse_args()

    d = np.load(a.dump_npz)
    if "dense_y" not in d.files:
        sys.exit(f"{a.dump_npz} missing dense_y; re-run dump_membership.py --dense")
    M = d["M_last"]; y = d["y_true"]; y_prob = d["y_prob"]; dy = d["dense_y"]
    print(f"M {M.shape}  y_pos={y.sum()}/{y.size}  dense_y {dy.shape}")
    os.makedirs(a.out_dir, exist_ok=True)

    if a.evobrain:
        # Format: "CLIP_ID,T1,T2,T3[,TITLE]"
        parts = [s.strip() for s in a.evobrain.split(",")]
        cid = int(parts[0])
        ts  = [int(x) for x in parts[1:4]]
        title = parts[4] if len(parts) > 4 else None
        M_all = M.astype(np.float32)
        baseline = M_all[y == 0].mean(axis=(0, 1))                   # (N, E_h)
        out_path = os.path.join(
            a.out_dir,
            f"{a.seed_tag}_evobrain_clip{cid}_t{ts[0]}-{ts[1]}-{ts[2]}.pdf")
        make_evobrain_style_figure(
            M_all[cid], baseline, ts, a.evobrain_win,
            CH_TUSZ, POS_TUSZ, out_path,
            panel_title=title, top_k=a.evobrain_topk,
            norm_mode=a.evobrain_norm)
        return

    if a.evobrain_scenarios:
        # Format: "CLIP:T:LABEL;CLIP:T:LABEL;..."
        scenarios = []
        for chunk in a.evobrain_scenarios.split(";"):
            parts = chunk.strip().split(":")
            if len(parts) < 3:
                continue
            cid = int(parts[0]); t = int(parts[1])
            lbl = ":".join(parts[2:]).strip()
            scenarios.append((cid, t, lbl))
        M_all = M.astype(np.float32)
        baseline = M_all[y == 0].mean(axis=(0, 1))
        tag = "_".join(f"c{c}t{t}" for c, t, _ in scenarios)
        out_path = os.path.join(
            a.out_dir, f"{a.seed_tag}_evobrain_scenarios_{tag}.pdf")
        make_evobrain_scenarios_figure(
            M_all, baseline, scenarios, a.evobrain_win,
            CH_TUSZ, POS_TUSZ, out_path, top_k=a.evobrain_topk)
        return

    if a.evobrain_repro:
        # Topology-only reproduction — uniform black edges, navy nodes,
        # no colorbar. `panel_dict` maps panel-label → edge list.
        panel_dict = {
            "(a) Normal":        EVOBRAIN_FIG6_EDGES["(a) Normal"],
            "(b) Pre-seizure":   EVOBRAIN_FIG6_EDGES["(b) Pre-seizure"],
            "(c) Post-seizure":  EVOBRAIN_FIG6_EDGES["(c) Post-seizure"],
        }
        out_path = os.path.join(
            a.out_dir, f"{a.seed_tag}_evobrain_fig6_reproduction.pdf")
        make_evobrain_repro_figure(
            panel_dict, CH_TUSZ, POS_TUSZ, out_path)
        return

    if a.transitions:
        make_transition_rep_figures(
            M, y, y_prob, dy, a.edge_idx, CH_TUSZ, POS_TUSZ,
            a.win, a.out_dir, a.seed_tag)
        return

    if a.transition_galleries:
        cats = np.array([categorize_clip(dy[s]) for s in range(len(y))])
        for kind, label in [("both", "0to1to0"),
                            ("onset_only", "0to1"),
                            ("offset_only", "1to0")]:
            mask = (cats == kind)
            make_clip_gallery_figure(
                M, y, y_prob, dy, a.edge_idx, CH_TUSZ,
                os.path.join(
                    a.out_dir,
                    f"{a.seed_tag}_edge{a.edge_idx}_gallery_{label}.pdf"),
                clip_filter_mask=mask, title_suffix=f"  ({label})")
        return

    if a.onset_topo:
        # Heatmap + 4-bin head topomap, both ONSET (0→1) aligned.
        make_aligned_heatmap_pdf(
            M, y, dy, a.edge_idx, find_onset, "onset", "0→1",
            win_pre=15, win_post=20, ch_names=CH_TUSZ,
            out_path=os.path.join(
                a.out_dir,
                f"{a.seed_tag}_edge{a.edge_idx}_onset_aligned_heatmap.pdf"))
        make_aligned_topomap_pdf(
            M, y, dy, a.edge_idx, CH_TUSZ, POS_TUSZ,
            find_onset, "onset", "0→1",
            bins=[(-15, -5), (-5, 0), (0, 5), (5, 20)],
            out_path=os.path.join(
                a.out_dir,
                f"{a.seed_tag}_edge{a.edge_idx}_onset_aligned_topo.pdf"))
        return

    if a.offset_topo:
        # Heatmap + 4-bin head topomap, both OFFSET (1→0) aligned.
        make_aligned_heatmap_pdf(
            M, y, dy, a.edge_idx, find_offset, "offset", "1→0",
            win_pre=20, win_post=15, ch_names=CH_TUSZ,
            out_path=os.path.join(
                a.out_dir,
                f"{a.seed_tag}_edge{a.edge_idx}_offset_aligned_heatmap.pdf"))
        make_aligned_topomap_pdf(
            M, y, dy, a.edge_idx, CH_TUSZ, POS_TUSZ,
            find_offset, "offset", "1→0",
            bins=[(-20, -5), (-5, 0), (0, 5), (5, 15)],
            out_path=os.path.join(
                a.out_dir,
                f"{a.seed_tag}_edge{a.edge_idx}_offset_aligned_topo.pdf"))
        return

    if a.clip_ids:
        clip_ids = [int(x) for x in a.clip_ids.split(",") if x.strip()]
        make_per_clip_pair(
            M, y, y_prob, dy, a.edge_idx, CH_TUSZ, POS_TUSZ,
            clip_ids, a.win, a.out_dir, a.seed_tag)
        return

    if a.clip_idx >= 0:
        chosen = a.clip_idx
        onset_t = find_onset(dy[chosen])
        if onset_t < 0:
            sys.exit(f"clip {chosen} has no seizure in dense_y")
        print(f"using user-specified clip {chosen} (onset={onset_t}s)")
    else:
        chosen, onset_t = pick_clip(
            M, y, y_prob, dy, a.edge_idx, a.win, n_ch_expected=len(CH_TUSZ))

    clip_M = M[chosen, :, :, a.edge_idx]
    make_figure(clip_M, onset_t, a.win, CH_TUSZ, POS_TUSZ, a.out_dir,
                chosen, a.seed_tag, a.edge_idx, float(y_prob[chosen]),
                M_all=M, y_all=y, y_prob_all=y_prob, dense_y_all=dy)


if __name__ == "__main__":
    main()
