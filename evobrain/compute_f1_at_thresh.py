"""Recompute F1/precision/recall at a fixed threshold from saved ROC data.

ROC data (fpr, tpr, thresholds) was saved per model in main.py. For binary
classification we can recover the full confusion matrix at any threshold
because:
    TP = tpr · P,   FN = P - TP
    FP = fpr · N,   TN = N - FP

No model re-inference needed.
"""
import numpy as np
from pathlib import Path

RUNS = [
    ("EvoBrain",      "/home/hkim3239/eeg/runs/evobrain_12s_20260512_212356/CHBMIT/detection/12/evobrain_dynamic_123_01"),
    ("Light-Attn",    "/home/hkim3239/eeg/runs/light_attention_12s_20260513_002041/CHBMIT/detection/12/light_attention_none_123_01"),
    ("PLV",           "/home/hkim3239/eeg/runs/light_mamba_band_plv_12s_20260512_212356/CHBMIT/detection/12/light_mamba_band_plv_dynamic_123_01"),
    ("Dyn-Hyper",     "/home/hkim3239/eeg/runs/light_dyn_hyper_12s_20260513_002041/CHBMIT/detection/12/light_dyn_hyper_none_123_01"),
    ("Static-Hyper",  "/home/hkim3239/eeg/runs/light_static_hyper_12s_20260513_002041/CHBMIT/detection/12/light_static_hyper_none_123_01"),
    ("no-PLV",        "/home/hkim3239/eeg/runs/light_attn_band_gated_12s_20260513_002041/CHBMIT/detection/12/light_attn_band_gated_none_123_01"),
]

# CHB-MIT test split (from filemarker line counts)
P_TEST = 162
N_TEST = 40514

THRESHOLDS = [0.5, 0.7, 0.9]


def metrics_at_thresh(fpr, tpr, thresholds, target, P, N):
    """sklearn's roc_curve returns thresholds in DESCENDING order. Find the
    largest threshold ≤ target — that's the operating point if you set τ=target."""
    # thresholds[0] is +inf or max+1; pick the index where thresholds <= target first
    idx = np.searchsorted(-thresholds, -target, side='left')
    idx = min(idx, len(thresholds) - 1)
    fpr_i, tpr_i, th_i = fpr[idx], tpr[idx], thresholds[idx]
    TP = tpr_i * P
    FN = P - TP
    FP = fpr_i * N
    TN = N - FP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = tpr_i
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auroc = float(np.trapezoid(tpr, fpr))
    return {
        "thresh_used": float(th_i),
        "TP": float(TP), "FN": float(FN), "FP": float(FP), "TN": float(TN),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "AUROC": float(auroc),
    }


def main():
    print(f"{'Model':<14s} {'τ':>5s} {'τ_used':>7s} {'TP':>5s} {'FP':>7s} {'FN':>5s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUROC':>6s}")
    print("-" * 84)
    for name, path in RUNS:
        npz = Path(path) / "test_roc_data.npz"
        if not npz.exists():
            print(f"  MISSING: {npz}")
            continue
        d = np.load(npz)
        fpr, tpr, thresholds = d["fpr"], d["tpr"], d["thresholds"]
        for tau in THRESHOLDS:
            m = metrics_at_thresh(fpr, tpr, thresholds, tau, P_TEST, N_TEST)
            print(f"{name:<14s} {tau:>5.2f} {m['thresh_used']:>7.3f} "
                  f"{m['TP']:>5.0f} {m['FP']:>7.0f} {m['FN']:>5.0f} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['F1']:>6.3f} "
                  f"{m['AUROC']:>6.3f}")
        print()


if __name__ == "__main__":
    main()
