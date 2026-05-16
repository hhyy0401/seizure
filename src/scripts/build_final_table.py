"""Final results table for paper.

Final config: uni-Mamba + no-aux + node_emb, E_h ∈ {1,2,3}, 3 seeds, 3 datasets.

Per (dataset, E_h):
  test AUROC at dev-AUROC-best checkpoint,
  F1@τ\* where τ\* = F1-best on dev,
  mean ± std over seeds.

Also reports paper EvoBrain reference numbers in the same columns.

Output: FINAL_RESULTS.md
"""
import os, glob, json
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

REPO = "/storage/project/r-nimam6-0/hkim3239/disease"
ROOTS = ["/storage/scratch1/3/hkim3239/eeg/runs",
         os.path.join(REPO, "runs")]

PAPER = {
    ("TUSZ", 12):   {"AUROC": (0.877, 0.005), "F1": (0.539, 0.009)},
    ("TUSZ", 60):   {"AUROC": (0.865, 0.009), "F1": (0.483, 0.006)},
    ("CHBMIT", 12): {"AUROC": (0.940, None),  "F1": (None, None)},
}

PATTERNS = {
    ("TUSZ", 12):   "final_tusz12_E{E}_s*",
    ("TUSZ", 60):   "final_tusz60_E{E}_s*",
    ("CHBMIT", 12): "final_chb12_E{E}_s*",
}


def thresh_max_f1(y, p):
    pr, re, th = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * re[:-1] / np.clip(pr[:-1] + re[:-1], 1e-12, None)
    if len(f) == 0 or not np.isfinite(f).any():
        return 0.5
    return float(th[int(np.nanargmax(f))])


def one_run(run_dir):
    cfgs = glob.glob(os.path.join(run_dir, "**", "args.json"), recursive=True)
    if not cfgs:
        return None
    work = os.path.dirname(cfgs[0])
    dn, tn = os.path.join(work, "dev_results.npz"), os.path.join(work, "test_results.npz")
    if not (os.path.exists(dn) and os.path.exists(tn)):
        return None
    d, t = np.load(dn), np.load(tn)
    thr = thresh_max_f1(d["y_true"], d["y_prob"])
    pred = (t["y_prob"] >= thr).astype(int)
    return (float(roc_auc_score(t["y_true"], t["y_prob"])),
            float(f1_score(t["y_true"], pred, zero_division=0)))


def collect(pat):
    aus, f1s, seen = [], [], set()
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, pat))):
            k = os.path.basename(d)
            if k in seen:
                continue
            seen.add(k)
            r = one_run(d)
            if r is not None:
                aus.append(r[0])
                f1s.append(r[1])
    return aus, f1s


def fmt(arr):
    a = np.array(arr, float)
    if a.size == 0:
        return "—"
    if a.size == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f} ±{a.std(ddof=1):.3f}"


def paper_cell(t):
    if t is None or t[0] is None:
        return "—"
    return f"{t[0]:.3f}" + (f" ±{t[1]:.3f}" if t[1] else "")


def main():
    datasets = [("TUSZ", 12), ("TUSZ", 60), ("CHBMIT", 12)]
    e_hs = [1, 2, 3]

    cells = {}  # cells[(metric, ds, cl, E)] = "..."
    for ds, cl in datasets:
        for E in e_hs:
            pat = PATTERNS[(ds, cl)].format(E=E)
            aus, f1s = collect(pat)
            n = len(aus)
            cells[("AUROC", ds, cl, E)] = fmt(aus) + (f" (n={n})" if n < 3 else "")
            cells[("F1", ds, cl, E)] = fmt(f1s) + (f" (n={n})" if n < 3 else "")

    def header(metric):
        ds_labels = {("TUSZ", 12): "TUSZ 12s", ("TUSZ", 60): "TUSZ 60s",
                     ("CHBMIT", 12): "CHB-MIT 12s"}
        lines = [f"## {metric}", "",
                 "| Config | " + " | ".join(ds_labels[d] for d in datasets) + " |",
                 "|" + "---|" * (len(datasets) + 1)]
        # Paper row
        prow = ["Paper EvoBrain"]
        for ds, cl in datasets:
            prow.append(paper_cell(PAPER[(ds, cl)].get(metric)))
        lines.append("| " + " | ".join(prow) + " |")
        # Our rows
        for E in e_hs:
            row = [f"uni-Mamba E={E}"]
            for ds, cl in datasets:
                row.append(cells[(metric, ds, cl, E)])
            lines.append("| " + " | ".join(row) + " |")
        return lines

    out = ["# Final results — uni-Mamba + no-aux + node_emb",
           "",
           "Config: backbone = uni-Mamba, aux head OFF, learnable node embedding ON.",
           "3 seeds (123, 456, 789), mean ± std.",
           "Test AUROC at dev-AUROC-best checkpoint; F1@τ\\* with τ\\* tuned on dev.",
           "Paper EvoBrain reference taken from the paper's Table 1.",
           ""]
    out += header("AUROC") + [""] + header("F1") + [""]
    txt = "\n".join(out) + "\n"

    with open(os.path.join(REPO, "FINAL_RESULTS.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
