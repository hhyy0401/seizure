"""Aggregate multi-seed runs (ms_*) into mean ± std table.

Groups by (dataset, clip_len, n_hyperedges, d_model, lr).
For each group, reports mean ± std of:
  - test AUROC  (at dev-AUROC-best ckpt)
  - F1@τ*       (τ* = F1-best on dev, applied to test — paper-style)

Usage:
    python scripts/aggregate_multiseed.py
"""
import os, glob, json
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

ROOTS = ["/storage/scratch1/3/hkim3239/eeg/runs",
         "/storage/project/r-nimam6-0/hkim3239/disease/runs"]


def thresh_max_f1(y_true, y_prob):
    p, r, th = precision_recall_curve(y_true, y_prob)
    f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    if len(f) == 0 or not np.isfinite(f).any(): return 0.5, 0.0
    idx = int(np.nanargmax(f))
    return float(th[idx]), float(f[idx])


def metrics(run_dir):
    cfgs = glob.glob(os.path.join(run_dir, "**", "args.json"), recursive=True)
    if not cfgs: return None
    cfg = cfgs[0]
    with open(cfg) as f: args = json.load(f)
    work = os.path.dirname(cfg)
    dev_npz = os.path.join(work, "dev_results.npz")
    tst_npz = os.path.join(work, "test_results.npz")
    if not (os.path.exists(dev_npz) and os.path.exists(tst_npz)): return None
    d = np.load(dev_npz); t = np.load(tst_npz)
    thr, _ = thresh_max_f1(d["y_true"], d["y_prob"])
    pred = (t["y_prob"] >= thr).astype(int)
    return dict(
        run=os.path.basename(run_dir),
        dataset=args.get("dataset", "?"),
        clip_len=args.get("max_seq_len", "?"),
        E=args.get("n_hyperedges", "-"),
        d_model=args.get("rnn_units", "-"),
        lr=args.get("lr_init", "-"),
        seed=args.get("rand_seed", "-"),
        dev_AU=float(roc_auc_score(d["y_true"], d["y_prob"])),
        tst_AU=float(roc_auc_score(t["y_true"], t["y_prob"])),
        tau_star=thr,
        tst_F1=float(f1_score(t["y_true"], pred, zero_division=0)),
    )


def main():
    rows = []
    seen = set()
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "ms_*"))):
            if not os.path.isdir(d): continue
            key = os.path.basename(d)
            if key in seen: continue
            seen.add(key)
            r = metrics(d)
            if r is not None: rows.append(r)

    print(f"Total multi-seed runs found: {len(rows)}")

    groups = {}
    for r in rows:
        k = (r["dataset"], r["clip_len"], r["E"], r["d_model"], r["lr"])
        groups.setdefault(k, []).append(r)

    PAPER = {
        ("TUSZ",   12): {"AUROC": (0.877, 0.005), "F1": (0.539, 0.009)},
        ("TUSZ",   60): {"AUROC": (0.865, 0.009), "F1": (0.483, 0.006)},
        ("CHBMIT", 12): {"AUROC": (0.940, None),  "F1": (None,  None)},
    }

    for (ds, cl) in [("TUSZ", 12), ("TUSZ", 60), ("CHBMIT", 12)]:
        title = f"{ds} {cl}s detection"
        print(f"\n=== {title} ===")
        p = PAPER.get((ds, cl), {})
        au = p.get("AUROC", (None, None))
        f1 = p.get("F1", (None, None))
        au_s = f"{au[0]:.3f}" + (f" ±{au[1]:.3f}" if au[1] else "") if au[0] else "—"
        f1_s = f"{f1[0]:.3f}" + (f" ±{f1[1]:.3f}" if f1[1] else "") if f1[0] else "—"
        print(f"  paper EvoBrain:  AUROC={au_s}  F1={f1_s}")

        sub = [(k, v) for k, v in groups.items() if k[0] == ds and k[1] == cl]
        if not sub:
            print("  ours: no runs yet")
            continue

        print(f"  {'cfg':<22} {'n':<3} {'AUROC (mean±std)':<18} {'F1@τ* (mean±std)':<18}")
        sub.sort(key=lambda kv: -np.mean([r["tst_F1"] for r in kv[1]]))
        for (ds2, cl2, E, d_, lr), runs in sub:
            n = len(runs)
            au_arr = np.array([r["tst_AU"] for r in runs])
            f1_arr = np.array([r["tst_F1"] for r in runs])
            cfg_s = f"E={E} d={d_} lr={lr}"
            au_s2 = f"{au_arr.mean():.3f} ±{au_arr.std(ddof=1) if n>1 else 0:.3f}"
            f1_s2 = f"{f1_arr.mean():.3f} ±{f1_arr.std(ddof=1) if n>1 else 0:.3f}"
            print(f"  {cfg_s:<22} {n:<3} {au_s2:<18} {f1_s2:<18}")
            for r in sorted(runs, key=lambda r: r["seed"]):
                print(f"      seed={r['seed']:<4}  tstAU={r['tst_AU']:.3f}  F1={r['tst_F1']:.3f}  ({r['run']})")


if __name__ == "__main__":
    main()
