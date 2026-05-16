"""Build a results table across datasets/windows:
  TUSZ 12s, TUSZ 60s, CHB-MIT 12s.

For each run with dev_results.npz + test_results.npz, compute:
  - test AUROC (best-dev-AUROC ckpt; AUROC is threshold-independent)
  - F1@τ*  where τ* = F1-best τ on dev_results, applied to test (paper-style)

Reports top-N per dataset + EvoBrain repro (if present) for comparison.
Paper baselines (manually entered from PDF Table 1 + text).

Usage:
    python scripts/build_results_table.py [--top 5]
"""
import os, sys, glob, json, argparse, re
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
        model=args.get("model_name","?"),
        E=args.get("n_hyperedges","-"),
        aux=args.get("aux_type","-"),
        d_model=args.get("rnn_units","-"),
        lr=args.get("lr_init","-"),
        dropout=args.get("dropout","-"),
        l2_wd=args.get("l2_wd","-"),
        dataset=args.get("dataset","?"),
        clip_len=args.get("max_seq_len","?"),
        dev_AU=float(roc_auc_score(d["y_true"], d["y_prob"])),
        tst_AU=float(roc_auc_score(t["y_true"], t["y_prob"])),
        tau_star=thr,
        tst_F1=float(f1_score(t["y_true"], pred, zero_division=0)),
    )


def collect():
    rows = []
    seen = set()
    for root in ROOTS:
        for d in glob.glob(os.path.join(root, "*")):
            if not os.path.isdir(d): continue
            key = os.path.basename(d)
            if key in seen: continue
            seen.add(key)
            r = metrics(d)
            if r is not None: rows.append(r)
    return rows


def bucket(rows):
    """Group by (dataset, clip_len)."""
    out = {}
    for r in rows:
        k = (r["dataset"], r["clip_len"])
        out.setdefault(k, []).append(r)
    return out


def print_table(rows, top=5):
    buckets = bucket(rows)

    paper_evobrain = {
        ("TUSZ", 12):  {"AUROC": (0.877, 0.005), "F1": (0.539, 0.009)},
        ("TUSZ", 60):  {"AUROC": (0.865, 0.009), "F1": (0.483, 0.006)},
        ("CHBMIT", 12):{"AUROC": (0.94, None),  "F1": (None, None)},  # text only, no F1
    }

    for (ds, cl) in [("TUSZ", 12), ("TUSZ", 60), ("CHBMIT", 12)]:
        title = f"{ds} {cl}s detection"
        print(f"\n=== {title} ===")
        p = paper_evobrain.get((ds, cl), {})
        if p:
            au = p.get("AUROC", (None, None))
            f1 = p.get("F1", (None, None))
            au_s = f"{au[0]:.3f}" + (f" ±{au[1]:.3f}" if au[1] else "") if au[0] else "—"
            f1_s = f"{f1[0]:.3f}" + (f" ±{f1[1]:.3f}" if f1[1] else "") if f1[0] else "—"
            print(f"  paper EvoBrain:    AUROC={au_s}  F1={f1_s}")
        # Filter: evobrain repro (model_name==evobrain) + best ours (light_st_hyper)
        runs = buckets.get((ds, cl), [])
        evobrain_runs = [r for r in runs if r["model"] == "evobrain"]
        ours_runs = [r for r in runs if r["model"].startswith("light_st_hyper")]
        if evobrain_runs:
            best = max(evobrain_runs, key=lambda r: r["dev_AU"])
            print(f"  our EvoBrain repro: AUROC={best['tst_AU']:.3f}  F1={best['tst_F1']:.3f}  "
                  f"(dev_AU={best['dev_AU']:.3f}, run={best['run']})")
        elif ds == "TUSZ" and cl == 12:
            print(f"  our EvoBrain repro: (see /storage/project/.../runs/tusz_evobrain_*)")
        if ours_runs:
            ours_runs.sort(key=lambda r: -r["tst_F1"])
            print(f"  ours top-{min(top,len(ours_runs))} (sorted by test F1@τ*):")
            hdr = f"    {'E':<2} {'aux':<5} {'d':<4} {'lr':<7} {'drop':<5} {'wd':<7} | {'devAU':<6} {'tstAU':<6} {'τ*':<5} {'tstF1':<6} | run"
            print(hdr)
            for r in ours_runs[:top]:
                lr_s = f"{r['lr']:.0e}" if isinstance(r['lr'], (int,float)) else str(r['lr'])
                wd_s = f"{r['l2_wd']:.0e}" if isinstance(r['l2_wd'], (int,float)) else str(r['l2_wd'])
                dr_s = f"{r['dropout']:.2f}" if isinstance(r['dropout'], (int,float)) else str(r['dropout'])
                print(f"    {r['E']:<2} {r['aux']:<5} {r['d_model']:<4} {lr_s:<7} {dr_s:<5} {wd_s:<7} | "
                      f"{r['dev_AU']:.3f}  {r['tst_AU']:.3f}  {r['tau_star']:.3f} {r['tst_F1']:.3f} | {r['run']}")
        else:
            print(f"  ours: no completed runs yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    rows = collect()
    print(f"Total finalized runs found: {len(rows)}")
    print_table(rows, top=args.top)
