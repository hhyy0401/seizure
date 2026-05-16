"""Aggregate 18-job grid (light_st_hyper × aux × E) into:
  (a) final-ckpt table: dev AUROC | test AUROC | dev-test gap | F1@τ* | P | R
      where τ* is F1-best on dev, applied to test (paper-style)
  (b) per-run (dev_AUROC, test_AUROC) trajectory from info.log — to inspect
      dev↔test correlation across epochs

The "dev" / "test" here are what main.py writes to dev_results.npz /
test_results.npz from the dev-AUROC-best checkpoint (best.pth.tar). Test in
info.log mid-eval is labelled "MidTest" but is the same TUSZ test split.
"""
import os, glob, json, re, sys
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
)

ROOT = "/storage/scratch1/3/hkim3239/eeg/runs"

def thresh_max_f1(y_true, y_prob):
    p, r, th = precision_recall_curve(y_true, y_prob)
    f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    if len(f) == 0 or not np.isfinite(f).any():
        return 0.5, 0.0
    idx = int(np.nanargmax(f))
    return float(th[idx]), float(f[idx])

# Parse info.log to get per-eval (epoch, dev_auroc, test_auroc, dev_f1, test_f1)
LOG_RE_DEV = re.compile(r"Dev loss: [\d.]+, acc: [\d.]+, F1: ([\d.]+), "
                        r"recall: [\d.]+, precision: [\d.]+, "
                        r"best_thresh: [\d.]+, auroc: ([\d.]+)")
LOG_RE_TST = re.compile(r"MidTest loss: [\d.]+, acc: [\d.]+, F1: ([\d.]+), "
                        r"recall: [\d.]+, precision: [\d.]+, "
                        r"best_thresh: [\d.]+, auroc: ([\d.]+)")
LOG_RE_EP  = re.compile(r"Evaluating at epoch (\d+)")

def parse_trajectory(log_path):
    """Return list of (epoch, dev_auroc, dev_f1, test_auroc, test_f1)."""
    out = []
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m_ep = LOG_RE_EP.search(lines[i])
        if m_ep:
            ep = int(m_ep.group(1))
            dev = test = None
            for j in range(i+1, min(i+30, len(lines))):
                if dev is None:
                    m = LOG_RE_DEV.search(lines[j])
                    if m: dev = (float(m.group(2)), float(m.group(1)))  # (auroc, f1)
                if test is None:
                    m = LOG_RE_TST.search(lines[j])
                    if m: test = (float(m.group(2)), float(m.group(1)))
                if dev is not None and test is not None:
                    break
                if "Starting epoch" in lines[j] and j > i:
                    break
            if dev is not None and test is not None:
                out.append((ep, dev[0], dev[1], test[0], test[1]))
        i += 1
    return out

def run_metrics(run_dir):
    cfgs = glob.glob(os.path.join(run_dir, "**", "args.json"), recursive=True)
    if not cfgs:
        return None
    cfg = cfgs[0]
    with open(cfg) as f:
        args = json.load(f)
    work = os.path.dirname(cfg)
    dev_npz = os.path.join(work, "dev_results.npz")
    tst_npz = os.path.join(work, "test_results.npz")
    log = os.path.join(work, "info.log")
    if not (os.path.exists(dev_npz) and os.path.exists(tst_npz)):
        return None
    d = np.load(dev_npz); t = np.load(tst_npz)
    y_d, p_d = d["y_true"], d["y_prob"]
    y_t, p_t = t["y_true"], t["y_prob"]
    thr, _ = thresh_max_f1(y_d, p_d)
    pr = (p_t >= thr).astype(int)
    dev_auroc = float(roc_auc_score(y_d, p_d))
    test_auroc = float(roc_auc_score(y_t, p_t))
    # also dev F1 at same τ
    pr_d = (p_d >= thr).astype(int)
    dev_f1  = float(f1_score(y_d, pr_d, zero_division=0))
    return dict(
        run=os.path.basename(run_dir),
        model=args.get("model_name","?"),
        E=args.get("n_hyperedges","-"),
        aux=args.get("aux_type","-"),
        params_path=cfg, log_path=log if os.path.exists(log) else None,
        dev_auroc=dev_auroc, test_auroc=test_auroc,
        thr=thr,
        dev_f1=dev_f1,
        P=float(precision_score(y_t, pr, zero_division=0)),
        R=float(recall_score(y_t, pr, zero_division=0)),
        test_f1=float(f1_score(y_t, pr, zero_division=0)),
    )

def main(prefix="tusz_light_st_hyper", show_traj=False):
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, prefix + "*"))):
        r = run_metrics(d)
        if r is None: continue
        rows.append(r)
    if not rows:
        print("(no completed runs yet)"); return
    rows.sort(key=lambda r: -r["test_f1"])
    print(f"\n=== FINAL (dev-AUROC-best ckpt) — sorted by test F1@τ* ===")
    print(f"{'model':<24} {'E':>2} {'aux':>8}  "
          f"{'dev_AU':>6} {'tst_AU':>6} {'gap':>6}  "
          f"{'τ*':>5} {'devF1':>5} {'tstF1':>5}  {'P':>5} {'R':>5}")
    print("-"*100)
    for r in rows:
        gap = r["test_auroc"] - r["dev_auroc"]
        print(f"{r['model']:<24} {r['E']:>2} {r['aux']:>8}  "
              f"{r['dev_auroc']:>6.3f} {r['test_auroc']:>6.3f} {gap:>+6.3f}  "
              f"{r['thr']:>5.3f} {r['dev_f1']:>5.3f} {r['test_f1']:>5.3f}  "
              f"{r['P']:>5.3f} {r['R']:>5.3f}")

    if show_traj:
        print(f"\n=== Per-run trajectory (epoch, dev_AUROC, test_AUROC) ===")
        for r in rows:
            if not r["log_path"]: continue
            traj = parse_trajectory(r["log_path"])
            if not traj: continue
            # compute corr (dev vs test AUROC across eval points)
            devs = np.array([t[1] for t in traj])
            tsts = np.array([t[3] for t in traj])
            if len(devs) >= 3 and devs.std() > 0 and tsts.std() > 0:
                corr = float(np.corrcoef(devs, tsts)[0,1])
            else:
                corr = float("nan")
            print(f"\n[{r['model']} E={r['E']} aux={r['aux']}]  "
                  f"#evals={len(traj)}  corr(dev,test AUROC)={corr:+.3f}")
            # show every 10th eval point
            for ep, dau, dfl, tau, tfl in traj[::max(1,len(traj)//10)]:
                print(f"  ep{ep:>3}  dev_AUROC={dau:.3f}  test_AUROC={tau:.3f}  "
                      f"(test-dev={tau-dau:+.3f})")

if __name__ == "__main__":
    show = "--traj" in sys.argv
    main(show_traj=show)
