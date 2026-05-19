"""Classical-ML baselines (SVM, Random Forest) on CHB-MIT or TUSZ seizure detection.

Loads the same FFT features the neural baselines see (T=clip_len, N=channels,
F=100 freq bins) via the project's CHB/TUSZ dataloader, then forms a per-clip
feature vector in one of two ways:

  --feature_mode mean    : mean over time -> (N*F)-dim
  --feature_mode flatten : flatten (T, N, F) -> (T*N*F)-dim

Evaluation protocol matches the neural runs:
  - AUROC on full test set
  - F1 at tau* = best-F1 threshold tuned on dev
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def collect_features(loader, mode):
    feats, labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.numpy()
        y = y.numpy().reshape(-1)
        if mode == "mean":
            v = x.mean(axis=1).reshape(x.shape[0], -1)
        elif mode == "flatten":
            v = x.reshape(x.shape[0], -1)
        else:
            raise ValueError(mode)
        feats.append(v.astype(np.float32))
        labels.append(y.astype(np.int64))
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def best_f1_threshold(scores, y):
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.5, 0.0
    s = np.sort(np.unique(scores))
    if len(s) > 4096:
        idx = np.linspace(0, len(s) - 1, 4096).astype(int)
        s = s[idx]
    cand = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    best_t, best_f1 = 0.5, 0.0
    for t in cand:
        pred = (scores >= t).astype(np.int64)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("CHBMIT", "TUSZ"), required=True)
    p.add_argument("--model", choices=("svm", "rf"), required=True)
    p.add_argument("--feature_mode", choices=("mean", "flatten"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--clip_len", type=int, default=12)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--raw_data_dir", required=True)
    p.add_argument("--preproc_dir", default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--svm_C", type=float, default=1.0)
    p.add_argument("--svm_kernel", default="rbf")
    p.add_argument("--svm_gamma", default="scale")
    p.add_argument("--rf_n_estimators", type=int, default=200)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--vanilla", action="store_true",
                   help="Use sklearn vanilla defaults: no class_weight, RF n_estimators=100.")
    args = p.parse_args()
    if args.vanilla:
        args.rf_n_estimators = 100

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.dataset} clip_len={args.clip_len} seed={args.seed} feat={args.feature_mode}")
    if args.dataset == "CHBMIT":
        from data.dataloader_chb import load_dataset_chb
        dataloaders, _, _ = load_dataset_chb(
            task="detection",
            input_dir=args.input_dir, raw_data_dir=args.raw_data_dir,
            train_batch_size=512, test_batch_size=512,
            time_step_size=1, max_seq_len=args.clip_len,
            standardize=False, num_workers=args.num_workers,
            augmentation=False, graph_type="none", top_k=3,
            filter_type="dual_random_walk", use_fft=True,
            sampling_ratio=1, seed=args.seed,
            preproc_dir=args.preproc_dir, return_raw=False, dense_labels=False,
        )
    else:  # TUSZ
        from data.dataloader_detection import load_dataset_detection
        dataloaders, _, _ = load_dataset_detection(
            input_dir=args.input_dir, raw_data_dir=args.raw_data_dir,
            train_batch_size=512, test_batch_size=512,
            time_step_size=1, max_seq_len=args.clip_len,
            standardize=True, num_workers=args.num_workers,
            augmentation=False,
            adj_mat_dir="./data/electrode_graph/adj_mx_3d.pkl",
            graph_type="none", top_k=3, filter_type="dual_random_walk",
            use_fft=True, sampling_ratio=1, seed=args.seed,
            preproc_dir=args.preproc_dir, dense_labels=False,
        )

    t0 = time.time()
    X_tr, y_tr = collect_features(dataloaders["train"], args.feature_mode)
    print(f"train: X={X_tr.shape} pos={y_tr.sum()}/{len(y_tr)}  ({time.time()-t0:.1f}s)")
    X_dv, y_dv = collect_features(dataloaders["dev"], args.feature_mode)
    print(f"dev:   X={X_dv.shape} pos={y_dv.sum()}/{len(y_dv)}  ({time.time()-t0:.1f}s)")
    X_te, y_te = collect_features(dataloaders["test"], args.feature_mode)
    print(f"test:  X={X_te.shape} pos={y_te.sum()}/{len(y_te)}  ({time.time()-t0:.1f}s)")

    class_weight = None if args.vanilla else "balanced"
    if args.model == "svm":
        clf = SVC(C=args.svm_C, kernel=args.svm_kernel, gamma=args.svm_gamma,
                  probability=False, class_weight=class_weight,
                  random_state=args.seed, cache_size=4000)
        print(f"Fitting SVC kernel={args.svm_kernel} C={args.svm_C} gamma={args.svm_gamma} class_weight={class_weight} ...")
        t1 = time.time()
        clf.fit(X_tr, y_tr)
        print(f"fit done ({time.time()-t1:.1f}s, n_sv={clf.n_support_})")
        score_fn = lambda X: clf.decision_function(X)
    else:
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
            n_jobs=-1, class_weight=class_weight, random_state=args.seed)
        print(f"Fitting RF n_est={args.rf_n_estimators} max_depth={args.rf_max_depth} class_weight={class_weight} ...")
        t1 = time.time()
        clf.fit(X_tr, y_tr)
        print(f"fit done ({time.time()-t1:.1f}s)")
        score_fn = lambda X: clf.predict_proba(X)[:, 1]

    s_dv = score_fn(X_dv)
    s_te = score_fn(X_te)
    dev_auroc = float(roc_auc_score(y_dv, s_dv))
    test_auroc = float(roc_auc_score(y_te, s_te))

    tau, dev_f1 = best_f1_threshold(s_dv, y_dv)
    pred_te = (s_te >= tau).astype(np.int64)
    test_f1 = float(f1_score(y_te, pred_te, zero_division=0))
    test_prec = float(precision_score(y_te, pred_te, zero_division=0))
    test_rec = float(recall_score(y_te, pred_te, zero_division=0))

    results = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "clip_len": args.clip_len, "feature_mode": args.feature_mode,
        "tau_star_on_dev": tau, "dev_F1": dev_f1, "dev_AUROC": dev_auroc,
        "test_AUROC": test_auroc, "test_F1": test_f1,
        "test_precision": test_prec, "test_recall": test_rec,
        "n_train": int(len(y_tr)), "n_dev": int(len(y_dv)), "n_test": int(len(y_te)),
        "n_train_pos": int(y_tr.sum()), "n_dev_pos": int(y_dv.sum()), "n_test_pos": int(y_te.sum()),
    }
    out = os.path.join(args.save_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== RESULTS ({args.dataset} {args.model} seed{args.seed} {args.feature_mode}) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
