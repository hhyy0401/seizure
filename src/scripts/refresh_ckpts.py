"""Pick best (E_h=1, seed=123) checkpoint per dataset from the final sweep
and copy into ckpts/{tusz12,tusz60,chb12}/. Used after the final sweep to
refresh the shipped checkpoints.

Usage:
    python scripts/refresh_ckpts.py [--e-h 1] [--seed 123]
"""
import argparse
import glob
import os
import shutil

REPO = "/storage/project/r-nimam6-0/hkim3239/disease"
ROOTS = ["/storage/scratch1/3/hkim3239/eeg/runs",
         os.path.join(REPO, "runs")]

TARGETS = {
    "tusz12": "final_tusz12_E{E}_s{seed}_*",
    "tusz60": "final_tusz60_E{E}_s{seed}_*",
    "chb12":  "final_chb12_E{E}_s{seed}_*",
}


def find_run_dir(pattern):
    for root in ROOTS:
        matches = sorted(glob.glob(os.path.join(root, pattern)))
        if matches:
            return matches[-1]
    return None


def find_artifacts(run_dir):
    ckpt = glob.glob(os.path.join(run_dir, "**", "best.pth.tar"), recursive=True)
    args = glob.glob(os.path.join(run_dir, "**", "args.json"), recursive=True)
    if not (ckpt and args):
        return None, None
    return ckpt[0], args[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-h", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    for dst_name, pat in TARGETS.items():
        run_dir = find_run_dir(pat.format(E=args.e_h, seed=args.seed))
        if run_dir is None:
            print(f"[skip] {dst_name}: no run dir matches E={args.e_h} seed={args.seed}")
            continue
        ckpt, argf = find_artifacts(run_dir)
        if ckpt is None:
            print(f"[skip] {dst_name}: {run_dir} has no best.pth.tar yet")
            continue
        dst = os.path.join(REPO, "ckpts", dst_name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(ckpt, os.path.join(dst, "best.pth.tar"))
        shutil.copy2(argf, os.path.join(dst, "args.json"))
        print(f"[ok]   {dst_name}: copied from {os.path.basename(run_dir)}")


if __name__ == "__main__":
    main()
