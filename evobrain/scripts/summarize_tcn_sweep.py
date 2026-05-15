"""Summarize the 6-run TCN-variant sweep (dwsep × timesnet, E_h ∈ {2,3,4}).

Walks ~/eeg/runs for save dirs matching `tusz_light_st_hyper_(dwsep|timesnet)_NE_E{2,3,4}_*`,
parses the final DEV / TEST line from info.log and the param count, and prints
one row per run plus a markdown table comparing to the λ baseline.

Usage:
    python scripts/summarize_tcn_sweep.py
"""
import os
import re
import glob
import json

RUNS = os.path.expanduser("~/eeg/runs")
PAT = re.compile(r"tusz_light_st_hyper_(dwsep|timesnet)_NE_E(\d+)_12s_(\d+_\d+)$")

LAMBDA_BASELINE = {"dev_auroc": 0.8694, "test_auroc": 0.4796}  # from memory


def parse_log(info_log: str):
    dev_line = None
    test_line = None
    with open(info_log) as f:
        for line in f:
            if "DEV set prediction results" in line:
                dev_line = line.strip()
            elif "TEST set prediction results" in line:
                test_line = line.strip()
    return dev_line, test_line


def extract(line: str, key: str):
    if not line:
        return None
    m = re.search(rf"{key}:\s*([0-9.]+)", line)
    return float(m.group(1)) if m else None


def main():
    rows = []
    for sd in sorted(glob.glob(os.path.join(RUNS, "tusz_light_st_hyper_*_NE_E*_12s_*"))):
        base = os.path.basename(sd)
        m = PAT.match(base)
        if not m:
            continue
        backbone, eh, stamp = m.group(1), int(m.group(2)), m.group(3)
        info_logs = glob.glob(os.path.join(sd, "**", "info.log"), recursive=True)
        if not info_logs:
            continue
        info_log = info_logs[0]
        args_path = os.path.join(os.path.dirname(info_log), "args.json")
        params = None
        if os.path.exists(args_path):
            with open(info_log) as f:
                for line in f:
                    pm = re.search(r"Total number of trainable parameters:\s*(\d+)", line)
                    if pm:
                        params = int(pm.group(1))
                        break
        dev_line, test_line = parse_log(info_log)
        rows.append({
            "backbone": backbone,
            "E_h": eh,
            "stamp": stamp,
            "params": params,
            "dev_auroc": extract(dev_line, "auroc"),
            "dev_f1": extract(dev_line, "F1"),
            "test_auroc": extract(test_line, "auroc"),
            "test_f1": extract(test_line, "F1"),
            "save_dir": sd,
        })

    rows.sort(key=lambda r: (r["backbone"], r["E_h"], r["stamp"]))
    print(f"{'backbone':>10} {'E_h':>4} {'params':>8} {'dev_auroc':>10} {'dev_f1':>8} {'test_auroc':>11} {'test_f1':>8}  stamp")
    for r in rows:
        p = f"{r['params']/1e3:.1f}k" if r['params'] else "-"
        print(f"{r['backbone']:>10} {r['E_h']:>4} {p:>8} "
              f"{r['dev_auroc'] or 0:>10.4f} {r['dev_f1'] or 0:>8.4f} "
              f"{r['test_auroc'] or 0:>11.4f} {r['test_f1'] or 0:>8.4f}  {r['stamp']}")

    print()
    print(f"λ baseline (memory): dev_auroc={LAMBDA_BASELINE['dev_auroc']:.4f}, "
          f"test_auroc={LAMBDA_BASELINE['test_auroc']:.4f}")

    # Markdown table for paper
    print("\n### Markdown table (paste into notes)\n")
    print("| backbone | E_h | params | dev AUROC | dev F1 | test AUROC | test F1 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        p = f"{r['params']/1e3:.1f}k" if r['params'] else "-"
        print(f"| {r['backbone']} | {r['E_h']} | {p} | "
              f"{r['dev_auroc'] or 0:.4f} | {r['dev_f1'] or 0:.4f} | "
              f"{r['test_auroc'] or 0:.4f} | {r['test_f1'] or 0:.4f} |")


if __name__ == "__main__":
    main()
