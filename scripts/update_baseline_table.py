#!/usr/bin/env python3
"""
Hourly Phoenix-side updater for docs/BASELINE_TABLE.md.

Scans /storage/scratch1/3/hkim3239/eeg/runs/ for baseline runs of the form
    {model}_{dataset}{clip}_s{seed}_{jobid}
parses each run's TEST set prediction results from info.log, aggregates over
seeds, and rewrites the "Current table" + "Aggregated CHB-MIT 12s detail" +
"Last updated" + "Job board" sections of BASELINE_TABLE.md.

Constraints:
- Read-only on the SLURM/run data. Only writes BASELINE_TABLE.md.
- Preserves the externally-supplied CHB s123 values for LSTM/BIOT/DCRNN
  (stored in the EXTERNAL_S123 dict below and in a manually-curated section
  at the top of the markdown file — that section is NEVER overwritten).
- No torch import, no GPU, no new sbatch submission.

Designed to run from a Phoenix login node under cron.
"""
import os
import re
import glob
import subprocess
from datetime import datetime
from statistics import mean, stdev

RUNS_ROOT = "/storage/scratch1/3/hkim3239/eeg/runs"
MD_PATH   = "/storage/project/r-nimam6-0/hkim3239/disease/docs/BASELINE_TABLE.md"
TEX_PATH  = "/storage/project/r-nimam6-0/hkim3239/disease/paper/main_results.tex"
USER      = "hkim3239"

# CHB-MIT 12s s123 — ran on an external machine, manually pasted by user.
# These are merged into the per-model aggregate when the local Phoenix runs
# only have s456/s789. Format: model_lower -> (auroc, f1).
EXTERNAL_S123 = {
    "lstm":  (0.884, 0.061),
    "biot":  (0.904, 0.000),
    "dcrnn": (0.883, 0.138),
}

MODELS = ("lstm", "cnnlstm", "BIOT", "dcrnn", "gru_gcn", "evolvegcn", "graphs4mer",
          "labram", "eegpt")

# --- LightSTHyper reference rows (do not recompute, just pin) ---
LSH_ROWS = [
    ("LightSTHyper (E_h=1)", "**0.898±0.003**", "**0.519±0.019**",
     "**0.877±0.017**", "**0.569±0.017**",
     "0.898±0.006", "0.142±0.007"),
    ("LightSTHyper (E_h=2)", "0.892±0.006", "0.440±0.023",
     "0.848±0.019", "0.463±0.040",
     "0.898±0.006", "**0.154±0.025**"),
    ("LightSTHyper (E_h=3)", "0.889±0.006", "0.504±0.029",
     "**0.877±0.005**", "0.537±0.039",
     "**0.904±0.007**", "0.134±0.021"),
]


def parse_test_line(log_path):
    """Return (auroc, f1) from the last 'TEST set prediction results' line,
    or None if the run hasn't finished testing yet."""
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path) as f:
            txt = f.read()
    except OSError:
        return None
    matches = [l for l in txt.splitlines() if "TEST set prediction results" in l]
    if not matches:
        return None
    line = matches[-1]
    m_auroc = re.search(r"auroc:\s*([0-9.]+)", line)
    m_f1    = re.search(r"F1:\s*([0-9.]+)", line)
    if not (m_auroc and m_f1):
        return None
    return float(m_auroc.group(1)), float(m_f1.group(1))


def gather_runs():
    """
    Walk RUNS_ROOT and return: dict[(model, dset, clip)] -> dict[seed] -> (auroc, f1, jobid)
    Keeps only the highest-jobid run per (model, dset, clip, seed).
    """
    pat = re.compile(
        r"^(?P<model>" + "|".join(MODELS) + r")"
        r"_(?P<dset>tusz|chbmit|chb)(?P<clip>12|60)"
        r"_s(?P<seed>\d+)"
        r"_(?P<jid>\d+)$"
    )
    buckets = {}
    if not os.path.isdir(RUNS_ROOT):
        return buckets
    for name in os.listdir(RUNS_ROOT):
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        # Normalize chbmit -> chb so groupings stay consistent.
        if d["dset"] == "chbmit":
            d["dset"] = "chb"
        logs = glob.glob(os.path.join(RUNS_ROOT, name, "**", "info.log"), recursive=True)
        if not logs:
            continue
        # Prefer the info.log that has a TEST line (handles the case where a
        # walltime-killed training run was resurrected by a separate test-only
        # sbatch that wrote a nested info.log).
        r = None
        for lp in sorted(logs, key=os.path.getmtime, reverse=True):
            r = parse_test_line(lp)
            if r is not None:
                break
        if r is None:
            continue
        key = (d["model"], d["dset"], d["clip"])
        seed = d["seed"]
        jid  = int(d["jid"])
        buckets.setdefault(key, {})
        prev = buckets[key].get(seed)
        if prev is None or jid > prev[2]:
            buckets[key][seed] = (r[0], r[1], jid)
    return buckets


def aggregate(buckets):
    """
    buckets: from gather_runs()
    Returns: dict[(model, dset, clip)] -> {"auroc_str", "f1_str", "n", "seeds_used"}
    Merges EXTERNAL_S123 for CHB-MIT 12s LSTM/BIOT/DCRNN when local s123 missing.
    """
    out = {}
    for key, runs in buckets.items():
        model, dset, clip = key
        aurocs = [runs[s][0] for s in runs]
        f1s    = [runs[s][1] for s in runs]
        seeds_used = sorted(runs.keys())
        # Merge external s123 for CHB LSTM/BIOT/DCRNN
        if dset == "chb" and clip == "12" and model.lower() in EXTERNAL_S123 and "123" not in runs:
            ext_auroc, ext_f1 = EXTERNAL_S123[model.lower()]
            aurocs.append(ext_auroc)
            f1s.append(ext_f1)
            seeds_used.append("123(ext)")
        n = len(aurocs)
        if n == 0:
            continue
        if n >= 2:
            a_str = f"{mean(aurocs):.3f}±{stdev(aurocs):.3f}"
            f_str = f"{mean(f1s):.3f}±{stdev(f1s):.3f}"
        else:
            a_str = f"{aurocs[0]:.3f}"
            f_str = f"{f1s[0]:.3f}"
        out[key] = {
            "auroc_str": a_str,
            "f1_str":    f_str,
            "n":         n,
            "seeds_used": seeds_used,
            "raw_auroc": aurocs,
            "raw_f1":    f1s,
        }
    return out


def cell(agg, ext_seed_used, prefer_bold_lsh_compare=None):
    """Render a markdown cell value for an aggregate entry."""
    if agg is None:
        return "—"
    n = agg["n"]
    suffix = f" *(n={n}{', w/ ext s123' if ext_seed_used else ''})*"
    return f"{agg['auroc_str']}{suffix}", f"{agg['f1_str']}{suffix}"


def render_current_table(agg_map):
    """Render the 'Current table' markdown block."""
    # row order matches the prior table
    rows_order = [
        ("LSTM",       "lstm"),
        ("CNN-LSTM",   "cnnlstm"),
        ("BIOT",       "BIOT"),
        ("LaBraM",     "labram"),
        ("EEGPT",      "eegpt"),
        ("EvolveGCN",  "evolvegcn"),
        ("DCRNN",      "dcrnn"),
        ("GRAPHS4MER", "graphs4mer"),
        ("GRU-GCN",    "gru_gcn"),
        ("EvoBrain",   None),
    ]
    lines = [
        "| Method | TUSZ 12s AUROC | TUSZ 12s F1 | TUSZ 60s AUROC | TUSZ 60s F1 | CHB-MIT 12s AUROC | CHB-MIT 12s F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for disp, key in rows_order:
        if key is None:
            lines.append(f"| {disp} | — | — | — | — | — | — |")
            continue
        cells = []
        for dset, clip in [("tusz", "12"), ("tusz", "60"), ("chb", "12")]:
            agg = agg_map.get((key, dset, clip))
            if agg is None:
                cells.extend(["—", "—"])
                continue
            ext = (dset == "chb" and clip == "12"
                   and key.lower() in EXTERNAL_S123
                   and "123(ext)" in agg["seeds_used"])
            n = agg["n"]
            suf = f" *(n={n}{', w/ ext s123' if ext else ''})*"
            cells.append(f"{agg['auroc_str']}{suf}")
            cells.append(f"{agg['f1_str']}{suf}")
        lines.append(f"| {disp:10s} | " + " | ".join(cells) + " |")
    # Append LightSTHyper reference rows
    for row in LSH_ROWS:
        lines.append("| **" + row[0] + "** | " + " | ".join(row[1:]) + " |")
    return "\n".join(lines)


def render_chb_detail(agg_map, raw_buckets):
    """Render the 'Aggregated CHB-MIT 12s detail' table."""
    lines = [
        "| Model | s123 (ext) | s456 | s789 | mean±std |",
        "|---|---|---|---|---|",
    ]
    for model_disp, model_key in [("LSTM", "lstm"), ("BIOT", "BIOT"), ("DCRNN", "dcrnn")]:
        runs = raw_buckets.get((model_key, "chb", "12"), {})
        ext = EXTERNAL_S123[model_key.lower()]
        s123_cell = f"{ext[0]:.3f} / {ext[1]:.3f}"
        def cell_for(s):
            r = runs.get(s)
            if r is None:
                return "—"
            return f"{r[0]:.3f} / {r[1]:.3f}"
        s456_cell = cell_for("456")
        s789_cell = cell_for("789")
        # mean over (ext, s456, s789) only when all three present
        agg = agg_map.get((model_key, "chb", "12"))
        if agg and agg["n"] >= 2:
            mean_cell = f"**{agg['auroc_str']} / {agg['f1_str']}**"
        else:
            mean_cell = "—"
        lines.append(f"| {model_disp}  | {s123_cell} | {s456_cell} | {s789_cell} | {mean_cell} |")
    return "\n".join(lines)


def get_squeue_snapshot():
    """Run squeue and bucket jobs by RUNNING vs PENDING."""
    try:
        out = subprocess.run(
            ["squeue", "-u", USER, "--format=%i|%j|%T|%M|%R", "--noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [], []
    running, pending = [], []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, state, t, reason = parts[0], parts[1], parts[2], parts[3], parts[4]
        entry = f"`{jid}` {name} ({t})"
        if state == "RUNNING":
            running.append(entry)
        elif state == "PENDING":
            pending.append(entry)
    return running, pending


def render_latex_main_table(agg_map):
    """
    Paper-ready LaTeX main results table.
    Bolds the per-column best (mean AUROC / F1), underlines the 2nd best.
    Excludes "still running" cells (—).
    """
    # row order — display name and key into agg_map; tuple of (label, key)
    rows = [
        ("LSTM",       "lstm"),
        ("CNN-LSTM",   "cnnlstm"),
        ("BIOT",       "BIOT"),
        ("LaBraM",     "labram"),
        ("EEGPT",      "eegpt"),
        ("EvolveGCN",  "evolvegcn"),
        ("DCRNN",      "dcrnn"),
        ("GRAPHS4MER", "graphs4mer"),
        ("GRU-GCN",    "gru_gcn"),
    ]
    # LightSTHyper rows pinned (best across own seeds; from RESULTS.md)
    lsh_rows = [
        ("LightSTHyper ($E_h{=}1$)",
         ("898", "3"), ("519", "19"),
         ("877", "17"), ("569", "17"),
         ("898", "6"), ("142", "7")),
        ("LightSTHyper ($E_h{=}2$)",
         ("892", "6"), ("440", "23"),
         ("848", "19"), ("463", "40"),
         ("898", "6"), ("154", "25")),
        ("LightSTHyper ($E_h{=}3$)",
         ("889", "6"), ("504", "29"),
         ("877", "5"), ("537", "39"),
         ("904", "7"), ("134", "21")),
    ]
    # Build a list of all rows with parsed (auroc, f1) tuples for each column.
    # Columns: (tusz12_auroc, tusz12_f1, tusz60_auroc, tusz60_f1, chb12_auroc, chb12_f1)
    def parse_mean_std(s):
        """'0.898±0.003' -> (0.898, 0.003); '0.879' -> (0.879, None)"""
        if s in (None, "—"):
            return None
        m = re.match(r"([0-9.]+)±([0-9.]+)", s)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        try:
            return (float(s), None)
        except ValueError:
            return None

    table_rows = []   # list of (label, [(mean,std) per col, ...])
    for label, key in rows:
        cols = []
        for dset, clip in [("tusz", "12"), ("tusz", "60"), ("chb", "12")]:
            a = agg_map.get((key, dset, clip))
            if a is None:
                cols.extend([None, None])
                continue
            # Skip n=1 columns (no std → hide to avoid misleading the reader)
            if a["n"] < 2:
                cols.extend([None, None])
                continue
            cols.append(parse_mean_std(a["auroc_str"]))
            cols.append(parse_mean_std(a["f1_str"]))
        table_rows.append((label, cols))
    # Append LightSTHyper rows (already in mean±std form, stored as ints x1000)
    for lsh in lsh_rows:
        label = lsh[0]
        cols = []
        for i in range(1, 7):
            mean_s, std_s = lsh[i]
            cols.append((int(mean_s) / 1000.0, int(std_s) / 1000.0))
        table_rows.append((label, cols))

    # Find best (max) and 2nd-best per column
    n_cols = 6
    best = [None] * n_cols
    second = [None] * n_cols
    for c in range(n_cols):
        vals = [(r_idx, table_rows[r_idx][1][c][0])
                for r_idx in range(len(table_rows))
                if table_rows[r_idx][1][c] is not None]
        if not vals:
            continue
        vals.sort(key=lambda x: -x[1])
        best[c] = vals[0][0]
        if len(vals) > 1 and vals[1][1] != vals[0][1]:
            second[c] = vals[1][0]

    def fmt_cell(val, c, r):
        if val is None:
            return "---"
        mean_v, std_v = val
        if std_v is None:
            base = f"{mean_v:.3f}"
        else:
            base = f"{mean_v:.3f}$\\pm${std_v:.3f}"
        if r == best[c]:
            return r"\textbf{" + base + "}"
        if r == second[c]:
            return r"\underline{" + base + "}"
        return base

    lines = []
    lines.append("% Auto-generated by scripts/update_baseline_table.py")
    lines.append("% Source: Phoenix runs under /storage/scratch1/3/hkim3239/eeg/runs/")
    lines.append("% TUSZ v2.0.6 (v1.5.2 is discontinued per Kotoge et al. README).")
    lines.append("% CHB-MIT 12s s123 for LSTM/BIOT/DCRNN: externally supplied by user.")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Method & \\multicolumn{2}{c}{TUSZ 12s} & \\multicolumn{2}{c}{TUSZ 60s} & \\multicolumn{2}{c}{CHB-MIT 12s} \\\\")
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    lines.append(" & AUROC & F1 & AUROC & F1 & AUROC & F1 \\\\")
    lines.append("\\midrule")
    for r_idx, (label, cols) in enumerate(table_rows):
        if "LightSTHyper" in label and r_idx > 0 and "LightSTHyper" not in table_rows[r_idx - 1][0]:
            lines.append("\\midrule")
        cells = [fmt_cell(cols[c], c, r_idx) for c in range(n_cols)]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def render_job_board():
    running, pending = get_squeue_snapshot()
    lines = ["### Running"]
    if running:
        for e in running:
            lines.append(f"- {e}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### Pending")
    if pending:
        for e in pending:
            lines.append(f"- {e}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def replace_section(md, header, new_body):
    """
    Replace the body under a markdown header (## level) with new_body, up to the
    next ## header or EOF. If header missing, append to file.
    """
    pat = re.compile(
        r"(^## " + re.escape(header) + r"\s*\n)(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if not pat.search(md):
        return md.rstrip() + f"\n\n## {header}\n\n{new_body}\n"
    return pat.sub(lambda m: m.group(1) + "\n" + new_body.rstrip() + "\n\n", md)


def main():
    buckets = gather_runs()
    agg = aggregate(buckets)

    current_tbl = render_current_table(agg)
    chb_detail  = render_chb_detail(agg, buckets)
    job_board   = render_job_board()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    if not ts.split()[-1].isalpha():
        ts = datetime.now().strftime("%Y-%m-%d %H:%M (local)")

    with open(MD_PATH) as f:
        md = f.read()

    md = replace_section(md, "Last updated",         ts)
    md = replace_section(md, "Current table (TUSZ + CHB-MIT)", current_tbl)
    md = replace_section(md, "Aggregated CHB-MIT 12s detail", chb_detail)
    md = replace_section(md, "Job board (snapshot, auto-updated)", job_board)

    with open(MD_PATH, "w") as f:
        f.write(md)

    # Also write a paper-ready LaTeX snippet.
    latex_tbl = render_latex_main_table(agg)
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)
    with open(TEX_PATH, "w") as f:
        f.write(latex_tbl + "\n")

    print(f"[{ts}] updated {MD_PATH}")
    print(f"[{ts}] updated {TEX_PATH}")
    print(f"  groups aggregated: {len(agg)}")
    for key, a in sorted(agg.items()):
        print(f"    {key}: n={a['n']}  AUROC={a['auroc_str']}  F1={a['f1_str']}")


if __name__ == "__main__":
    main()
