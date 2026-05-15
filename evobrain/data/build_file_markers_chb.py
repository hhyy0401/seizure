"""
Generate file-marker text files for CHB-MIT seizure detection, matching the
naming convention the EvoBrain dataloader expects:

    {trainSet,devSet,testSet}_seq2seq_{clip_len}s_{sz,nosz}.txt

Each line: <edf_basename>.edf_<clip_idx>.h5,<label>

Steps per patient/EDF:
  1) parse chb<NN>-summary.txt for seizure (start, end) windows
  2) compute number of clips = floor(duration_h5 / clip_len)
  3) label each clip 1 if it overlaps any seizure window, else 0
  4) per-patient random 70/15/15 split of clip indices (across all EDFs of that
     patient) using a fixed seed — matches the EvoBrain paper convention of
     "randomly selected 15% of the patient's data for testing"

Usage:
    python data/build_file_markers_chb.py \
        --raw_dir   /storage/scratch1/.../chb_mit \
        --input_dir /storage/scratch1/.../chb_mit_resampled \
        --out_dir   data/file_markers_chb \
        --clip_len  12 \
        --seed      123
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from constants_chb import CHB_PATIENTS, FREQUENCY


def time_to_seconds(text: str) -> float:
    """Parse 'NNN seconds' or 'NNN' to float seconds."""
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        raise ValueError(f"Cannot parse seconds from: {text!r}")
    return float(m.group(1))


def parse_summary(summary_path: Path):
    """Return {edf_basename_no_ext: [(start_s, end_s), ...]} for all EDFs
    listed in the summary file. Files with no seizures map to [].

    Robust to two formats:
      "Seizure Start Time: 2996 seconds"  (single seizure)
      "Seizure 1 Start Time: 130 seconds" (multi seizure)
    """
    out = {}
    current = None
    pending_starts = {}  # idx -> start (idx may be None when single seizure)

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("File Name:"):
                current = line.split(":", 1)[1].strip()
                if current.endswith(".edf"):
                    current = current[:-4]
                out[current] = []
                pending_starts = {}
                continue
            if current is None:
                continue
            m = re.match(
                r"Seizure(?:\s+(\d+))?\s+Start Time\s*:\s*(.+)$", line
            )
            if m:
                idx = m.group(1)  # None for single-seizure format
                pending_starts[idx] = time_to_seconds(m.group(2))
                continue
            m = re.match(
                r"Seizure(?:\s+(\d+))?\s+End Time\s*:\s*(.+)$", line
            )
            if m:
                idx = m.group(1)
                end_s = time_to_seconds(m.group(2))
                if idx in pending_starts:
                    out[current].append((pending_starts.pop(idx), end_s))
                else:
                    # tolerate missing matching start (shouldn't happen)
                    out[current].append((max(0.0, end_s - 1.0), end_s))
    return out


def num_clips_from_h5(h5_path: Path, clip_len: int) -> int:
    with h5py.File(h5_path, "r") as f:
        n = f["resampled_signal"].shape[1]
    return int(n // (FREQUENCY * clip_len))


def clip_overlaps_seizure(clip_idx: int, clip_len: int, sz_windows):
    """sz_windows in seconds; clip occupies [clip_idx*clip_len, (clip_idx+1)*clip_len)."""
    start = clip_idx * clip_len
    end = start + clip_len
    for s, e in sz_windows:
        if not (end <= s or start >= e):
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="CHB-MIT raw dir containing chbXX/ subfolders")
    ap.add_argument("--input_dir", required=True,
                    help="Dir with resampled h5 files (output of preprocess_chb.py)")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write {split}_seq2seq_{L}s_{sz,nosz}.txt files")
    ap.add_argument("--clip_len", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--patients", type=str, nargs="*", default=None,
                    help="Subset of patient ids. Default: all 22 from CHB_PATIENTS.")
    args = ap.parse_args()
    patient_list = args.patients if args.patients else CHB_PATIENTS

    raw_dir = Path(args.raw_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # all_records: {split: {"sz": [...], "nosz": [...]}}
    buckets = {s: {"sz": [], "nosz": []} for s in ("train", "dev", "test")}

    summary_missing = []
    for patient in patient_list:
        pdir = raw_dir / patient
        summary = pdir / f"{patient}-summary.txt"
        if not summary.exists():
            summary_missing.append(patient)
            continue

        sz_map = parse_summary(summary)

        # Build per-clip records for this patient across all EDFs
        records = []  # list of (marker_str, label)
        for edf_base, sz_windows in sz_map.items():
            h5_path = input_dir / f"{edf_base}.h5"
            if not h5_path.exists():
                continue
            n_clips = num_clips_from_h5(h5_path, args.clip_len)
            for ci in range(n_clips):
                label = clip_overlaps_seizure(ci, args.clip_len, sz_windows)
                marker = f"{edf_base}.edf_{ci}.h5,{label}"
                records.append((marker, label))

        if not records:
            continue

        # per-patient split
        order = rng.permutation(len(records))
        n = len(order)
        n_train = int(round(args.train_frac * n))
        n_dev = int(round(args.dev_frac * n))
        train_idx = order[:n_train]
        dev_idx = order[n_train:n_train + n_dev]
        test_idx = order[n_train + n_dev:]

        for split_name, idxs in (
            ("train", train_idx),
            ("dev", dev_idx),
            ("test", test_idx),
        ):
            for i in idxs:
                marker, label = records[i]
                key = "sz" if label == 1 else "nosz"
                buckets[split_name][key].append(marker)

    # Write outputs
    for split in ("train", "dev", "test"):
        for key in ("sz", "nosz"):
            fn = out_dir / f"{split}Set_seq2seq_{args.clip_len}s_{key}.txt"
            with open(fn, "w") as f:
                for line in buckets[split][key]:
                    f.write(line + "\n")
            print(f"wrote {fn} ({len(buckets[split][key])} lines)")

    if summary_missing:
        print(f"WARNING: missing summary.txt for patients: {summary_missing}")


if __name__ == "__main__":
    main()
