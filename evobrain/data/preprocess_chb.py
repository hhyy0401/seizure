"""
Resample CHB-MIT EDF recordings to 200 Hz and save as h5 files, restricted to
the 18 standard bipolar channels (constants_chb.INCLUDED_CHANNELS_CHB).

Output naming matches the TUSZ preprocessing convention used by the EvoBrain
dataloader: <save_dir>/<edf_basename>.h5 with datasets:
    - resampled_signal: float32 array, shape (18, num_samples_at_200Hz)
    - resample_freq:    int, the target frequency (200)
    - channels:         JSON-encoded list of channel names (for verification)

Usage:
    python data/preprocess_chb.py \
        --raw_dir /storage/scratch1/.../chb_mit \
        --save_dir /storage/scratch1/.../chb_mit_resampled
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyedflib
from scipy.signal import resample
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from constants_chb import (
    CHB_PATIENTS,
    CHB_RAW_FREQ,
    FREQUENCY,
    INCLUDED_CHANNELS_CHB,
    NUM_NODES_CHB,
)


def normalize_label(s: str) -> str:
    # CHB-MIT labels sometimes carry trailing spaces or version suffixes.
    # Strip whitespace and any trailing "-N" version tag (rare).
    return s.strip().upper()


def find_channel_indices(edf_labels):
    """Return the EDF channel indices corresponding to INCLUDED_CHANNELS_CHB,
    or raise if any are missing."""
    norm_to_idx = {}
    for i, lbl in enumerate(edf_labels):
        key = normalize_label(lbl)
        if key not in norm_to_idx:
            norm_to_idx[key] = i
    idxs = []
    missing = []
    for ch in INCLUDED_CHANNELS_CHB:
        key = normalize_label(ch)
        if key not in norm_to_idx:
            missing.append(ch)
        else:
            idxs.append(norm_to_idx[key])
    if missing:
        raise KeyError(f"Missing channels: {missing}; available: {edf_labels}")
    return idxs


def list_patient_edfs(raw_dir: Path, patients=None):
    if patients is None:
        patients = CHB_PATIENTS
    edfs = []
    for patient in patients:
        pdir = raw_dir / patient
        if not pdir.is_dir():
            continue
        for f in sorted(pdir.iterdir()):
            if f.is_file() and f.suffix == ".edf":
                edfs.append(f)
    return edfs


def resample_one(edf_path: Path, save_dir: Path) -> str:
    save_path = save_dir / (edf_path.stem + ".h5")
    if save_path.exists():
        return "skip"

    reader = None
    try:
        reader = pyedflib.EdfReader(str(edf_path))
        labels = reader.getSignalLabels()
        idxs = find_channel_indices(labels)

        sample_freq = reader.getSampleFrequency(0)
        n_samples = reader.getNSamples()[0]

        signals = np.zeros((NUM_NODES_CHB, n_samples), dtype=np.float32)
        for out_i, edf_i in enumerate(idxs):
            signals[out_i, :] = reader.readSignal(edf_i)
    finally:
        if reader is not None:
            reader.close()

    if sample_freq != FREQUENCY:
        duration_s = n_samples / sample_freq
        num_out = int(round(FREQUENCY * duration_s))
        signals = resample(signals, num=num_out, axis=1).astype(np.float32)

    tmp_path = save_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as hf:
        hf.create_dataset("resampled_signal", data=signals)
        hf.create_dataset("resample_freq", data=FREQUENCY)
        hf.create_dataset("channels",
                          data=json.dumps(INCLUDED_CHANNELS_CHB).encode("utf-8"))
    tmp_path.replace(save_path)
    return "ok"


def main():
    parser = argparse.ArgumentParser("Resample CHB-MIT to 200Hz h5")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="CHB-MIT raw dir containing chbXX/ subfolders")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output dir for h5 files (flat)")
    parser.add_argument("--patients", type=str, nargs="*", default=None,
                        help="Subset of patient ids (e.g. chb01 chb02). "
                             "Default: all 22 from CHB_PATIENTS.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    edfs = list_patient_edfs(raw_dir, patients=args.patients)
    print(f"Found {len(edfs)} EDFs across {len(CHB_PATIENTS)} patients")

    ok, skipped, failed = 0, 0, []
    t0 = time.time()
    for edf in tqdm(edfs):
        try:
            status = resample_one(edf, save_dir)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
        except Exception as e:
            failed.append((str(edf), str(e)))
            print(f"FAIL {edf}: {e}", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  ok={ok}  skipped={skipped}  failed={len(failed)}")
    if failed:
        print("Failed files:")
        for fp, err in failed:
            print(f"  {fp}: {err}")


if __name__ == "__main__":
    main()
