"""Chunked wrapper around resample_signals.resample_all so N workers split work.

Each worker processes only the slice
    edf_files[ (N*idx)//K : (N*(idx+1))//K ]
preventing the 8x duplicate-write race of running resample_signals.py 8 times.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import pyedflib
import scipy
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import INCLUDED_CHANNELS, FREQUENCY
from data_utils import resampleData, getEDFsignals, getOrderedChannels


def main(raw_edf_dir, save_dir, chunk_idx, n_chunks):
    os.makedirs(save_dir, exist_ok=True)
    edf_files = []
    for path, _, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))
    edf_files.sort()  # deterministic chunking across workers

    n = len(edf_files)
    start = (n * chunk_idx) // n_chunks
    end = (n * (chunk_idx + 1)) // n_chunks
    my = edf_files[start:end]
    print(f"[chunk {chunk_idx}/{n_chunks}] total={n} slice=[{start}:{end}) size={len(my)}")

    skipped = failed = written = 0
    for edf_fn in tqdm(my, desc=f"chunk{chunk_idx}"):
        save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            skipped += 1
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)
            orderedChannels = getOrderedChannels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            signals = getEDFsignals(f)
            signal_array = np.array(signals[orderedChannels, :])
            sample_freq = f.getSampleFrequency(0)
            if sample_freq != FREQUENCY:
                signal_array = resampleData(
                    signal_array,
                    to_freq=FREQUENCY,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )
            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
                hf.create_dataset("resample_freq", data=FREQUENCY)
            written += 1
        except Exception as e:
            failed += 1
            print(f"  fail {edf_fn}: {e}")
            continue
        finally:
            try:
                f._close()
            except Exception:
                pass

    print(f"[chunk {chunk_idx}/{n_chunks}] DONE — written={written} "
          f"skipped={skipped} failed={failed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_edf_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--chunk_idx", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, required=True)
    args = ap.parse_args()
    main(args.raw_edf_dir, args.save_dir, args.chunk_idx, args.n_chunks)
