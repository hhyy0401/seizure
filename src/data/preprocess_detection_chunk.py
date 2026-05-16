"""Chunked + skip-existing wrapper around preprocess_detection.main loop.

Splits the full clip list by (--chunk_idx, --n_chunks) so N parallel workers
divide the work evenly. Skips any output file that already exists, so reruns
and overlapping submissions are idempotent.

Run with the same CLI as preprocess_detection.py, plus:
    --chunk_idx 0  --n_chunks 8
"""
import argparse
import os
import sys

import h5py
from tqdm import tqdm

# Make `dataloader_detection` importable when run from the data/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader_detection import computeSliceMatrix

FILE_MARKER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "file_markers_detection"
)


def main(resample_dir, raw_data_dir, output_dir, clip_len,
         time_step_size, is_fft, chunk_idx, n_chunks):
    # Load file markers (same as preprocess_detection.py)
    def _load(fn):
        with open(os.path.join(FILE_MARKER_DIR, fn), "r") as f:
            return [s.strip("\n").split(",") for s in f.readlines()]

    train_tuples = (_load(f"trainSet_seq2seq_{clip_len}s_sz.txt")
                    + _load(f"trainSet_seq2seq_{clip_len}s_nosz.txt"))
    dev_tuples = (_load(f"devSet_seq2seq_{clip_len}s_sz.txt")
                  + _load(f"devSet_seq2seq_{clip_len}s_nosz.txt"))
    test_tuples = (_load(f"testSet_seq2seq_{clip_len}s_sz.txt")
                   + _load(f"testSet_seq2seq_{clip_len}s_nosz.txt"))
    all_tuples = train_tuples + dev_tuples + test_tuples

    # Slice this worker's chunk
    n = len(all_tuples)
    start = (n * chunk_idx) // n_chunks
    end = (n * (chunk_idx + 1)) // n_chunks
    my = all_tuples[start:end]
    print(f"[chunk {chunk_idx}/{n_chunks}] total={n}  slice=[{start}:{end}) "
          f"size={len(my)}")

    # Mirror preprocess_detection.py's output layout
    output_dir = os.path.join(
        output_dir, f"clipLen{clip_len}_timeStepSize{time_step_size}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Pre-scan edf files for path lookup
    edf_files = []
    for path, _, files in os.walk(raw_data_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    skipped = 0
    written = 0
    failed = 0
    for h5_fn_raw, _ in tqdm(my, desc=f"chunk{chunk_idx}"):
        edf_stem = h5_fn_raw.split(".edf")[0]
        clip_idx = int(h5_fn_raw.split("_")[-1].split(".h5")[0])
        out_fn = os.path.join(output_dir, f"{edf_stem}.edf_{clip_idx}.h5")
        if os.path.exists(out_fn):
            skipped += 1
            continue

        edf_matches = [f for f in edf_files if edf_stem + ".edf" in f]
        if not edf_matches:
            failed += 1
            continue
        try:
            eeg_clip, _ = computeSliceMatrix(
                h5_fn=os.path.join(resample_dir, edf_stem + ".h5"),
                edf_fn=edf_matches[0],
                clip_idx=clip_idx,
                time_step_size=time_step_size,
                clip_len=clip_len,
                is_fft=is_fft,
            )
        except (FileNotFoundError, OSError) as e:
            # truncated h5 from parallel-resample race, missing edf, etc.
            # Skip the clip; we keep the run going.
            print(f"  skip bad input for {edf_stem}: {e}")
            failed += 1
            continue
        with h5py.File(out_fn, "w") as hf:
            hf.create_dataset("clip", data=eeg_clip)
        written += 1

    print(f"[chunk {chunk_idx}/{n_chunks}] DONE — "
          f"written={written}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resampled_dir", required=True)
    ap.add_argument("--raw_data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clip_len", type=int, default=12)
    ap.add_argument("--time_step_size", type=int, default=1)
    ap.add_argument("--is_fft", action="store_true")
    ap.add_argument("--chunk_idx", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, required=True)
    args = ap.parse_args()
    main(args.resampled_dir, args.raw_data_dir, args.output_dir,
         args.clip_len, args.time_step_size, args.is_fft,
         args.chunk_idx, args.n_chunks)
