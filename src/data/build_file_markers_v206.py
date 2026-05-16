"""Generate file_markers_detection/*.txt for TUSZ v2.0.6.

Walks the v2.0.6 EDF tree, parses .csv_bi annotations to find seizure
intervals, and for each (clip_len) writes per-clip lines into
{split}Set_seq2seq_{clip_len}s_{sz,nosz}.txt matching the format the
EvoBrain detection dataloader expects.

Output line format: {edf_basename}.edf_{clip_idx}.h5,{label}
  - edf_basename: e.g. aaaaagus_s001_t000  (no path, no extension)
  - clip_idx: int, floor(duration / clip_len) clips per EDF
  - label: 1 if clip overlaps any seizure interval, else 0

TUSZ v2.0.6 split layout under raw_data_dir/edf/: train, dev, eval.
EvoBrain calls the third split 'test', so eval -> testSet here.

Usage:
  python data/build_file_markers_v206.py \
      --raw_data_dir ~/eeg/tusz/v2.0.6 \
      --out_dir data/file_markers_detection_v206 \
      --clip_lens 12 60
"""
import argparse
import os
from pathlib import Path


def parse_csv_bi(csv_bi_path):
    duration = None
    seizures = []
    with open(csv_bi_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# duration"):
                duration = float(line.split("=", 1)[1].strip().split()[0])
                continue
            if not line or line.startswith("#") or line.startswith("channel"):
                continue
            parts = line.split(",")
            if len(parts) >= 4 and parts[3].strip().lower() == "seiz":
                seizures.append((float(parts[1]), float(parts[2])))
    return duration, seizures


def clip_has_seizure(clip_start, clip_end, seizures):
    for s_start, s_end in seizures:
        if not (clip_end < s_start or clip_start > s_end):
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data_dir", required=True,
                   help="TUSZ v2.0.6 root (containing edf/ subdir)")
    p.add_argument("--out_dir", required=True,
                   help="Output dir for *Set_seq2seq_Ns_{sz,nosz}.txt")
    p.add_argument("--clip_lens", type=int, nargs="+", default=[12, 60])
    args = p.parse_args()

    root = Path(args.raw_data_dir).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    edf_root = root / "edf" if (root / "edf").exists() else root
    split_map = {"train": "trainSet", "dev": "devSet", "eval": "testSet"}

    for split_dir, prefix in split_map.items():
        split_path = edf_root / split_dir
        if not split_path.exists():
            print(f"[skip] {split_path} not found")
            continue
        edfs = sorted(split_path.rglob("*.edf"))
        print(f"[{split_dir}] {len(edfs)} EDFs")

        for clip_len in args.clip_lens:
            sz_lines, nosz_lines = [], []
            n_no_csv = 0
            for edf in edfs:
                csv_bi = edf.with_suffix(".csv_bi")
                if not csv_bi.exists():
                    n_no_csv += 1
                    continue
                duration, seizures = parse_csv_bi(csv_bi)
                if duration is None:
                    continue
                n_clips = int(duration // clip_len)
                base = edf.name  # "aaaaagus_s001_t000.edf"
                for ci in range(n_clips):
                    cs, ce = ci * clip_len, (ci + 1) * clip_len
                    label = 1 if clip_has_seizure(cs, ce, seizures) else 0
                    line = f"{base}_{ci}.h5,{label}\n"
                    (sz_lines if label else nosz_lines).append(line)

            (out / f"{prefix}_seq2seq_{clip_len}s_sz.txt").write_text("".join(sz_lines))
            (out / f"{prefix}_seq2seq_{clip_len}s_nosz.txt").write_text("".join(nosz_lines))
            print(f"  clip_len={clip_len}s -> sz={len(sz_lines)}, nosz={len(nosz_lines)}"
                  f"{f' (skipped {n_no_csv} edfs w/o csv_bi)' if n_no_csv else ''}")


if __name__ == "__main__":
    main()
