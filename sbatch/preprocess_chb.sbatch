#!/bin/bash
#SBATCH -J chb_prep
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/chb_prep_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/chb_prep_%j.err

set -euo pipefail

RAW=/storage/scratch1/3/hkim3239/data/chb_mit
H5=/storage/scratch1/3/hkim3239/data/chb_mit_resampled
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
PY=/storage/home/hcoda1/3/hkim3239/miniconda3/envs/fastenv/bin/python
mkdir -p /storage/scratch1/3/hkim3239/logs

echo "Host: $(hostname)"
echo "Start: $(date)"
cd "$REPO"

# Stage 1: resample all 22 patients to 200 Hz, 18 bipolar channels
$PY data/preprocess_chb.py \
    --raw_dir "$RAW" \
    --save_dir "$H5"

# Stage 2: build file markers for 12s clips (EvoBrain main setting)
$PY data/build_file_markers_chb.py \
    --raw_dir "$RAW" \
    --input_dir "$H5" \
    --out_dir data/file_markers_chb \
    --clip_len 12 \
    --seed 123

# Stage 3: also build 60s clips (paper also reports 60s)
$PY data/build_file_markers_chb.py \
    --raw_dir "$RAW" \
    --input_dir "$H5" \
    --out_dir data/file_markers_chb \
    --clip_len 60 \
    --seed 123

echo "--- h5 dir size ---"
du -sh "$H5"
echo "--- h5 file count ---"
ls "$H5" | wc -l

echo "--- 12s marker counts ---"
for s in train dev test; do for k in sz nosz; do
    f="data/file_markers_chb/${s}Set_seq2seq_12s_${k}.txt"
    echo "$f: $(wc -l < "$f")"
done; done

echo "--- 60s marker counts ---"
for s in train dev test; do for k in sz nosz; do
    f="data/file_markers_chb/${s}Set_seq2seq_60s_${k}.txt"
    echo "$f: $(wc -l < "$f")"
done; done

echo "End: $(date)"
