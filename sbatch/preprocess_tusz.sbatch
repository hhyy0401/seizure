#!/bin/bash
#SBATCH -J tusz_prep
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -p cpu-medium
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz_prep_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz_prep_%j.err

set -euo pipefail

# TUSZ v2.0.6 preprocessing — 2 stages (markers already shipped with repo):
#   Stage 1: resample to 200 Hz                    → tusz_resampled/ (~128 GB)
#   Stage 2: build FFT clip h5 files (12s & 60s)   → tusz_preproc/   (~66 GB)
# Override TUSZ_VER for a different version (file_markers may need regen).

TUSZ_VER=${TUSZ_VER:-v2.0.6}
REPO=/storage/project/r-nimam6-0/hkim3239/disease/evobrain
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/$TUSZ_VER
RAW_EDF=$RAW/edf
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PREPROC=/storage/scratch1/3/hkim3239/eeg/tusz_preproc
PY=/storage/home/hcoda1/3/hkim3239/miniconda3/envs/fastenv/bin/python
mkdir -p "$RESAMPLED" "$PREPROC" /storage/scratch1/3/hkim3239/logs

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Repo:       $REPO"
echo "Raw EDF:    $RAW_EDF"
echo "Resampled:  $RESAMPLED"
echo "Preproc:    $PREPROC"

if [[ ! -d "$RAW_EDF" ]]; then
    echo "ERROR: $RAW_EDF not found — run rsync_tusz_nedc.sbatch first" >&2
    exit 1
fi
echo "raw EDF count: $(find "$RAW_EDF" -name '*.edf' | wc -l)"

cd "$REPO"
export PYTHONPATH=.

echo "=== Stage 1/2: resample to 200 Hz ==="
$PY data/resample_signals.py \
    --raw_edf_dir "$RAW_EDF" \
    --save_dir   "$RESAMPLED"

echo "  resampled h5 count: $(find "$RESAMPLED" -name '*.h5' | wc -l)"
du -sh "$RESAMPLED"

echo "=== Stage 2a: preprocess_detection (clip_len=12, with FFT) ==="
$PY data/preprocess_detection.py \
    --resampled_dir "$RESAMPLED" \
    --raw_data_dir  "$RAW" \
    --output_dir    "$PREPROC" \
    --clip_len      12 \
    --time_step_size 1 \
    --is_fft

echo "=== Stage 2b: preprocess_detection (clip_len=60, with FFT) ==="
$PY data/preprocess_detection.py \
    --resampled_dir "$RESAMPLED" \
    --raw_data_dir  "$RAW" \
    --output_dir    "$PREPROC" \
    --clip_len      60 \
    --time_step_size 1 \
    --is_fft

echo "=== summary ==="
du -sh "$RESAMPLED" "$PREPROC"
echo "preproc h5 count: $(find "$PREPROC" -name '*.h5' | wc -l)"
echo "End: $(date)"
