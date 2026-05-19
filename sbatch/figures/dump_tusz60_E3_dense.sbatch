#!/bin/bash
#SBATCH -J dump_tusz60_E3
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -p gpu-rtx6000
#SBATCH -o /storage/scratch1/3/hkim3239/logs/dump_tusz60_E3_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/dump_tusz60_E3_%j.err

# Dump M + dense (per-second) labels from TUSZ60 E=3 best ckpt (seed 123).
# Used by scripts/viz_onset_edge.py to plot onset-anchored figures.
set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6/edf
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO/src"

mkdir -p "$REPO/figures/dumps"
python scripts/dump_membership.py \
    "$REPO/ckpts/tusz60_E3_noaux_s123" \
    "$REPO/figures/dumps/tusz60_E3_s123_dense.npz" \
    --raw_data_dir "$RAW" --input_dir "$RESAMPLED" \
    --dense

echo "End: $(date)"
ls -lh "$REPO/figures/dumps/tusz60_E3_s123_dense.npz"
