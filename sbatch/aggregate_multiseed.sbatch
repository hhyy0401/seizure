#!/bin/bash
#SBATCH -J ms_aggr
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/ms_aggr_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/ms_aggr_%j.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

OUT_FILE="$REPO/MULTISEED_RESULTS.md"
{
    echo "# Multi-seed Results — generated $(date)"
    echo
    echo "Top-3 distinct cfgs × 3 seeds × 3 datasets (TUSZ 12s / TUSZ 60s / CHB-MIT 12s)."
    echo
    echo '```'
    python src/scripts/aggregate_multiseed.py
    echo '```'
} > "$OUT_FILE"

echo "Wrote $OUT_FILE"
ls -la "$OUT_FILE"
