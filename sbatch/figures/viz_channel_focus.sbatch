#!/bin/bash
#SBATCH -J viz_chan
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0:15:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/viz_chan_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/viz_chan_%j.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO/src"

python scripts/viz_channel_focus.py \
    "$REPO/figures/membership_tusz12.npz" \
    "$REPO/figures/channel"

echo "End: $(date)"
ls -la "$REPO/figures/channel/"
