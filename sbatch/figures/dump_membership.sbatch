#!/bin/bash
#SBATCH -J dump_M
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0:45:00
#SBATCH -p gpu-rtx6000
#SBATCH -o /storage/scratch1/3/hkim3239/logs/dump_M_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/dump_M_%j.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO/src"

mkdir -p "$REPO/figures"
python scripts/dump_membership.py \
    "$REPO/ckpts/tusz12" \
    "$REPO/figures/membership_tusz12.npz"

echo "End: $(date)"
ls -lh "$REPO/figures/membership_tusz12.npz"
