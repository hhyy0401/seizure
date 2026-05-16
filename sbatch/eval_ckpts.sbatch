#!/bin/bash
#SBATCH -J eval_ckpts
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-2
#SBATCH -o /storage/scratch1/3/hkim3239/logs/eval_ckpts_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/eval_ckpts_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Evaluate the 3 final ckpts in repo (one per dataset/window).
CKPT_DIRS=(
    "/storage/project/r-nimam6-0/hkim3239/disease/ckpts/tusz12"
    "/storage/project/r-nimam6-0/hkim3239/disease/ckpts/tusz60"
    "/storage/project/r-nimam6-0/hkim3239/disease/ckpts/chb12"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
CKPT_DIR=${CKPT_DIRS[$IDX]}

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src

echo "Host: $(hostname)  CKPT_DIR=$CKPT_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

# finalize_runs.py: rebuilds model + dataloaders from args.json, loads
# best.pth.tar, runs dev/test eval, dumps {dev,test}_results.npz alongside.
python scripts/finalize_runs.py "$CKPT_DIR"

echo "End: $(date)"
ls "$CKPT_DIR"
