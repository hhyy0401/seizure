#!/bin/bash
#SBATCH -J refresh_ckpts
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 0:05:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/refresh_ckpts_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/refresh_ckpts_%j.err

# Copy the (E_h=1, seed=123) best.pth.tar from the final sweep into
# ckpts/{tusz12,tusz60,chb12}/ so the shipped checkpoints match the
# final config. Override with --export=ALL,E_H=...,SEED=...

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

E_H="${E_H:-1}"
SEED="${SEED:-123}"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"
python src/scripts/refresh_ckpts.py --e-h "$E_H" --seed "$SEED"
