#!/bin/bash
#SBATCH -J slim_ck
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0:10:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/slim_ck_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/slim_ck_%j.err

set -euo pipefail
ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

PRE=/storage/scratch1/3/hkim3239/eeg/pretrained
python /storage/project/r-nimam6-0/hkim3239/disease/scripts/slim_eegpt_ckpt.py \
    --in  "$PRE/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt" \
    --out "$PRE/eegpt/eegpt_mcae_58chs_4s_large4E_slim_fp16.pt" --fp16

ls -lh "$PRE/eegpt/"
