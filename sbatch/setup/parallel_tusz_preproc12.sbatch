#!/bin/bash
#SBATCH -J tusz_pre12
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -p cpu-small
#SBATCH --array=0-7
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz_pre12_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz_pre12_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PREPROC=/storage/scratch1/3/hkim3239/eeg/tusz_preproc

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"
export PYTHONPATH=.

N_CHUNKS=8
IDX=${SLURM_ARRAY_TASK_ID}

echo "Worker $IDX/$N_CHUNKS  start: $(date)"

python data/preprocess_detection_chunk.py \
    --resampled_dir "$RESAMPLED" \
    --raw_data_dir  "$RAW" \
    --output_dir    "$PREPROC" \
    --clip_len      12 \
    --time_step_size 1 \
    --is_fft \
    --chunk_idx     "$IDX" \
    --n_chunks      "$N_CHUNKS"

echo "Worker $IDX  end: $(date)"
echo "  preproc-now: $(find $PREPROC -name '*.h5' | wc -l)"
