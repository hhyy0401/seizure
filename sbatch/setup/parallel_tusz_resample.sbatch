#!/bin/bash
#SBATCH -J tusz_resamp
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -p cpu-small
#SBATCH --array=0-7
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz_resamp_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz_resamp_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# resample_signals.py already skips files with `if os.path.exists(save_fn): continue`,
# so running N copies in parallel naturally divides remaining work between them.

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"
export PYTHONPATH=.

N_CHUNKS=8
IDX=${SLURM_ARRAY_TASK_ID}

echo "Worker $IDX/$N_CHUNKS  start: $(date)"
echo "  resampled-so-far: $(ls $RESAMPLED 2>/dev/null | wc -l)"

python data/resample_signals_chunk.py \
    --raw_edf_dir "$RAW/edf" \
    --save_dir   "$RESAMPLED" \
    --chunk_idx  "$IDX" \
    --n_chunks   "$N_CHUNKS"

echo "Worker $SLURM_ARRAY_TASK_ID  end: $(date)"
echo "  resampled-now: $(ls $RESAMPLED 2>/dev/null | wc -l)"
