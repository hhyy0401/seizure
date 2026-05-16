#!/bin/bash
#SBATCH -J finalize
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-31
#SBATCH -o /storage/scratch1/3/hkim3239/logs/finalize_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/finalize_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Build list of run dirs to finalize: all run dirs matching the two grid arrays
# that already have a best.pth.tar.
RUNS=()
for d in /storage/scratch1/3/hkim3239/eeg/runs/tusz_light_st_hyper*; do
    if compgen -G "$d/TUSZ/detection/12/*/best.pth.tar" > /dev/null; then
        # Restrict to our two grids
        if [[ "$d" =~ _(842457[0-9]|842458[0-9]|842459[0-9]|842460[0-9])_[0-9]+$ ]]; then
            RUNS+=("$d")
        fi
    fi
done

echo "Total run dirs found: ${#RUNS[@]}"

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ $IDX -ge ${#RUNS[@]} ]]; then
    echo "Index $IDX >= ${#RUNS[@]}, nothing to do."
    exit 0
fi
RUN_DIR="${RUNS[$IDX]}"
echo "ArrayIdx=$IDX  RUN_DIR=$RUN_DIR"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python scripts/finalize_runs.py "$RUN_DIR"
echo "Done: $(date)"
