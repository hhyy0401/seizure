#!/bin/bash
#SBATCH -J finalize2
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-30
#SBATCH -o /storage/scratch1/3/hkim3239/logs/finalize2_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/finalize2_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Map SLURM .out files (cancelled training runs) → save_dirs, then finalize each.
LOG_PREFIXES=(
    "tusz_grid_8424572"
    "tusz_tcn_8428189"
    "tusz_tcn_8428190"
    "chb_top3_8428218"
    "tusz60_top3_8428283"
)

RUNS=()
for prefix in "${LOG_PREFIXES[@]}"; do
    for f in /storage/scratch1/3/hkim3239/logs/${prefix}_*.out; do
        [ -f "$f" ] || continue
        save=$(grep -oE "Save:.*" "$f" 2>/dev/null | head -1 | awk '{print $2}')
        [ -z "$save" ] && continue
        if compgen -G "$save/*/detection/*/*/best.pth.tar" > /dev/null; then
            RUNS+=("$save")
        fi
    done
done

echo "Total run dirs to finalize: ${#RUNS[@]}"
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ $IDX -ge ${#RUNS[@]} ]]; then
    echo "Index $IDX >= ${#RUNS[@]}, nothing to do."
    exit 0
fi
RUN_DIR="${RUNS[$IDX]}"
echo "ArrayIdx=$IDX  RUN_DIR=$RUN_DIR"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/evobrain
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python scripts/finalize_runs.py "$RUN_DIR"
echo "Done: $(date)"
