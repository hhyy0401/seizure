#!/bin/bash
#SBATCH -J build_table
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/build_table_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/build_table_%j.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

OUT_FILE="$REPO/RESULTS_TABLE.md"
{
    echo "# Results Table — generated $(date)"
    echo
    echo '```'
    python src/scripts/build_results_table.py --top 10
    echo '```'
} > "$OUT_FILE"

echo "Wrote $OUT_FILE"
ls -la "$OUT_FILE"
