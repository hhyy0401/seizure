#!/bin/bash
#SBATCH -J make_paper_plots
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 0:20:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/make_paper_plots_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/make_paper_plots_%j.err

# Render sweep + ablation PDFs after the TUSZ ablation jobs finish.
# Submit with --dependency=afterany:<jobid1>:<jobid2>:<jobid3>.
set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

python "$REPO/scripts/make_paper_plots.py"
echo "---"
ls -l "$REPO/figures/"*.pdf
