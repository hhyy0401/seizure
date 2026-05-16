#!/bin/bash
#SBATCH -J viz_node
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0:15:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/viz_node_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/viz_node_%j.err

# Render t-SNE + cosine heatmap of learnable channel embeddings from the
# TUSZ 12s checkpoint into figures/node_emb/.
set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO/src"

python scripts/viz_node_emb.py \
    "$REPO/ckpts/tusz12/best.pth.tar" \
    "$REPO/figures/node_emb"

echo "End: $(date)"
ls -la "$REPO/figures/node_emb/"
