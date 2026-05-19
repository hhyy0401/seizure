#!/bin/bash
#SBATCH -J regen_evobrain
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 0:15:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/regen_evobrain_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/regen_evobrain_%j.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease
DUMP="$REPO/figures/dumps/tusz60_E3_s123_dense.npz"
OUT="$REPO/figures/onset_edge"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

cd "$REPO"

# 1) EvoBrain Fig 6 topology reproduction on our head template.
python src/scripts/viz_onset_edge.py "$DUMP" "$OUT" --evobrain_repro

# 2) Our model's 3-scenario figure (Normal / Pre-seizure / Post-seizure).
python src/scripts/viz_onset_edge.py "$DUMP" "$OUT" \
    --evobrain_scenarios "1323:30:non-seizure;316:42:Pre-seizure;1283:52:Seizure"

# Rename scenarios output to keep the v5_ prefix (so it overwrites cleanly).
SRC="$OUT/s123_evobrain_scenarios_c1323t30_c316t42_c1283t52.pdf"
DST="$OUT/s123_v5_evobrain_scenarios_c1323t30_c316t42_c1283t52.pdf"
if [ -f "$SRC" ]; then mv -f "$SRC" "$DST"; fi

# 3) 6 per-clip pairs (heatmap + topo3) — uses the SAME clip IDs as the
#    existing figures in figures/onset_edge/.
python src/scripts/viz_onset_edge.py "$DUMP" "$OUT" \
    --clip_ids "1205,4270,31,4486,4303,973"

ls -l "$OUT"/s123_evobrain_fig6_reproduction.pdf "$DST" \
      "$OUT"/s123_edge0_*_topo3.pdf "$OUT"/s123_edge0_*_heatmap.pdf 2>&1 || true
