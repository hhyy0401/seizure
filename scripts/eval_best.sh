#!/usr/bin/env bash
# Reload a saved best.pth.tar and re-evaluate on TUSZ test set.
#
# Usage:  scripts/eval_best.sh <clip_tag> <seed>
#   clip_tag = tusz12 | tusz60
#   seed     = 123 | 456 | 789
#
# Edit RAW / RESAMPLED below to match your data layout.
set -euo pipefail

# ------- EDIT THESE TO YOUR LAYOUT -------
RAW=/mnt/data0/sheoyon/EvoBrain/data/tusz_v2.0.6/v2.0.6/edf
RESAMPLED=/mnt/data0/sheoyon/EvoBrain/data/tusz_resampled
# -----------------------------------------

CLIP_TAG=${1:-tusz12}    # tusz12 / tusz60
SEED=${2:-123}
CUDA=${CUDA_VISIBLE_DEVICES:-0}

case $CLIP_TAG in
    tusz12)
        CLIP=12; E_H=1; TEST_BS=256
        TAG="tusz12_E1_noaux_s${SEED}"
        ;;
    tusz60)
        CLIP=60; E_H=3; TEST_BS=64
        TAG="tusz60_E3_noaux_s${SEED}"
        ;;
    *)
        echo "unknown clip tag: $CLIP_TAG (use tusz12 or tusz60)"; exit 1 ;;
esac

PKG=$(cd "$(dirname "$0")/.." && pwd)
CKPT=$PKG/ckpts/$TAG/best.pth.tar
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

OUT=$(mktemp -d -t evo_eval_XXXX)
echo "[eval] $TAG -> $OUT"

cd "$PKG/src"
CUDA_VISIBLE_DEVICES=$CUDA python main.py \
    --dataset TUSZ --task detection --model_name light_st_hyper \
    --num_nodes 19 --max_seq_len $CLIP --time_step_size 1 \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units 128 --n_hyper_layers 2 --n_hyperedges $E_H \
    --aux_type none --dropout 0.0 --l2_wd 5e-4 \
    --test_batch_size $TEST_BS --num_workers 4 \
    --rand_seed $SEED \
    --raw_data_dir "$RAW" \
    --input_dir   "$RESAMPLED" \
    --load_model_path "$CKPT" \
    --test \
    --save_dir "$OUT"

echo "[eval] done. logs/results in $OUT"
