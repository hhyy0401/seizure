#!/usr/bin/env bash
# Train LightSTHyper from scratch with the best config for either clip length.
#
# Usage:  scripts/train_best.sh <clip_tag> <seed>
#   clip_tag = tusz12 | tusz60
#   seed     = any integer (we used 123, 456, 789)
#
# Edit RAW / RESAMPLED / SAVE_BASE below to match your data layout.
set -euo pipefail

# ------- EDIT THESE TO YOUR LAYOUT -------
RAW=/mnt/data0/sheoyon/EvoBrain/data/tusz_v2.0.6/v2.0.6/edf
RESAMPLED=/mnt/data0/sheoyon/EvoBrain/data/tusz_resampled
SAVE_BASE=$HOME/light_sthyper_runs
# -----------------------------------------

CLIP_TAG=${1:-tusz12}
SEED=${2:-123}
CUDA=${CUDA_VISIBLE_DEVICES:-0}

case $CLIP_TAG in
    tusz12)
        CLIP=12; E_H=1
        NUM_EPOCHS=80; TRAIN_BS=128; TEST_BS=256; PATIENCE=10
        TAG="tusz12_E1_noaux_s${SEED}"
        ;;
    tusz60)
        CLIP=60; E_H=3
        NUM_EPOCHS=100; TRAIN_BS=32; TEST_BS=64; PATIENCE=10
        TAG="tusz60_E3_noaux_s${SEED}"
        ;;
    *)
        echo "unknown clip tag: $CLIP_TAG (use tusz12 or tusz60)"; exit 1 ;;
esac

PKG=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR=$SAVE_BASE/$TAG
mkdir -p "$SAVE_DIR"
echo "[train] $TAG -> $SAVE_DIR"

cd "$PKG/src"
CUDA_VISIBLE_DEVICES=$CUDA python main.py \
    --dataset TUSZ --task detection --model_name light_st_hyper \
    --num_nodes 19 --max_seq_len $CLIP --time_step_size 1 \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units 128 --n_hyper_layers 2 --n_hyperedges $E_H \
    --aux_type none --dropout 0.0 --l2_wd 5e-4 \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BS --test_batch_size $TEST_BS \
    --lr_init 1e-3 --num_workers 8 \
    --rand_seed $SEED --metric_name auroc \
    --eval_every 1 --patience $PATIENCE --data_augment \
    --raw_data_dir "$RAW" \
    --input_dir   "$RESAMPLED" \
    --save_dir    "$SAVE_DIR"

echo "[train] done. results in $SAVE_DIR/TUSZ/detection/$CLIP/light_st_hyper_none_${SEED}_01/"
