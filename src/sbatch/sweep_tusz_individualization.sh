#!/bin/bash
# Patient-individualization sweep on TUSZ.
# Three orthogonal mechanisms layered on best baseline (light_st_hyper):
#   1) light_st_hyper_norm    — per-recording instance norm at input
#   2) light_st_hyper_xattn   — cross-attention adaptive prototypes
#   3) light_st_hyper_kmeans  — iterative (soft k-means) adaptive prototypes
# All else identical to baseline light_st_hyper (which got 0.848 on TUSZ).
set -euo pipefail

NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
CLIP_LEN="${CLIP_LEN:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MODELS="${MODELS:-light_st_hyper_norm light_st_hyper_xattn light_st_hyper_kmeans}"
SAMPLING_RATIO="${SAMPLING_RATIO:-1}"
FIX_THRESHOLD="${FIX_THRESHOLD:-0.5}"

REPO=/home/hkim3239/eeg/evobrain
RAW=/home/hkim3239/eeg/tusz/v2.0.6/edf
H5=/home/hkim3239/eeg/tusz_resampled
PREPROC=/home/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1
RUNS_BASE=/home/hkim3239/eeg/runs
PY=/home/hkim3239/eeg/venv/bin/python

mkdir -p "$RUNS_BASE" "$REPO/logs"
cd "$REPO"

stamp=$(date +%Y%m%d_%H%M%S)
echo "=== TUSZ individualization sweep started: $stamp ==="
echo "Models: $MODELS"
echo "Epochs: $NUM_EPOCHS  LR: $LR  Seed: $SEED"
echo

for MODEL in $MODELS; do
    SAVE_DIR="$RUNS_BASE/tusz_${MODEL}_${CLIP_LEN}s_${stamp}"
    LOG="$REPO/logs/tusz_${MODEL}_${stamp}.log"
    echo "[$(date +%H:%M:%S)] >>> START $MODEL  →  $SAVE_DIR"
    "$PY" main.py \
        --dataset TUSZ \
        --task detection \
        --model_name "$MODEL" \
        --num_nodes 19 \
        --raw_data_dir "$RAW" \
        --input_dir "$H5" \
        --preproc_dir "$PREPROC" \
        --save_dir "$SAVE_DIR" \
        --max_seq_len "$CLIP_LEN" \
        --time_step_size 1 \
        --graph_type none \
        --top_k 3 \
        --use_fft \
        --num_epochs "$NUM_EPOCHS" \
        --train_batch_size "$TRAIN_BS" \
        --test_batch_size "$TEST_BS" \
        --lr_init "$LR" \
        --num_workers "$NUM_WORKERS" \
        --rand_seed "$SEED" \
        --metric_name auroc \
        --eval_every "$EVAL_EVERY" \
        --data_augment \
        --sampling_ratio "$SAMPLING_RATIO" \
        ${FIX_THRESHOLD:+--fix_threshold "$FIX_THRESHOLD"} \
        > "$LOG" 2>&1 \
        && echo "[$(date +%H:%M:%S)] <<< OK   $MODEL  (log: $LOG)" \
        || { echo "[$(date +%H:%M:%S)] !!! FAIL $MODEL — see $LOG"; }
done

echo "=== sweep done: $(date) ==="
