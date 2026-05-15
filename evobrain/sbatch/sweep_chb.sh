#!/bin/bash
# Sweep 5 models on CHB-MIT sequentially.
#
# Single g4-standard-48 GPU, no SLURM. Each run writes to its own save_dir.
# All runs share the same hyperparameters; only --model_name changes.
#
# Usage:
#   bash sbatch/sweep_chb.sh                 # default settings
#   NUM_EPOCHS=10 bash sbatch/sweep_chb.sh   # override
#   MODELS="evobrain light_attention" bash sbatch/sweep_chb.sh   # subset
set -euo pipefail

# ---- config (override via env) ----
NUM_EPOCHS="${NUM_EPOCHS:-50}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
CLIP_LEN="${CLIP_LEN:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-5}"
MODELS="${MODELS:-evobrain light_attention light_dyn_hyper light_static_hyper light_mamba_band_plv}"
SAMPLING_RATIO="${SAMPLING_RATIO:-50}"
POS_WEIGHT="${POS_WEIGHT:-50}"
FIX_THRESHOLD="${FIX_THRESHOLD:-0.5}"

# ---- paths ----
REPO=/home/hkim3239/eeg/evobrain
RAW=/home/hkim3239/eeg/chbmit
H5=/home/hkim3239/eeg/chbmit_resampled
RUNS_BASE=/home/hkim3239/eeg/runs
PY=/home/hkim3239/eeg/venv/bin/python

mkdir -p "$RUNS_BASE"
mkdir -p "$REPO/logs"

cd "$REPO"

stamp=$(date +%Y%m%d_%H%M%S)
echo "=== Sweep started: $stamp ==="
echo "Models:     $MODELS"
echo "Epochs:     $NUM_EPOCHS  Batch (train/test): $TRAIN_BS / $TEST_BS  LR: $LR"
echo "Clip len:   ${CLIP_LEN}s  Seed: $SEED"
echo "GPU:"; nvidia-smi -L || true
echo

for MODEL in $MODELS; do
    SAVE_DIR="$RUNS_BASE/${MODEL}_${CLIP_LEN}s_${stamp}"
    LOG="$REPO/logs/${MODEL}_${stamp}.log"
    echo "[$(date +%H:%M:%S)] >>> START $MODEL  →  $SAVE_DIR"
    "$PY" main.py \
        --dataset CHBMIT \
        --task detection \
        --model_name "$MODEL" \
        --num_nodes 22 \
        --raw_data_dir "$RAW" \
        --input_dir "$H5" \
        --save_dir "$SAVE_DIR" \
        --max_seq_len "$CLIP_LEN" \
        --time_step_size 1 \
        --graph_type dynamic \
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
        --pos_weight "$POS_WEIGHT" \
        ${FIX_THRESHOLD:+--fix_threshold "$FIX_THRESHOLD"} \
        > "$LOG" 2>&1 \
        && echo "[$(date +%H:%M:%S)] <<< OK   $MODEL  (log: $LOG)" \
        || { echo "[$(date +%H:%M:%S)] !!! FAIL $MODEL  — see $LOG"; }
done

echo "=== Sweep done: $(date) ==="
