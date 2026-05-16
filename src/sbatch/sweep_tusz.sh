#!/bin/bash
# Sweep 5 surviving ST models + evobrain baseline on TUSZ v2.0.6, sequentially.
#
# Uses preproc_dir cache (clipLen12_timeStepSize1) — no on-the-fly STFT.
# evobrain alone needs graph_type=dynamic (its edge Mamba consumes adj);
# all light_st_hyper_* variants run with graph_type=none.
#
# Usage:
#   bash sbatch/sweep_tusz.sh
#   NUM_EPOCHS=10 bash sbatch/sweep_tusz.sh
#   MODELS="evobrain" bash sbatch/sweep_tusz.sh
set -euo pipefail

# ---- config (override via env) ----
NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
CLIP_LEN="${CLIP_LEN:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MODELS="${MODELS:-evobrain light_st_hyper_linear light_st_hyper_uni light_st_hyper_mscale light_st_hyper}"
SAMPLING_RATIO="${SAMPLING_RATIO:-1}"
FIX_THRESHOLD="${FIX_THRESHOLD:-0.5}"

# ---- paths ----
REPO=/home/hkim3239/eeg/evobrain
RAW=/home/hkim3239/eeg/tusz/v2.0.6/edf
H5=/home/hkim3239/eeg/tusz_resampled
PREPROC=/home/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1
RUNS_BASE=/home/hkim3239/eeg/runs
PY=/home/hkim3239/eeg/venv/bin/python

mkdir -p "$RUNS_BASE"
mkdir -p "$REPO/logs"

cd "$REPO"

stamp=$(date +%Y%m%d_%H%M%S)
echo "=== TUSZ Sweep started: $stamp ==="
echo "Models:     $MODELS"
echo "Epochs:     $NUM_EPOCHS  Batch (train/test): $TRAIN_BS / $TEST_BS  LR: $LR"
echo "Clip len:   ${CLIP_LEN}s  Seed: $SEED"
echo "Sampling:   1:${SAMPLING_RATIO} (sz : nosz)  fix_threshold=$FIX_THRESHOLD"
echo "Preproc:    $PREPROC"
echo "GPU:"; nvidia-smi -L || true
echo

for MODEL in $MODELS; do
    if [ "$MODEL" = "evobrain" ]; then
        GRAPH_TYPE=dynamic
    else
        GRAPH_TYPE=none
    fi

    SAVE_DIR="$RUNS_BASE/tusz_${MODEL}_${CLIP_LEN}s_${stamp}"
    LOG="$REPO/logs/tusz_${MODEL}_${stamp}.log"
    echo "[$(date +%H:%M:%S)] >>> START $MODEL (graph_type=$GRAPH_TYPE)  →  $SAVE_DIR"
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
        --graph_type "$GRAPH_TYPE" \
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
        || { echo "[$(date +%H:%M:%S)] !!! FAIL $MODEL  — see $LOG"; }
done

echo "=== TUSZ Sweep done: $(date) ==="
