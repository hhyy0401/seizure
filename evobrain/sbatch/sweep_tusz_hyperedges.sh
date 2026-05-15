#!/bin/bash
# n_hyperedges capacity sweep on TUSZ for the 4 surviving ST-hyper variants.
# Tests hypothesis: 8 prototype queries can't span 600+ patient diversity.
# Runs sequentially: 4 models × 2 E_h settings = 8 runs.
set -euo pipefail

NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
CLIP_LEN="${CLIP_LEN:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MODELS="${MODELS:-light_st_hyper light_st_hyper_uni light_st_hyper_mscale light_st_hyper_linear}"
N_HYPEREDGES_LIST="${N_HYPEREDGES_LIST:-32 64}"
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
echo "=== TUSZ n_hyperedges sweep started: $stamp ==="
echo "Models:           $MODELS"
echo "n_hyperedges:     $N_HYPEREDGES_LIST"
echo "Epochs:           $NUM_EPOCHS"
echo

for EH in $N_HYPEREDGES_LIST; do
    for MODEL in $MODELS; do
        SAVE_DIR="$RUNS_BASE/tusz_${MODEL}_E${EH}_${CLIP_LEN}s_${stamp}"
        LOG="$REPO/logs/tusz_${MODEL}_E${EH}_${stamp}.log"
        echo "[$(date +%H:%M:%S)] >>> START $MODEL E_h=$EH  →  $SAVE_DIR"
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
            --n_hyperedges "$EH" \
            ${FIX_THRESHOLD:+--fix_threshold "$FIX_THRESHOLD"} \
            > "$LOG" 2>&1 \
            && echo "[$(date +%H:%M:%S)] <<< OK   $MODEL E_h=$EH  (log: $LOG)" \
            || { echo "[$(date +%H:%M:%S)] !!! FAIL $MODEL E_h=$EH — see $LOG"; }
    done
done

echo "=== sweep done: $(date) ==="
