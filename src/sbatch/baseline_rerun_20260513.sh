#!/bin/bash
# Re-run 3 hypergraph baselines × 2 datasets with FIXED implementation.
# Fixes applied:
#   1. Ada-MSHyper: dual loss (constraint loss in backward) — was missing
#   2. d_inner bumped: ada=32, ms=16, st_hyper=32 (was 8 — 12x too narrow)
#   3. topk=3 (was 10 — paper default)
#   4. window_size aligned to short seq_len=12 (was degenerate at last scale)
#
# MSHyper: NO inter-scale change — official code also discards inter-scale
# edges in its `get_mask` return; our intra-only port already matches official
# behavior.
#
# ST-Hyper: this is NOT real ST-Hyper (paper has STPM/memory/GCRU/edge-edge
# GAT — code not public). We run it as "Ada-MSHyper with joint (N×T) reshape"
# ablation. Paper writeup will be honest about this scope.

set -uo pipefail

REPO=/home/hkim3239/eeg/evobrain
PY=/home/hkim3239/eeg/venv/bin/python
RUNS=/home/hkim3239/eeg/runs

CHB_RAW=/home/hkim3239/eeg/chbmit
CHB_H5=/home/hkim3239/eeg/chbmit_resampled
TUSZ_RAW=/home/hkim3239/eeg/tusz/v2.0.6/edf
TUSZ_H5=/home/hkim3239/eeg/tusz_resampled
TUSZ_PRE=/home/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1

mkdir -p "$REPO/logs"
cd "$REPO"

stamp=$(date +%Y%m%d_%H%M%S)
echo "=== Fixed-baseline re-launch: $stamp ==="
echo "GPU usage:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

pids=()

# ----- CHB-MIT × 3 -----
for MODEL in ada_mshyper mshyper st_hyper; do
    SAVE="$RUNS/chbmit_${MODEL}_FIXED_${stamp}"
    LOG="$REPO/logs/chbmit_${MODEL}_FIXED_${stamp}.log"
    "$PY" main.py \
        --dataset CHBMIT --task detection --model_name "$MODEL" \
        --num_nodes 22 \
        --raw_data_dir "$CHB_RAW" \
        --input_dir "$CHB_H5" \
        --save_dir "$SAVE" \
        --max_seq_len 12 --time_step_size 1 \
        --graph_type none --top_k 3 --use_fft \
        --num_epochs 100 \
        --train_batch_size 128 --test_batch_size 256 \
        --lr_init 1e-4 --num_workers 2 \
        --rand_seed 123 --metric_name auroc \
        --eval_every 10 --data_augment \
        --sampling_ratio 1 --fix_threshold 0.5 \
        > "$LOG" 2>&1 &
    pids+=($!)
    echo "[chbmit_${MODEL}] PID=${pids[-1]}  LOG=$LOG"
done

# ----- TUSZ × 3 -----
for MODEL in ada_mshyper mshyper st_hyper; do
    SAVE="$RUNS/tusz_${MODEL}_FIXED_${stamp}"
    LOG="$REPO/logs/tusz_${MODEL}_FIXED_${stamp}.log"
    "$PY" main.py \
        --dataset TUSZ --task detection --model_name "$MODEL" \
        --num_nodes 19 \
        --preproc_dir "$TUSZ_PRE" \
        --raw_data_dir "$TUSZ_RAW" \
        --input_dir "$TUSZ_H5" \
        --save_dir "$SAVE" \
        --max_seq_len 12 --time_step_size 1 \
        --graph_type none --top_k 3 --use_fft \
        --num_epochs 100 \
        --train_batch_size 128 --test_batch_size 256 \
        --lr_init 1e-4 --num_workers 2 \
        --rand_seed 123 --metric_name auroc \
        --eval_every 10 --data_augment \
        --sampling_ratio 1 --fix_threshold 0.5 \
        > "$LOG" 2>&1 &
    pids+=($!)
    echo "[tusz_${MODEL}] PID=${pids[-1]}  LOG=$LOG"
done

echo ""
echo "All 6 jobs launched. PIDs: ${pids[*]}"
echo "Stamp: $stamp"
echo ""
echo "Monitor with: tail -f $REPO/logs/*_FIXED_${stamp}.log"

# Quick sanity check at 20s
sleep 20
echo ""
echo "=== Status after 20s ==="
for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "  PID $pid: ALIVE"
    else
        echo "  PID $pid: DEAD (check log!)"
    fi
done
echo ""
echo "=== GPU after 20s ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
