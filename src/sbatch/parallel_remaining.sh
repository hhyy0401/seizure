#!/bin/bash
# Launch the remaining 4 models in parallel on the same GPU.
#
# Memory check: total ~10GB out of 96GB available, no contention risk.
# CPU workers: 4 per model × 4 = 16 workers on 48 cores → fine.
# Each model gets its own log + save_dir.
#
# Pre-req: E+ test eval finished (sweep_chb.sh already killed).
set -uo pipefail

NUM_EPOCHS="${NUM_EPOCHS:-50}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
CLIP_LEN="${CLIP_LEN:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-25}"
SAMPLING_RATIO="${SAMPLING_RATIO:-50}"   # natural prevalence ~290:1; 50 is the stable middle
POS_WEIGHT="${POS_WEIGHT:-50}"           # match SAMPLING_RATIO for proper calibration
FIX_THRESHOLD="${FIX_THRESHOLD:-0.5}"    # fair cross-model comparison; "" to disable

# 5 models to launch concurrently (all graph_type=none — fast dataloader)
MODELS=(
    "light_dyn_hyper"
    "light_static_hyper"
    "light_attention"
    "light_attn_band_gated"    # the no-PLV ablation
    "light_mamba_band_plv"     # the PLV variant
)

REPO=/home/hkim3239/eeg/evobrain
RAW=/home/hkim3239/eeg/chbmit
H5=/home/hkim3239/eeg/chbmit_resampled
RUNS_BASE=/home/hkim3239/eeg/runs
PY=/home/hkim3239/eeg/venv/bin/python

mkdir -p "$RUNS_BASE" "$REPO/logs"
cd "$REPO"

stamp=$(date +%Y%m%d_%H%M%S)
echo "=== Parallel launch: $stamp ==="
echo "Models:  ${MODELS[*]}"
echo "Epochs:  $NUM_EPOCHS  Batch (train/test): $TRAIN_BS / $TEST_BS  LR: $LR  EVAL_EVERY: $EVAL_EVERY"
echo "GPU:";  nvidia-smi -L || true
echo

pids=()
for MODEL in "${MODELS[@]}"; do
    SAVE_DIR="$RUNS_BASE/${MODEL}_${CLIP_LEN}s_${stamp}"
    LOG="$REPO/logs/${MODEL}_${stamp}.log"
    # Models that never touch the dataloader-built `adj` can skip the
    # per-timestep correlation (~12× faster eval). Only `evobrain` and
    # `dcrnn`/`evolvegcn`/`gru_gcn` actually use it.
    case "$MODEL" in
        evobrain|dcrnn|evolvegcn|gru_gcn) GTYPE="dynamic" ;;
        *) GTYPE="none" ;;
    esac
    echo "[$(date +%H:%M:%S)] launching $MODEL  (graph_type=$GTYPE)  →  log: $LOG"
    nohup "$PY" main.py \
        --dataset CHBMIT --task detection --model_name "$MODEL" --num_nodes 22 \
        --raw_data_dir "$RAW" --input_dir "$H5" --save_dir "$SAVE_DIR" \
        --max_seq_len "$CLIP_LEN" --time_step_size 1 \
        --graph_type "$GTYPE" --top_k 3 --use_fft \
        --num_epochs "$NUM_EPOCHS" \
        --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
        --lr_init "$LR" --num_workers "$NUM_WORKERS" \
        --rand_seed "$SEED" --metric_name auroc \
        --eval_every "$EVAL_EVERY" --data_augment \
        --sampling_ratio "$SAMPLING_RATIO" --pos_weight "$POS_WEIGHT" \
        ${FIX_THRESHOLD:+--fix_threshold "$FIX_THRESHOLD"} \
        > "$LOG" 2>&1 &
    pids+=($!)
    sleep 2   # stagger launches so model-build messages don't clobber
done

echo
echo "Launched PIDs: ${pids[*]}"
echo "Waiting for all 4 to finish (this will block until done)..."
echo

# Wait for all to complete; record exit status
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    model="${MODELS[$i]}"
    wait "$pid"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] <<< OK   $model  (pid $pid)"
    else
        echo "[$(date +%H:%M:%S)] !!! FAIL $model  (pid $pid, exit $rc)"
    fi
done

echo "=== All 4 done: $(date) ==="
