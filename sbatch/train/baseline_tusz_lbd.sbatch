#!/bin/bash
#SBATCH -J base_tusz_lbd
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/base_tusz_lbd_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/base_tusz_lbd_%A_%a.err

# TUSZ detection runner for the "light backbone" baselines:
#   {lstm, cnnlstm, BIOT, dcrnn, gru_gcn}.
# Hyperparameters mirror baseline_tusz.sbatch (12s/60s split).
#
# Examples:
#   sbatch --array=0-2 --export=ALL,MODEL=lstm,CLIP_LEN=12  baseline_tusz_lbd.sbatch
#   sbatch --array=0-2 --export=ALL,MODEL=BIOT,CLIP_LEN=60  baseline_tusz_lbd.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

MODEL="${MODEL:?must set MODEL=lstm|cnnlstm|BIOT|dcrnn|gru_gcn}"
CLIP_LEN="${CLIP_LEN:-12}"

SEEDS=(123 456 789)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
else
    SEED="${SEED:-123}"
fi

if [ "$CLIP_LEN" = "60" ]; then
    NUM_EPOCHS="${NUM_EPOCHS:-100}"
    TRAIN_BS="${TRAIN_BS:-32}"
    TEST_BS="${TEST_BS:-128}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-80}"
    TRAIN_BS="${TRAIN_BS:-128}"
    TEST_BS="${TEST_BS:-512}"
fi

LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"
L2_WD="${L2_WD:-5e-4}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Per-model graph_type + FFT:
#   lstm / cnnlstm        -> none, use_fft
#   BIOT                  -> none, no_fft  (main.py forces use_fft=False)
#   dcrnn / gru_gcn       -> individual (correlation), use_fft
case "$MODEL" in
    lstm|cnnlstm)    GRAPH_TYPE="none";       FFT_FLAG=(--use_fft) ;;
    BIOT)            GRAPH_TYPE="none";       FFT_FLAG=() ;;
    dcrnn|gru_gcn)   GRAPH_TYPE="individual"; FFT_FLAG=(--use_fft) ;;
    *) echo "Unsupported MODEL='$MODEL'" >&2; exit 1 ;;
esac

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PRE=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen${CLIP_LEN}_timeStepSize1

if [ -d "$PRE" ]; then
    PRE_ARG=(--preproc_dir "$PRE")
else
    PRE_ARG=()
    echo "preproc cache $PRE not found - on-the-fly FFT."
fi

RUN_NAME="${MODEL}_tusz${CLIP_LEN}_s${SEED}_${SLURM_JOB_ID:-nojob}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "MODEL=$MODEL  CLIP_LEN=$CLIP_LEN  SEED=$SEED  GRAPH_TYPE=$GRAPH_TYPE"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py \
    --dataset TUSZ --task detection --model_name "$MODEL" \
    --num_nodes 19 \
    --raw_data_dir "$RAW/edf" --input_dir "$RESAMPLED" \
    "${PRE_ARG[@]}" --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type "$GRAPH_TYPE" --top_k 3 \
    "${FFT_FLAG[@]}" \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers "$NUM_WORKERS" \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    --skip_midtest

echo "End: $(date)"
ls "$SAVE_DIR"
