#!/bin/bash
#SBATCH -J base_fnd
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/base_fnd_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/base_fnd_%A_%a.err

# EEG foundation-model baseline runner for {labram, eegpt} on
# {TUSZ 12s/60s, CHB-MIT 12s} detection. Pretrained encoder + fresh head,
# end-to-end fine-tune with low LR (matches the LaBraM/EEGPT defaults).
#
# Examples:
#   sbatch --array=0-2 --export=ALL,MODEL=labram,DATASET=TUSZ,CLIP_LEN=12 baseline_foundation.sbatch
#   sbatch --array=0-2 --export=ALL,MODEL=eegpt,DATASET=TUSZ,CLIP_LEN=60   baseline_foundation.sbatch
#   sbatch --array=0-2 --export=ALL,MODEL=labram,DATASET=CHBMIT,CLIP_LEN=12 baseline_foundation.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

MODEL="${MODEL:?must set MODEL=labram|eegpt}"
DATASET="${DATASET:?must set DATASET=TUSZ|CHBMIT}"
CLIP_LEN="${CLIP_LEN:-12}"

SEEDS=(123 456 789)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
else
    SEED="${SEED:-123}"
fi

# Foundation transformers are heavier than baselines — shrink the batch.
if [ "$CLIP_LEN" = "60" ]; then
    NUM_EPOCHS="${NUM_EPOCHS:-50}"
    TRAIN_BS="${TRAIN_BS:-8}"
    TEST_BS="${TEST_BS:-16}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-50}"
    TRAIN_BS="${TRAIN_BS:-16}"
    TEST_BS="${TEST_BS:-32}"
fi

# LaBraM paper default = 5e-4 with weight_decay=0.05. Foundation models
# diverge fast at our usual 1e-3, so go lower.
LR="${LR:-5e-4}"
PATIENCE="${PATIENCE:-10}"
L2_WD="${L2_WD:-0.05}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src

if [ "$DATASET" = "TUSZ" ]; then
    NUM_NODES=19
    RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
    RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
    RAW_ARG=(--raw_data_dir "$RAW/edf" --input_dir "$RESAMPLED")
    PRE=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen${CLIP_LEN}_timeStepSize1
    if [ -d "$PRE" ]; then
        PRE_ARG=(--preproc_dir "$PRE")
    else
        PRE_ARG=()
        echo "preproc cache $PRE not found - on-the-fly resample/FFT."
    fi
elif [ "$DATASET" = "CHBMIT" ]; then
    NUM_NODES=22
    RAW=/storage/scratch1/3/hkim3239/data/chb_mit
    RESAMPLED=/storage/scratch1/3/hkim3239/data/chb_mit_resampled
    RAW_ARG=(--raw_data_dir "$RAW" --input_dir "$RESAMPLED")
    PRE_ARG=()
else
    echo "Unsupported DATASET=$DATASET" >&2; exit 1
fi

RUN_NAME="${MODEL}_${DATASET,,}${CLIP_LEN}_s${SEED}_${SLURM_JOB_ID:-nojob}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "MODEL=$MODEL  DATASET=$DATASET  CLIP_LEN=$CLIP_LEN  SEED=$SEED"
echo "TRAIN_BS=$TRAIN_BS  LR=$LR  L2_WD=$L2_WD"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py \
    --dataset "$DATASET" --task detection --model_name "$MODEL" \
    --num_nodes "$NUM_NODES" \
    "${RAW_ARG[@]}" "${PRE_ARG[@]}" --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type none --top_k 3 \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers "$NUM_WORKERS" \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    --skip_midtest

echo "End: $(date)"
ls "$SAVE_DIR"
