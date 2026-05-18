#!/bin/bash
#SBATCH -J base_tuszP
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/base_tuszP_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/base_tuszP_%A_%a.err

# TUSZ seizure-prediction baseline runner for {evolvegcn, graphs4mer}.
#
# IMPORTANT: this script requires src/data/file_markers_prediction/ to exist
# (the prediction split with preictal/interictal clip indices). It is NOT
# shipped in this repo; generate it from the TUSZ raw data before running.
# The loader expects the same naming convention as file_markers_detection,
# e.g. trainSet_seq2seq_<CLIP_LEN>s_{sz,nosz}.txt etc.
#
# Examples:
#   sbatch --export=ALL,MODEL=evolvegcn,CLIP_LEN=12,SEED=123 baseline_tusz_prediction.sbatch
#   sbatch --export=ALL,MODEL=graphs4mer,CLIP_LEN=60,SEED=123 baseline_tusz_prediction.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

MODEL="${MODEL:?must set MODEL=evolvegcn or MODEL=graphs4mer}"
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
    TEST_BS="${TEST_BS:-64}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-80}"
    TRAIN_BS="${TRAIN_BS:-128}"
    TEST_BS="${TEST_BS:-256}"
fi

LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"
L2_WD="${L2_WD:-5e-4}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-1}"

if [ "$MODEL" = "evolvegcn" ]; then
    GRAPH_TYPE="dynamic"
elif [ "$MODEL" = "graphs4mer" ]; then
    GRAPH_TYPE="none"
else
    echo "Unsupported MODEL='$MODEL'" >&2
    exit 1
fi

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PRE=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen${CLIP_LEN}_timeStepSize1

# Bail early if the prediction markers aren't present.
if [ ! -d "$REPO/data/file_markers_prediction" ]; then
    echo "ERROR: $REPO/data/file_markers_prediction/ missing." >&2
    echo "Generate the prediction split first." >&2
    exit 2
fi

RUN_NAME="${MODEL}_tuszP${CLIP_LEN}_s${SEED}_${SLURM_JOB_ID:-nojob}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "MODEL=$MODEL  CLIP_LEN=$CLIP_LEN  SEED=$SEED  GRAPH_TYPE=$GRAPH_TYPE  TASK=prediction"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py \
    --dataset TUSZ --task prediction --model_name "$MODEL" \
    --num_nodes 19 \
    --raw_data_dir "$RAW/edf" --input_dir "$RESAMPLED" \
    --preproc_dir "$PRE" --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type "$GRAPH_TYPE" --top_k 3 --use_fft \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment

echo "End: $(date)"
ls "$SAVE_DIR"
