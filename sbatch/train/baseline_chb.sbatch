#!/bin/bash
#SBATCH -J base_chb
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/base_chb_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/base_chb_%A_%a.err

# Generic CHB-MIT detection baseline runner for the
# {evolvegcn, graphs4mer} ports. CHB-MIT has 22 channels.
#
# Override via env vars. Examples:
#
#   sbatch --export=ALL,MODEL=evolvegcn,CLIP_LEN=12,SEED=123 baseline_chb.sbatch
#   sbatch --export=ALL,MODEL=graphs4mer,CLIP_LEN=60,SEED=123 baseline_chb.sbatch
#
# Note: CHB-MIT loader currently supports task=detection only (prediction is
# blocked with NotImplementedError in data/dataloader_chb.py).

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
    NUM_EPOCHS="${NUM_EPOCHS:-100}"
    TRAIN_BS="${TRAIN_BS:-128}"
    TEST_BS="${TEST_BS:-256}"
fi

LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-15}"
L2_WD="${L2_WD:-5e-4}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-1}"

if [ "$MODEL" = "evolvegcn" ]; then
    GRAPH_TYPE="dynamic"
elif [ "$MODEL" = "graphs4mer" ]; then
    GRAPH_TYPE="none"
else
    echo "Unsupported MODEL='$MODEL' (use evolvegcn or graphs4mer)" >&2
    exit 1
fi

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/data/chb_mit
RESAMPLED=/storage/scratch1/3/hkim3239/data/chb_mit_resampled

RUN_NAME="${MODEL}_chb${CLIP_LEN}_s${SEED}_${SLURM_JOB_ID:-nojob}"
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
    --dataset CHBMIT --task detection --model_name "$MODEL" \
    --num_nodes 22 \
    --raw_data_dir "$RAW" --input_dir "$RESAMPLED" \
    --save_dir "$SAVE_DIR" \
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
