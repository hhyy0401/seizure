#!/bin/bash
#SBATCH -J base_tusz
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/base_tusz_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/base_tusz_%A_%a.err

# Generic TUSZ-detection baseline runner for the {evolvegcn, graphs4mer} ports.
#
# Override via env vars. Examples:
#
#   # EvolveGCN, 12s
#   sbatch --export=ALL,MODEL=evolvegcn,CLIP_LEN=12,SEED=123 baseline_tusz.sbatch
#
#   # GraphS4mer, 60s
#   sbatch --export=ALL,MODEL=graphs4mer,CLIP_LEN=60,SEED=123 baseline_tusz.sbatch
#
#   # 3-seed array (use SLURM_ARRAY_TASK_ID 0..2 to pick seed)
#   sbatch --array=0-2 --export=ALL,MODEL=evolvegcn,CLIP_LEN=12 baseline_tusz.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

MODEL="${MODEL:?must set MODEL=evolvegcn or MODEL=graphs4mer}"
CLIP_LEN="${CLIP_LEN:-12}"

# Seed selection: explicit SEED wins, otherwise picked from array index.
SEEDS=(123 456 789)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
else
    SEED="${SEED:-123}"
fi

# Clip-length-dependent hyperparams (mirror LightSTHyper recipe).
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

# Per-model graph_type: EvolveGCN consumes the dataloader's dynamic adjacency;
# GraphS4mer learns its own — disable dataloader graph construction for speed.
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
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PRE=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen${CLIP_LEN}_timeStepSize1

# Use FFT preproc cache if pre-built; otherwise fall back to on-the-fly FFT.
if [ -d "$PRE" ]; then
    PRE_ARG=(--preproc_dir "$PRE")
else
    PRE_ARG=()
    echo "preproc cache $PRE not found — running with on-the-fly FFT."
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
    --graph_type "$GRAPH_TYPE" --top_k 3 --use_fft \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment

echo "End: $(date)"
ls "$SAVE_DIR"
