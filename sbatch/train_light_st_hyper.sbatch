#!/bin/bash
#SBATCH -J train_light
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/train_light_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/train_light_%j.err

set -euo pipefail

# ---- LightEvoBrain family training on CHB-MIT ---------------------
# MODEL_NAME (override via env): default light_st_hyper
#   Valid model names (see main.py for full list):
#     light_st_hyper, light_dot, light_bilinear, light_attention,
#     light_dyn_hyper, light_static_hyper, light_mamba_band_plv,
#     light_attn_band_gated, light_st_hyper_band, light_st_hyper_norm,
#     light_st_hyper_mscale, ada_mshyper, st_hyper, mshyper, ...
# Usage: sbatch --export=ALL,MODEL_NAME=light_st_hyper,TAG=run1 ...
MODEL_NAME="${MODEL_NAME:-${EDGE_TYPE:-light_st_hyper}}"   # back-compat with EDGE_TYPE
CLIP_LEN="${CLIP_LEN:-12}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
TAG="${TAG:-$MODEL_NAME}"
EVAL_EVERY="${EVAL_EVERY:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# -------------------------------------------------------------------

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src     # ← project (code on project, data on scratch)
RAW=/storage/scratch1/3/hkim3239/data/chb_mit
H5=/storage/scratch1/3/hkim3239/data/chb_mit_resampled
SAVE_BASE=/storage/scratch1/3/hkim3239/eeg/runs/chb
SAVE_DIR="$SAVE_BASE/${MODEL_NAME}_${CLIP_LEN}s_${TAG}_${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR" /storage/scratch1/3/hkim3239/logs

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_NAME  clip_len: $CLIP_LEN  epochs: $NUM_EPOCHS  bs(train): $TRAIN_BS"
echo "Repo:  $REPO"
echo "Save:  $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

cd "$REPO"

python main.py \
    --dataset CHBMIT \
    --task detection \
    --model_name "$MODEL_NAME" \
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
    --data_augment

echo "End: $(date)"
echo "--- save dir ---"
ls "$SAVE_DIR"
