#!/bin/bash
#SBATCH -J train_chb
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -p gpu-v100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/train_chb_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/train_chb_%j.err

set -euo pipefail

# ---- Configurable via --export=ALL,VAR=val,... at sbatch time --------
MODEL_NAME="${MODEL_NAME:-evobrain}"          # evobrain | node_mamba_gcn
CLIP_LEN="${CLIP_LEN:-12}"                    # 12 | 60
NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
TAG="${TAG:-run1}"
# ----------------------------------------------------------------------

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/data/chb_mit
H5=/storage/scratch1/3/hkim3239/data/chb_mit_resampled
SAVE_BASE=/storage/scratch1/3/hkim3239/runs/chb
SAVE_DIR="$SAVE_BASE/${MODEL_NAME}_${CLIP_LEN}s_${TAG}_${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR" /storage/scratch1/3/hkim3239/logs

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_NAME  clip_len: $CLIP_LEN  epochs: $NUM_EPOCHS  bs(train): $TRAIN_BS"
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
    --num_workers 4 \
    --rand_seed "$SEED" \
    --metric_name auroc \
    --eval_every "${EVAL_EVERY:-5}" \
    --data_augment

echo "End: $(date)"
echo "--- save dir contents ---"
ls "$SAVE_DIR"
