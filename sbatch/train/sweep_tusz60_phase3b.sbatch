#!/bin/bash
#SBATCH -J sw60_p3b
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-4
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sw60_p3b_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sw60_p3b_%A_%a.err

# Round 3-b — 추가 tattn 튜닝 (LR + heads).
# Base = dp=0.2 + tattn (Round 2 winner). cap=40, patience=10, eval_every=3.

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Defaults (= dp=0.2 + tattn baseline)
SEED=123
DROPOUT=0.2
N_HYPER_LAYERS=2
N_PMA=1
LR=1e-3
L2_WD=5e-4
TATTN_HEADS=4
TATTN_N_LAYERS=1
TAG="tattn_b_ref"

case $IDX in
    0)  LR=5e-4;                                       TAG="tattn_lr5e-4" ;;
    1)  LR=2e-3;                                       TAG="tattn_lr2e-3" ;;
    2)  TATTN_HEADS=8;                                 TAG="tattn_h8" ;;
    3)  L2_WD=1e-3;                                    TAG="tattn_wd1e-3" ;;
    4)  TATTN_HEADS=8; TATTN_N_LAYERS=2;               TAG="tattn_h8_2L" ;;
    *)  echo "Unknown IDX=$IDX" >&2; exit 1 ;;
esac

E_H=3
CLIP_LEN=60
NUM_EPOCHS=40
PATIENCE=10
EVAL_EVERY=3
TRAIN_BS=32
TEST_BS=64
D_MODEL=128

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="sw60_p3b_${TAG}_s${SEED}_${SLURM_JOB_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "IDX=$IDX TAG=$TAG SEED=$SEED"
echo "lr=$LR wd=$L2_WD dropout=$DROPOUT n_heads=$TATTN_HEADS n_layers=$TATTN_N_LAYERS"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py --dataset TUSZ --task detection --model_name light_st_hyper \
    --num_nodes 19 \
    --raw_data_dir "$RAW/edf" --input_dir "$RESAMPLED" \
    --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units "$D_MODEL" \
    --n_hyper_layers "$N_HYPER_LAYERS" --n_hyperedges "$E_H" \
    --n_pma_seeds "$N_PMA" \
    --aux_type none \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    --temporal_attn \
    --temporal_attn_heads "$TATTN_HEADS" \
    --tattn_n_layers "$TATTN_N_LAYERS"

echo "End: $(date)"
echo "--- TEST line ---"
find "$SAVE_DIR" -name info.log -exec grep "TEST set prediction" {} \; | tail -1
