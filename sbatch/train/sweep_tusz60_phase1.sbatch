#!/bin/bash
#SBATCH -J sw60_p1
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-10
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sw60_p1_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sw60_p1_%A_%a.err

# Phase 1 sensitivity sweep — LightSTHyper TUSZ 60s.
# karpathy/autoresearch style: each task changes ONE knob from baseline.
# Single seed (s123), E_h=3, cap=30 epochs, patience=7, eval_every=3 (fast).
#
# Baseline (task 0):  lr=1e-3, wd=5e-4, dropout=0.0, readout off,
#                     n_pma_seeds=1, max_grad_norm=5.0
#
# Tasks 1-10 each flip ONE axis. After this finishes, pick top-2 axes
# and combine in Phase 2 with 3 seeds + E_h ∈ {1,2,3}.

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Default = baseline
LR=1e-3
L2_WD=5e-4
DROPOUT=0.0
READOUT_FLAG=()           # off
N_PMA=1
GRAD_NORM=5.0
TAG="baseline"

case $IDX in
    0)  TAG="baseline" ;;
    1)  LR=5e-4;                              TAG="lr5e-4" ;;
    2)  LR=2e-3;                              TAG="lr2e-3" ;;
    3)  L2_WD=1e-3;                           TAG="wd1e-3" ;;
    4)  L2_WD=2e-3;                           TAG="wd2e-3" ;;
    5)  DROPOUT=0.1;                          TAG="dp0.1" ;;
    6)  DROPOUT=0.2;                          TAG="dp0.2" ;;
    7)  READOUT_FLAG=(--readout_concat);      TAG="readout_on" ;;
    8)  N_PMA=2;                              TAG="pma2" ;;
    9)  N_PMA=4;                              TAG="pma4" ;;
    10) GRAD_NORM=1.0;                        TAG="gn1.0" ;;
    *)  echo "Unknown IDX=$IDX" >&2; exit 1 ;;
esac

SEED=123
E_H=3
CLIP_LEN=60
NUM_EPOCHS=30
PATIENCE=7
EVAL_EVERY=3
TRAIN_BS=32
TEST_BS=64
D_MODEL=128

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="sw60_p1_${TAG}_s${SEED}_${SLURM_JOB_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "IDX=$IDX TAG=$TAG"
echo "lr=$LR wd=$L2_WD dropout=$DROPOUT readout=${READOUT_FLAG[*]:-off} n_pma=$N_PMA grad_norm=$GRAD_NORM"
echo "epochs=$NUM_EPOCHS patience=$PATIENCE eval_every=$EVAL_EVERY"
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
    --n_hyper_layers 2 --n_hyperedges "$E_H" \
    --n_pma_seeds "$N_PMA" \
    --aux_type none \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --max_grad_norm "$GRAD_NORM" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    "${READOUT_FLAG[@]}"

echo "End: $(date)"
echo "--- TEST line ---"
find "$SAVE_DIR" -name info.log -exec grep "TEST set prediction" {} \; | tail -1
