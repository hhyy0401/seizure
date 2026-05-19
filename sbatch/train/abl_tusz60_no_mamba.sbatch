#!/bin/bash
#SBATCH -J abl_tusz60_no_mamba
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-2
#SBATCH -o /storage/scratch1/3/hkim3239/logs/abl_tusz60_no_mamba_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/abl_tusz60_no_mamba_%A_%a.err

# Ablation on TUSZ 60s: replace BiMamba backbone with LinearBackbone
# (per-channel per-t linear projection; spatial mixing still by hyperedge).
# Baseline = uni-Mamba + no-aux + E_h=3 + node_emb.
set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

SEEDS=(123 456 789)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$IDX]}

MODEL_NAME=light_st_hyper_linear
CLIP_LEN=60; NUM_EPOCHS=100; TRAIN_BS=32; TEST_BS=64
LR=1e-3; PATIENCE=10
D_MODEL=128; DROPOUT=0.0; L2_WD=5e-4
E_H=3

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="abl_tusz60_no_mamba_s${SEED}_${SLURM_JOB_ID}_${IDX}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py --dataset TUSZ --task detection --model_name $MODEL_NAME \
    --num_nodes 19 \
    --raw_data_dir "$RAW/edf" --input_dir "$RESAMPLED" \
    --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units "$D_MODEL" \
    --n_hyper_layers 2 --n_hyperedges "$E_H" \
    --aux_type none \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every 1 --patience "$PATIENCE" --data_augment

TR=$(find "$SAVE_DIR" -name test_results.npz | head -1)
[ -z "$TR" ] && python scripts/finalize_runs.py "$SAVE_DIR" || true
