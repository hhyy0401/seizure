#!/bin/bash
#SBATCH -J tusz60_top3
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-2
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz60_top3_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz60_top3_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Top-3 TUSZ 12s winners → re-run on TUSZ 60s clips (no preproc dir;
# on-the-fly FFT — CHB-MIT 12s OTFFT runs at ~190 it/s so should be OK).
#   j0: mamba E=3 bce
#   j1: mamba E=3 none
#   j2: mamba E=4 bce
AUX_TYPES=(bce none bce)
E_LIST=(   3   3    4)

IDX=${SLURM_ARRAY_TASK_ID:-0}
AUX_TYPE=${AUX_TYPES[$IDX]}
E_H=${E_LIST[$IDX]}
MODEL_NAME="light_st_hyper"

CLIP_LEN="${CLIP_LEN:-60}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-64}"      # 60s clips are 5× bigger than 12s — halve BS
TEST_BS="${TEST_BS:-128}"
LR="${LR:-3e-4}"
SEED="${SEED:-123}"
EVAL_EVERY="${EVAL_EVERY:-1}"
PATIENCE="${PATIENCE:-10}"
AUX_WEIGHT="${AUX_WEIGHT:-0.3}"
TAG="${TAG:-tusz60-top3}"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="tusz60_${MODEL_NAME}_NE_aux-${AUX_TYPE}_E${E_H}_${CLIP_LEN}s_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SAVE_BASE=/storage/scratch1/3/hkim3239/eeg/runs
SAVE_DIR="$SAVE_BASE/$RUN_NAME"
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "ArrayIdx=$IDX  model=$MODEL_NAME (mamba)  aux=$AUX_TYPE  E_h=$E_H  lr=$LR  patience=$PATIENCE"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

cd "$REPO"

python main.py \
    --dataset TUSZ \
    --task detection \
    --model_name "$MODEL_NAME" \
    --num_nodes 19 \
    --raw_data_dir "$RAW/edf" \
    --input_dir "$RESAMPLED" \
    --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" \
    --time_step_size 1 \
    --graph_type none \
    --top_k 3 \
    --use_fft \
    --use_node_emb \
    --n_hyper_layers 2 \
    --n_hyperedges "$E_H" \
    --aux_type "$AUX_TYPE" \
    --aux_weight "$AUX_WEIGHT" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" \
    --test_batch_size "$TEST_BS" \
    --lr_init "$LR" \
    --num_workers 8 \
    --rand_seed "$SEED" \
    --metric_name auroc \
    --eval_every "$EVAL_EVERY" \
    --patience "$PATIENCE" \
    --data_augment

echo "End: $(date)"
echo "--- save dir ---"
ls "$SAVE_DIR"
