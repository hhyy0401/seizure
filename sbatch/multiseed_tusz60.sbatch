#!/bin/bash
#SBATCH -J ms_tusz60
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-8
#SBATCH -o /storage/scratch1/3/hkim3239/logs/ms_tusz60_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/ms_tusz60_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Same top-3 cfgs on TUSZ 60s, × 3 seeds.
# 60s clips are ~5× larger → halve TRAIN_BS, NUM_EPOCHS=100.
E_ARR=(3   5   3)
D_ARR=(128 96  64)
LR_ARR=(1e-3 1e-3 3e-4)
BS_ARR=(32  48  64)
TEST_BS_ARR=(64 96 128)
SEED_ARR=(123 456 789)

IDX=${SLURM_ARRAY_TASK_ID:-0}
CFG_IDX=$(( IDX % 3 ))
SEED_IDX=$(( IDX / 3 ))

E_H=${E_ARR[$CFG_IDX]}
D_MODEL=${D_ARR[$CFG_IDX]}
LR=${LR_ARR[$CFG_IDX]}
TRAIN_BS=${BS_ARR[$CFG_IDX]}
TEST_BS=${TEST_BS_ARR[$CFG_IDX]}
SEED=${SEED_ARR[$SEED_IDX]}

MODEL_NAME="light_st_hyper"
CLIP_LEN=60
NUM_EPOCHS=100
PATIENCE=10
AUX_TYPE=bce
AUX_WEIGHT=0.3
DROPOUT=0.0
L2_WD=5e-4

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="ms_tusz60_E${E_H}_d${D_MODEL}_lr${LR}_s${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)  ArrayIdx=$IDX  cfg$CFG_IDX (E=$E_H d=$D_MODEL lr=$LR bs=$TRAIN_BS)  seed=$SEED"
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
    --use_fft \
    --use_node_emb \
    --bidirectional \
    --rnn_units "$D_MODEL" \
    --n_hyper_layers 2 \
    --n_hyperedges "$E_H" \
    --aux_type "$AUX_TYPE" \
    --aux_weight "$AUX_WEIGHT" \
    --dropout "$DROPOUT" \
    --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" \
    --test_batch_size "$TEST_BS" \
    --lr_init "$LR" \
    --num_workers 8 \
    --rand_seed "$SEED" \
    --metric_name auroc \
    --eval_every 1 \
    --patience "$PATIENCE" \
    --data_augment

if [ ! -f "$SAVE_DIR/TUSZ/detection/$CLIP_LEN"/*/test_results.npz ]; then
    python scripts/finalize_runs.py "$SAVE_DIR" || true
fi

echo "End: $(date)"
ls "$SAVE_DIR"
