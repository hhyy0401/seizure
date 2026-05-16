#!/bin/bash
#SBATCH -J sweep_reg
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-8
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sweep_reg_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sweep_reg_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Regularization sweep: dropout × wd = 3 × 3 = 9 jobs.
# Anchor config: TUSZ 12s top-1 = E=3, d_model=64, lr=3e-4, aux=bce.
DROPOUT_LIST=(0.0 0.1 0.2)
WD_LIST=(1e-4 5e-4 1e-3)

IDX=${SLURM_ARRAY_TASK_ID:-0}
DP_I=$(( IDX / 3 ))
WD_I=$(( IDX % 3 ))
DROPOUT=${DROPOUT_LIST[$DP_I]}
L2_WD=${WD_LIST[$WD_I]}

MODEL_NAME="light_st_hyper"
CLIP_LEN=12
NUM_EPOCHS=80
TRAIN_BS=128
TEST_BS=256
LR=3e-4
SEED=123
EVAL_EVERY=1
PATIENCE=10
AUX_TYPE=bce
AUX_WEIGHT=0.3
E_H=3
D_MODEL=64
TAG="sweep-reg"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PREPROC_DIR=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1

RUN_NAME="sweep_reg_${MODEL_NAME}_dp${DROPOUT}_wd${L2_WD}_${CLIP_LEN}s_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"
echo "ArrayIdx=$IDX  dropout=$DROPOUT  l2_wd=$L2_WD  (E=$E_H, d=$D_MODEL, lr=$LR)"
echo "Save: $SAVE_DIR"

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
    --bidirectional \
    --n_hyper_layers 2 \
    --n_hyperedges "$E_H" \
    --rnn_units "$D_MODEL" \
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
    --eval_every "$EVAL_EVERY" \
    --patience "$PATIENCE" \
    --data_augment \
    --preproc_dir "$PREPROC_DIR"

NPZ=$(ls "$SAVE_DIR"/TUSZ/detection/12/*/test_results.npz 2>/dev/null | head -1)
if [ -z "$NPZ" ]; then
    echo "test_results.npz missing — running finalize fallback"
    python scripts/finalize_runs.py "$SAVE_DIR" || true
fi
echo "End: $(date)"
