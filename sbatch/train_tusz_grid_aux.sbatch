#!/bin/bash
#SBATCH -J tusz_grid
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-17
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz_grid_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz_grid_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# ---------------------------------------------------------------------------
# Grid:  backbone (mamba | dwsep)
#      × aux_type (none  | bce | entropy)
#      × E_h      (2 | 3 | 4)
# = 18 jobs. Built from the current best:
#   light_st_hyper_* + use_node_emb=true, n_hyper_layers=2, 12s TUSZ.
# Model selection by AUROC (default). F1 is auto-reported at F1-best τ tuned
# on dev (handled by main.py since --fix_threshold is unset).
# ---------------------------------------------------------------------------
BACKBONES=(mamba mamba mamba mamba mamba mamba mamba mamba mamba \
           dwsep dwsep dwsep dwsep dwsep dwsep dwsep dwsep dwsep)
AUX_TYPES=(none  none  none  bce   bce   bce   entropy entropy entropy \
           none  none  none  bce   bce   bce   entropy entropy entropy)
E_LIST=(   2     3     4     2     3     4     2       3       4 \
           2     3     4     2     3     4     2       3       4)

IDX=${SLURM_ARRAY_TASK_ID:-0}
BACKBONE=${BACKBONES[$IDX]}
AUX_TYPE=${AUX_TYPES[$IDX]}
E_H=${E_LIST[$IDX]}

case "$BACKBONE" in
    mamba) MODEL_NAME="light_st_hyper" ;;
    dwsep) MODEL_NAME="light_st_hyper_dwsep" ;;
    *) echo "unknown backbone: $BACKBONE"; exit 1 ;;
esac

# ---- training config ------------------------------------------------------
CLIP_LEN="${CLIP_LEN:-12}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-128}"
TEST_BS="${TEST_BS:-256}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PATIENCE="${PATIENCE:-15}"
AUX_WEIGHT="${AUX_WEIGHT:-0.3}"
TAG="${TAG:-grid}"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/evobrain
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

# Optional preproc dir (huge speedup). Empty → on-the-fly FFT.
# Set via: sbatch --export=ALL,PREPROC_DIR=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1 ...
PREPROC_DIR="${PREPROC_DIR:-}"
PREPROC_ARG=""
if [[ -n "$PREPROC_DIR" && -d "$PREPROC_DIR" ]]; then
    PREPROC_ARG="--preproc_dir $PREPROC_DIR"
    echo "Using precomputed clips: $PREPROC_DIR"
else
    echo "No preproc dir; computing FFT on the fly (slower)."
fi

RUN_NAME="tusz_${MODEL_NAME}_NE_${BACKBONE}_aux-${AUX_TYPE}_E${E_H}_${CLIP_LEN}s_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SAVE_BASE=/storage/scratch1/3/hkim3239/eeg/runs
SAVE_DIR="$SAVE_BASE/$RUN_NAME"
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "ArrayIdx=$IDX  backbone=$BACKBONE  aux=$AUX_TYPE  E_h=$E_H  model=$MODEL_NAME"
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
    --data_augment \
    $PREPROC_ARG

echo "End: $(date)"
echo "--- save dir ---"
ls "$SAVE_DIR"
