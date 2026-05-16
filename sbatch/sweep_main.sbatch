#!/bin/bash
#SBATCH -J sweep_main
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-35
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sweep_main_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sweep_main_%A_%a.err

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

# Main TUSZ 12s grid sweep: E × d_model × lr = 4 × 3 × 3 = 36 jobs.
# Fixed: backbone=mamba (BiMamba), aux=bce, dropout=0.0, wd=5e-4,
# aux_w=0.3, use_node_emb=True, n_hyper_layers=2, patience=10.
E_LIST=(2 3 4 5)
D_LIST=(64 96 128)
LR_LIST=(3e-4 5e-4 1e-3)

IDX=${SLURM_ARRAY_TASK_ID:-0}
# Decode IDX → (e_i, d_i, lr_i): IDX = e_i*9 + d_i*3 + lr_i
E_I=$(( IDX / 9 ))
D_I=$(( (IDX / 3) % 3 ))
LR_I=$(( IDX % 3 ))
E_H=${E_LIST[$E_I]}
D_MODEL=${D_LIST[$D_I]}
LR=${LR_LIST[$LR_I]}

MODEL_NAME="light_st_hyper"
CLIP_LEN=12
NUM_EPOCHS=80
TRAIN_BS=128
TEST_BS=256
SEED=123
EVAL_EVERY=1
PATIENCE=10
AUX_TYPE=bce
AUX_WEIGHT=0.3
DROPOUT=0.0
L2_WD=5e-4
TAG="sweep-main"

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
PREPROC_DIR=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1

RUN_NAME="sweep_main_${MODEL_NAME}_E${E_H}_d${D_MODEL}_lr${LR}_${CLIP_LEN}s_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"
echo "ArrayIdx=$IDX  E=$E_H  d_model=$D_MODEL  lr=$LR"
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

# Walltime safety: if main.py exited without dumping test_results.npz (e.g.,
# SIGTERM mid-eval), run finalize fallback from best.pth.tar.
NPZ=$(ls "$SAVE_DIR"/TUSZ/detection/12/*/test_results.npz 2>/dev/null | head -1)
if [ -z "$NPZ" ]; then
    echo "test_results.npz missing — running finalize fallback"
    python scripts/finalize_runs.py "$SAVE_DIR" || true
fi
echo "End: $(date)"
