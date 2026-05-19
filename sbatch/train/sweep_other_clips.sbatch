#!/bin/bash
#SBATCH -J sw_other
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-1
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sw_other_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sw_other_%A_%a.err

# Spin-off: apply current best config (tattn+hL3) to TUSZ 12s and CHB-MIT 12s
# for a quick "각 보기" measurement. Single seed s123, cap=40, patience=10.

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Shared best config from Round 3 leader (tattn+hL3) — cap=50, eval_every=3
DROPOUT=0.2
N_HYPER_LAYERS=3
N_HYPEREDGES=3
N_PMA=1
LR=1e-3
L2_WD=5e-4
SEED=123
NUM_EPOCHS=50
PATIENCE=10
EVAL_EVERY=3
D_MODEL=128

PRE_ARG=()

case $IDX in
    0)  # TUSZ 12s — copy resampled to local /tmp for fast small-file reads
        TAG="tusz12_tattn_hL3"
        DATASET=TUSZ
        NUM_NODES=19
        CLIP_LEN=12
        TRAIN_BS=128
        TEST_BS=1024
        RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6/edf
        RESAMPLED_SRC=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
        ;;
    1)  # CHB-MIT 12s — copy resampled to local /tmp
        TAG="chbmit12_tattn_hL3"
        DATASET=CHBMIT
        NUM_NODES=22
        CLIP_LEN=12
        TRAIN_BS=128
        TEST_BS=1024
        RAW=/storage/scratch1/3/hkim3239/data/chb_mit
        RESAMPLED_SRC=/storage/scratch1/3/hkim3239/data/chb_mit_resampled
        ;;
    *)  echo "Unknown IDX=$IDX" >&2; exit 1 ;;
esac

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src

RUN_NAME="sw_other_${TAG}_s${SEED}_${SLURM_JOB_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "IDX=$IDX TAG=$TAG DATASET=$DATASET CLIP_LEN=$CLIP_LEN NUM_NODES=$NUM_NODES"
echo "Config: dp=$DROPOUT n_hyper=$N_HYPER_LAYERS E_h=$N_HYPEREDGES tattn=ON"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

# === Copy resampled data to local /tmp for fast small-file reads ===
LOCAL_DATA=/tmp/${USER}_eeg_${SLURM_JOB_ID:-nojob}
RESAMPLED="$LOCAL_DATA/resampled"
mkdir -p "$RESAMPLED"
echo "=== Copying $RESAMPLED_SRC → $RESAMPLED via tar pipe ==="
COPY_START=$(date +%s)
(cd "$RESAMPLED_SRC" && tar cf - .) | (cd "$RESAMPLED" && tar xf -)
COPY_END=$(date +%s)
echo "Copy elapsed: $((COPY_END - COPY_START))s"
echo "Local size: $(du -sh $RESAMPLED 2>&1 | head -1)"
echo "Local file count: $(ls $RESAMPLED | wc -l)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py --dataset "$DATASET" --task detection --model_name light_st_hyper \
    --num_nodes "$NUM_NODES" \
    --raw_data_dir "$RAW" --input_dir "$RESAMPLED" \
    "${PRE_ARG[@]}" \
    --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units "$D_MODEL" \
    --n_hyper_layers "$N_HYPER_LAYERS" --n_hyperedges "$N_HYPEREDGES" \
    --n_pma_seeds "$N_PMA" \
    --aux_type none \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 16 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    --temporal_attn

echo "End: $(date)"
echo "--- TEST line ---"
find "$SAVE_DIR" -name info.log -exec grep "TEST set prediction" {} \; | tail -1
