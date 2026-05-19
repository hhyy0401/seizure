#!/bin/bash
#SBATCH -J sw60_p3
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-11
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sw60_p3_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sw60_p3_%A_%a.err

# Round 3 — LightSTHyper TUSZ 60s.
# Mix: (a) seed verification of Round 2 winners (tattn, hL3) with s456/s789,
#      (b) hL4 extension of hL3 trend,
#      (c) new temporal-axis experiments on top of tattn (Round 2 winner).
# cap=40 epochs, patience=10, eval_every=3, dropout baseline=0.2.

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Defaults = Round 2 winner baseline (dp=0.2 + tattn, single layer).
SEED=123
DROPOUT=0.2
N_HYPER_LAYERS=2
N_PMA=1
TATTN_FLAGS=(--temporal_attn)        # tattn ON by default
TATTN_N_LAYERS=1
EXTRA_FLAGS=()
TAG="tattn_ref"

case $IDX in
    # --- seed verification of Round 2 winners (s456, s789) ---
    0)  SEED=456;                                                                  TAG="tattn_s456" ;;
    1)  SEED=789;                                                                  TAG="tattn_s789" ;;
    2)  SEED=456; TATTN_FLAGS=(); N_HYPER_LAYERS=3;                                TAG="hL3_s456" ;;
    3)  SEED=789; TATTN_FLAGS=(); N_HYPER_LAYERS=3;                                TAG="hL3_s789" ;;
    # --- hL trend extension ---
    4)  TATTN_FLAGS=(); N_HYPER_LAYERS=4;                                          TAG="hL4" ;;
    # --- tattn-base stacks ---
    5)  N_HYPER_LAYERS=3;                                                          TAG="tattn+hL3" ;;
    6)  N_HYPER_LAYERS=4;                                                          TAG="tattn+hL4" ;;
    7)  TATTN_N_LAYERS=2;                                                          TAG="tattn_2L" ;;
    8)  EXTRA_FLAGS=(--tattn_pos_enc);                                             TAG="tattn_posenc" ;;
    9)  EXTRA_FLAGS=(--use_fft_mixer);                                             TAG="tattn_FFT" ;;
    10) DROPOUT=0.3;                                                               TAG="tattn+dp0.3" ;;
    11) EXTRA_FLAGS=(--tattn_causal);                                              TAG="tattn_causal" ;;
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
NUM_RNN_LAYERS=2
LR=1e-3
L2_WD=5e-4
GRAD_NORM=5.0

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="sw60_p3_${TAG}_s${SEED}_${SLURM_JOB_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "IDX=$IDX TAG=$TAG SEED=$SEED"
echo "dp=$DROPOUT n_hyper=$N_HYPER_LAYERS pma=$N_PMA tattn=${TATTN_FLAGS[*]:-off} tattn_n_layers=$TATTN_N_LAYERS"
echo "extra=${EXTRA_FLAGS[*]:-none}"
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
    --num_rnn_layers "$NUM_RNN_LAYERS" \
    --n_hyper_layers "$N_HYPER_LAYERS" --n_hyperedges "$E_H" \
    --n_pma_seeds "$N_PMA" \
    --aux_type none \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --max_grad_norm "$GRAD_NORM" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    --tattn_n_layers "$TATTN_N_LAYERS" \
    "${TATTN_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo "End: $(date)"
echo "--- TEST line ---"
find "$SAVE_DIR" -name info.log -exec grep "TEST set prediction" {} \; | tail -1
