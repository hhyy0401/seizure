#!/bin/bash
#SBATCH -J sw60_p2
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=0-11
#SBATCH -o /storage/scratch1/3/hkim3239/logs/sw60_p2_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/sw60_p2_%A_%a.err

# Round 2 sweep — LightSTHyper TUSZ 60s.
# karpathy/autoresearch step: take Round 1 winner (dp=0.2) as baseline,
# vary ONE knob per task. Single seed (s123), E_h=3, aux=none, no bidir.
# cap=40 epochs, patience=10, eval_every=3.
#
# Tasks 0-9: CLI-only knob sweeps. Task 10: stacked top-axes from Round 1.
# Task 11: Hybrid A (temporal attention after hyperedge layers).

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

IDX=${SLURM_ARRAY_TASK_ID:-0}

# --- defaults = dp=0.2 baseline (Round 1 winner) ---
LR=1e-3
L2_WD=5e-4
DROPOUT=0.2
N_PMA=1
GRAD_NORM=5.0
N_HYPER_LAYERS=2
NUM_RNN_LAYERS=2
D_MODEL=128
EXTRA_FLAGS=()
TAG="dp0.2_ref"

case $IDX in
    0)  TAG="dp0.2_ref" ;;
    1)  DROPOUT=0.3;                                          TAG="dp0.3" ;;
    2)  DROPOUT=0.4;                                          TAG="dp0.4" ;;
    3)  N_PMA=4;                                              TAG="dp0.2_pma4" ;;
    4)  L2_WD=1e-3;                                           TAG="dp0.2_wd1e-3" ;;
    5)  GRAD_NORM=1.0;                                        TAG="dp0.2_gn1.0" ;;
    6)  D_MODEL=192;                                          TAG="dp0.2_d192" ;;
    7)  D_MODEL=256;                                          TAG="dp0.2_d256" ;;
    8)  N_HYPER_LAYERS=3;                                     TAG="dp0.2_hL3" ;;
    9)  NUM_RNN_LAYERS=3;                                     TAG="dp0.2_mL3" ;;
    10) L2_WD=1e-3; GRAD_NORM=1.0; N_PMA=4;                   TAG="dp0.2_stack" ;;
    11) EXTRA_FLAGS=(--temporal_attn);                        TAG="dp0.2_tattn" ;;
    *)  echo "Unknown IDX=$IDX" >&2; exit 1 ;;
esac

SEED=123
E_H=3
CLIP_LEN=60
NUM_EPOCHS=40
PATIENCE=10
EVAL_EVERY=3
TRAIN_BS=32
TEST_BS=64

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
RESAMPLED=/storage/scratch1/3/hkim3239/eeg/tusz_resampled

RUN_NAME="sw60_p2_${TAG}_s${SEED}_${SLURM_JOB_ID}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "IDX=$IDX TAG=$TAG"
echo "lr=$LR wd=$L2_WD dropout=$DROPOUT n_pma=$N_PMA grad_norm=$GRAD_NORM"
echo "n_hyper_layers=$N_HYPER_LAYERS num_rnn_layers=$NUM_RNN_LAYERS d_model=$D_MODEL"
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
    "${EXTRA_FLAGS[@]}"

echo "End: $(date)"
echo "--- TEST line ---"
find "$SAVE_DIR" -name info.log -exec grep "TEST set prediction" {} \; | tail -1
