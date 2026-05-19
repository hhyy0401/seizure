#!/bin/bash
#SBATCH -J chb_lbd
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/chb_lbd_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/chb_lbd_%A_%a.err

# CHB-MIT 12s detection baseline runner for {lstm, cnnlstm, BIOT, dcrnn, gru_gcn}.
#
# Mirrors baseline_chb.sbatch hyperparameters (epochs / bs / lr / patience).
# Per-model graph_type:
#   lstm / cnnlstm  -> none (no graph)
#   BIOT            -> none (raw input; main.py forces use_fft=False)
#   dcrnn / gru_gcn -> individual (correlation graph; combined-3D not available on CHB)
#
# Examples:
#   sbatch --array=0-2 --export=ALL,MODEL=cnnlstm  baseline_chb_lbd.sbatch
#   sbatch --array=0-2 --export=ALL,MODEL=gru_gcn  baseline_chb_lbd.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

MODEL="${MODEL:?must set MODEL=lstm|cnnlstm|BIOT|dcrnn|gru_gcn}"
CLIP_LEN="${CLIP_LEN:-12}"

SEEDS=(123 456 789)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
else
    SEED="${SEED:-123}"
fi

if [ "$CLIP_LEN" = "60" ]; then
    NUM_EPOCHS="${NUM_EPOCHS:-100}"
    TRAIN_BS="${TRAIN_BS:-32}"
    TEST_BS="${TEST_BS:-128}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-100}"
    TRAIN_BS="${TRAIN_BS:-128}"
    TEST_BS="${TEST_BS:-512}"
fi

LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-15}"
L2_WD="${L2_WD:-5e-4}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-2}"

case "$MODEL" in
    lstm|cnnlstm)  GRAPH_TYPE="none";       FFT_FLAG=(--use_fft) ;;
    BIOT)          GRAPH_TYPE="none";       FFT_FLAG=() ;;  # main.py forces use_fft=False
    dcrnn|gru_gcn) GRAPH_TYPE="individual"; FFT_FLAG=(--use_fft) ;;
    *) echo "Unsupported MODEL='$MODEL'" >&2; exit 1 ;;
esac

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
RAW=/storage/scratch1/3/hkim3239/data/chb_mit
RESAMPLED=/storage/scratch1/3/hkim3239/data/chb_mit_resampled

RUN_NAME="${MODEL}_chb${CLIP_LEN}_s${SEED}_${SLURM_JOB_ID:-nojob}"
SAVE_DIR=/storage/scratch1/3/hkim3239/eeg/runs/$RUN_NAME
mkdir -p "$SAVE_DIR"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "MODEL=$MODEL  CLIP_LEN=$CLIP_LEN  SEED=$SEED  GRAPH_TYPE=$GRAPH_TYPE"
echo "Save: $SAVE_DIR"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py \
    --dataset CHBMIT --task detection --model_name "$MODEL" \
    --num_nodes 22 \
    --raw_data_dir "$RAW" --input_dir "$RESAMPLED" \
    --save_dir "$SAVE_DIR" \
    --max_seq_len "$CLIP_LEN" --time_step_size 1 \
    --graph_type "$GRAPH_TYPE" --top_k 3 \
    "${FFT_FLAG[@]}" \
    --dropout "$DROPOUT" --l2_wd "$L2_WD" \
    --num_epochs "$NUM_EPOCHS" \
    --train_batch_size "$TRAIN_BS" --test_batch_size "$TEST_BS" \
    --lr_init "$LR" --num_workers 8 \
    --rand_seed "$SEED" --metric_name auroc \
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE" --data_augment \
    --skip_midtest

echo "End: $(date)"
ls "$SAVE_DIR"
