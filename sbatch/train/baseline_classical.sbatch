#!/bin/bash
#SBATCH -J cls_clsc
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/cls_clsc_%A_%a.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/cls_clsc_%A_%a.err

# Classical-ML baseline runner (SVM / Random Forest) on CHB-MIT or TUSZ.
# Uses the project's FFT dataloader and reduces to a per-clip feature vector
# via either mean-over-time or full flatten.
#
# Examples:
#   sbatch --export=ALL,DATASET=TUSZ,MODEL=svm,FEAT=mean,SEED=123 baseline_classical.sbatch
#   sbatch --export=ALL,DATASET=TUSZ,MODEL=rf,FEAT=flatten,SEED=456 baseline_classical.sbatch
#   sbatch --export=ALL,DATASET=CHBMIT,MODEL=svm,FEAT=mean,SEED=789 baseline_classical.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

DATASET="${DATASET:?must set DATASET=TUSZ|CHBMIT}"
MODEL="${MODEL:?must set MODEL=svm|rf}"
FEAT="${FEAT:?must set FEAT=mean|flatten}"
SEED="${SEED:-123}"
CLIP_LEN="${CLIP_LEN:-12}"
VANILLA_FLAG=()
if [ "${VANILLA:-0}" = "1" ]; then VANILLA_FLAG=(--vanilla); fi

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src

if [ "$DATASET" = "TUSZ" ]; then
    RAW=/storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6/edf
    INPUT=/storage/scratch1/3/hkim3239/eeg/tusz_resampled
    PRE=/storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen${CLIP_LEN}_timeStepSize1
    if [ -d "$PRE" ]; then
        PRE_ARG=(--preproc_dir "$PRE")
    else
        PRE_ARG=()
    fi
else
    RAW=/storage/scratch1/3/hkim3239/data/chb_mit
    INPUT=/storage/scratch1/3/hkim3239/data/chb_mit_resampled
    PRE_ARG=()
fi

RUN_SUFFIX=""
if [ "${VANILLA:-0}" = "1" ]; then RUN_SUFFIX="_vanilla"; fi
RUN=${MODEL}_${DATASET,,}${CLIP_LEN}_${FEAT}${RUN_SUFFIX}_s${SEED}_${SLURM_JOB_ID:-nojob}
SAVE=/storage/scratch1/3/hkim3239/eeg/runs/$RUN
mkdir -p "$SAVE"

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "DATASET=$DATASET MODEL=$MODEL FEAT=$FEAT SEED=$SEED CLIP_LEN=$CLIP_LEN"
echo "Save: $SAVE"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python run_classical.py \
    --dataset "$DATASET" --model "$MODEL" --feature_mode "$FEAT" \
    --seed "$SEED" --clip_len "$CLIP_LEN" \
    --input_dir "$INPUT" --raw_data_dir "$RAW" \
    "${PRE_ARG[@]}" "${VANILLA_FLAG[@]}" \
    --num_workers 16 \
    --save_dir "$SAVE"

echo "End: $(date)"
ls "$SAVE"
