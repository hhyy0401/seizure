#!/bin/bash
#SBATCH -J chb_test
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -p gpu-a100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/chb_test_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/chb_test_%j.err

# Test-only evaluation from an existing best.pth.tar.
# Re-uses the original run's args.json to keep model/config consistent;
# only --test, --load_model_path, --save_dir are overridden.
#
# Usage:
#   sbatch --export=ALL,SAVE_DIR=/path/to/run/.../<model>_<graph>_<seed>_01 \
#          sbatch/train/baseline_chb_test_only.sbatch
#
# SAVE_DIR must contain both best.pth.tar and args.json.

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

SAVE_DIR="${SAVE_DIR:?must set SAVE_DIR=/path/to/run/.../<tag>_01}"
CKPT="$SAVE_DIR/best.pth.tar"
ARGS_JSON="$SAVE_DIR/args.json"

[ -f "$CKPT" ] || { echo "Missing $CKPT" >&2; exit 1; }
[ -f "$ARGS_JSON" ] || { echo "Missing $ARGS_JSON" >&2; exit 1; }

# Pull fields we need from args.json so the test run matches the original.
python3 - <<PY > /tmp/test_args_$$.sh
import json
a = json.load(open("$ARGS_JSON"))
keys = ["dataset","task","model_name","num_nodes","raw_data_dir","input_dir",
        "max_seq_len","time_step_size","graph_type","top_k","use_fft",
        "dropout","l2_wd","train_batch_size","test_batch_size",
        "lr_init","rand_seed","metric_name"]
for k in keys:
    v = a.get(k)
    if isinstance(v, bool):
        if v: print(f"--{k}")
    elif v is not None:
        print(f"--{k} {v}")
PY
mapfile -t EXTRA_ARGS < /tmp/test_args_$$.sh
rm -f /tmp/test_args_$$.sh

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "SAVE_DIR=$SAVE_DIR"
echo "Reusing args from $ARGS_JSON"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

python main.py \
    --test --load_model_path "$CKPT" \
    --save_dir "$SAVE_DIR" \
    --num_workers 16 \
    "${EXTRA_ARGS[@]}"

echo "End: $(date)"
echo "--- TEST set line ---"
grep "TEST set" "$SAVE_DIR/../../../../../logs/"*.out 2>/dev/null | tail -1 || true
tail -5 "$SAVE_DIR/info.log" 2>/dev/null || true
