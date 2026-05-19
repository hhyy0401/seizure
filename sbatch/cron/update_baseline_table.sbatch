#!/bin/bash
#SBATCH -J upd_basetbl
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 0:10:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/upd_basetbl_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/upd_basetbl_%j.err

# Phoenix self-resubmitting hourly updater for docs/BASELINE_TABLE.md.
# Replaces a missing user-crontab (Phoenix disallows crontab -e for users).
#
# Behavior:
#   1. Run the python parser → rewrites BASELINE_TABLE.md in place.
#   2. Re-submit THIS script with --begin=now+1hour so the next run fires
#      ~1h after this job ends.
#
# To stop the chain: scancel the next pending job (look for J=upd_basetbl
# in `squeue -u $USER`) or set NO_RESUBMIT=1 in env when starting.
#
# To start the chain initially:
#   sbatch /storage/project/r-nimam6-0/hkim3239/disease/sbatch/cron/update_baseline_table.sbatch

set -euo pipefail
mkdir -p /storage/scratch1/3/hkim3239/logs

REPO=/storage/project/r-nimam6-0/hkim3239/disease
PY="/storage/home/hcoda1/3/hkim3239/miniconda3/envs/fastenv/bin/python3"

echo "=== upd_basetbl run @ $(date) on $(hostname) ==="

if [ -x "$PY" ]; then
    "$PY" "$REPO/scripts/update_baseline_table.py"
else
    # fallback to system python3 if fastenv missing
    python3 "$REPO/scripts/update_baseline_table.py"
fi

echo "=== done @ $(date) ==="

# Re-arm for ~1h later, unless explicitly disabled.
if [ "${NO_RESUBMIT:-0}" != "1" ]; then
    sbatch --begin="now+1hour" "$REPO/sbatch/cron/update_baseline_table.sbatch"
    echo "=== resubmitted; next run ~1h from now ==="
else
    echo "=== NO_RESUBMIT=1; chain stopped ==="
fi
