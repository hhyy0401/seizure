#!/bin/bash
#SBATCH -J runs_pull
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 2:00:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/runs_pull_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/runs_pull_%j.err

set -euo pipefail

# STEP 4: pull GCP eeg/runs/ (~21 GB) to Phoenix scratch.
# Idempotent: rsync skips unchanged files; --partial resumes truncated ones.

GCP_USER=hkim3239
GCP_HOST=34.106.137.183
KEY=$HOME/.ssh/gcp_transfer
SRC=$GCP_USER@$GCP_HOST:/home/$GCP_USER/eeg/runs/
DEST=/storage/scratch1/3/hkim3239/eeg/runs/
mkdir -p "$DEST" /storage/scratch1/3/hkim3239/logs

echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "Src:   $SRC"
echo "Dest:  $DEST"
df -h /storage/scratch1

rsync -avhP \
    -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    "$SRC" "$DEST"

echo "End: $(date)"
du -sh "$DEST"
echo "--- top-level contents ---"
ls "$DEST" | head -40
