#!/bin/bash
#SBATCH -J chbmit_dl
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/chbmit_dl_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/chbmit_dl_%j.err

set -euo pipefail

DEST=/storage/scratch1/3/hkim3239/data/chb_mit
mkdir -p "$DEST"
mkdir -p /storage/scratch1/3/hkim3239/logs

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Dest:  $DEST"
df -h /storage/scratch1

# PhysioNet CHB-MIT Scalp EEG Database v1.0.0 (~42 GB, 24 patients)
# Use AWS S3 mirror (physionet-open bucket) — much faster than HTTP from physionet.org
AWS=/storage/home/hcoda1/3/hkim3239/.local/bin/aws

# --no-sign-request: public bucket, no creds
# sync resumes by skipping files that already match in size+mtime
$AWS s3 sync --no-sign-request \
    s3://physionet-open/chbmit/1.0.0/ \
    "$DEST/"

echo "End: $(date)"
du -sh "$DEST"
echo "--- top-level contents ---"
ls "$DEST" | head -40
echo "--- file count ---"
find "$DEST" -type f | wc -l
