#!/bin/bash
#SBATCH -J tusz_gcp
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz_gcp_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz_gcp_%j.err

set -euo pipefail

# Pull TUSZ v2.0.6 directly from GCP source (where it was already downloaded).
# 79 GB, ~8140 EDFs. Reproducible: GCP path == /home/hkim3239/eeg/tusz/v2.0.6/

GCP_USER=hkim3239
GCP_HOST=34.106.137.183
KEY=$HOME/.ssh/gcp_transfer
TUSZ_VER=${TUSZ_VER:-v2.0.6}
SRC=$GCP_USER@$GCP_HOST:/home/$GCP_USER/eeg/tusz/$TUSZ_VER/
DEST=/storage/scratch1/3/hkim3239/eeg/tusz/$TUSZ_VER/
mkdir -p "$DEST" /storage/scratch1/3/hkim3239/logs

echo "Host:    $(hostname)"
echo "Start:   $(date)"
echo "Version: $TUSZ_VER"
echo "Src:     $SRC"
echo "Dest:    $DEST"
df -h /storage/scratch1 | tail -1

# --partial: 끊겨도 다음 sbatch에서 이어받음
rsync -ahP --info=progress2 \
    -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    "$SRC" "$DEST"

echo "End: $(date)"
du -sh --apparent-size "$DEST"
echo "--- edf file count ---"
find "$DEST" -name '*.edf' | wc -l
echo "--- top-level ---"
ls "$DEST"
