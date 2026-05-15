#!/bin/bash
#SBATCH -J gc_pull
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 2:00:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/gc_pull_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/gc_pull_%j.err

set -euo pipefail

# STEP 6: pull GCP graph_cache (3 locations, total ~23 GB) to Phoenix scratch.

GCP_USER=hkim3239
GCP_HOST=34.106.137.183
KEY=$HOME/.ssh/gcp_transfer
SSH_OPT="-e ssh -i $KEY -o StrictHostKeyChecking=accept-new"
DEST_BASE=/storage/scratch1/3/hkim3239/eeg
mkdir -p "$DEST_BASE" "$DEST_BASE/graph_cache" "$DEST_BASE/graph_cache_home" "$DEST_BASE/evobrain/graph_cache" /storage/scratch1/3/hkim3239/logs

echo "Host:  $(hostname)"
echo "Start: $(date)"
df -h /storage/scratch1

echo "=== [1/3] eeg/graph_cache/ ==="
rsync -avhP -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    $GCP_USER@$GCP_HOST:/home/$GCP_USER/eeg/graph_cache/ \
    "$DEST_BASE/graph_cache/"

echo "=== [2/3] ~/graph_cache/ ==="
rsync -avhP -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    $GCP_USER@$GCP_HOST:/home/$GCP_USER/graph_cache/ \
    "$DEST_BASE/graph_cache_home/"

echo "=== [3/3] eeg/evobrain/graph_cache/ ==="
rsync -avhP -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    $GCP_USER@$GCP_HOST:/home/$GCP_USER/eeg/evobrain/graph_cache/ \
    "$DEST_BASE/evobrain/graph_cache/"

echo "End: $(date)"
du -sh "$DEST_BASE/graph_cache" "$DEST_BASE/graph_cache_home" "$DEST_BASE/evobrain/graph_cache"
