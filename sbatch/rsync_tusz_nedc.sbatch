#!/bin/bash
#SBATCH -J tusz_dl
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 24:00:00
#SBATCH -p cpu-small
#SBATCH -o /storage/scratch1/3/hkim3239/logs/tusz_dl_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/tusz_dl_%j.err

set -euo pipefail

# TUSZ from NEDC (Temple Univ ISIP), via rsync over SSH.
# Requires ~/.ssh/id_ed25519_nedc registered with NEDC (pulled from GCP as STEP 5).
# Source host: nedc-tuh-eeg@www.isip.piconepress.com
# Override version via: sbatch --export=ALL,TUSZ_VER=v2.0.5 ...

KEY=${NEDC_KEY:-$HOME/.ssh/id_ed25519_nedc}
TUSZ_VER=${TUSZ_VER:-v2.0.6}
DEST=/storage/scratch1/3/hkim3239/eeg/tusz/$TUSZ_VER
mkdir -p "$DEST" /storage/scratch1/3/hkim3239/logs

if [[ ! -f "$KEY" ]]; then
    echo "ERROR: NEDC SSH key not found at $KEY" >&2
    echo "       Run STEP 5 of TRANSFER_COMMANDS.txt first." >&2
    exit 1
fi
chmod 600 "$KEY"

echo "Host:    $(hostname)"
echo "Start:   $(date)"
echo "Version: $TUSZ_VER"
echo "Dest:    $DEST"
echo "Key:     $KEY"
df -h /storage/scratch1

# --partial: 끊겨도 다음 sbatch에서 이어받음 (~79 GB v2.0.6)
rsync -ahP --info=progress2 \
    -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_seizure/$TUSZ_VER/ \
    "$DEST/"

echo "End: $(date)"
du -sh "$DEST"
echo "--- top-level contents ---"
ls "$DEST" | head -20
echo "--- edf file count ---"
find "$DEST" -name '*.edf' | wc -l
