#!/bin/bash
# Download the LaBraM + EEGPT pretrained weights into $PRETRAINED_DIR
# (defaults to /storage/scratch1/$USER/eeg/pretrained on Phoenix).
#
# Usage:
#   bash scripts/download_pretrained.sh
#   PRETRAINED_DIR=/your/scratch bash scripts/download_pretrained.sh

set -euo pipefail

DEFAULT_DIR="/storage/scratch1/3/${USER}/eeg/pretrained"
PRETRAINED_DIR="${PRETRAINED_DIR:-$DEFAULT_DIR}"

mkdir -p "$PRETRAINED_DIR/labram" "$PRETRAINED_DIR/eegpt"

LABRAM_PT="$PRETRAINED_DIR/labram/braindecode_labram_base.pt"
EEGPT_CKPT="$PRETRAINED_DIR/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt"

echo "=== 1/2: LaBraM-base from HuggingFace (23 MB, automatic) ==="
if [ -s "$LABRAM_PT" ]; then
    echo "already at $LABRAM_PT — skipping"
else
    curl -L -o "$LABRAM_PT" \
        https://huggingface.co/braindecode/Labram-Braindecode/resolve/main/braindecode_labram_base.pt
fi
ls -lh "$LABRAM_PT"

echo
echo "=== 2/2: EEGPT-large from figshare (974 MB, MANUAL) ==="
if [ -s "$EEGPT_CKPT" ]; then
    echo "already at $EEGPT_CKPT — skipping"
else
    cat <<EOF
figshare's WAF blocks bot downloads, so curl / wget cannot fetch this.

Manual steps (one-time):
  1. Open in a browser:
     https://figshare.com/s/e37df4f8a907a866df4b
  2. Click "Download all" — you get a ~975 MB zip (e.g. 25866970.zip).
  3. Unzip and copy ONE file:
       EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
     to:
       $EEGPT_CKPT

When that file is in place, re-run this script to verify size.
EOF
    exit 1
fi
ls -lh "$EEGPT_CKPT"

echo
echo "=== both weights ready under $PRETRAINED_DIR ==="
echo "Export these env vars (or pass --pretrained_path) in your sbatch:"
echo "  export LABRAM_CKPT=$LABRAM_PT"
echo "  export EEGPT_CKPT=$EEGPT_CKPT"
