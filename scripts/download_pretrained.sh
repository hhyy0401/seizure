#!/bin/bash
# Fallback fetcher for LaBraM + EEGPT pretrained weights.
#
# In normal use you don't need to run this — the slim weights ship in the
# repo at ckpts/pretrained/ (98 MB EEGPT + 23 MB LaBraM). This script only
# helps when:
#   1. The repo was cloned with a sparse-checkout that excluded ckpts/, or
#   2. You want the ORIGINAL (un-slimmed) EEGPT 974 MB Lightning ckpt, e.g.
#      for re-running the slim pipeline yourself.
#
# Usage:
#   bash scripts/download_pretrained.sh
#   PRETRAINED_DIR=/your/scratch bash scripts/download_pretrained.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
DEFAULT_DIR="$REPO_ROOT/ckpts/pretrained"
PRETRAINED_DIR="${PRETRAINED_DIR:-$DEFAULT_DIR}"

mkdir -p "$PRETRAINED_DIR"

LABRAM_PT="$PRETRAINED_DIR/labram_base.pt"
EEGPT_SLIM="$PRETRAINED_DIR/eegpt_base_slim_fp16.pt"
EEGPT_FULL="$PRETRAINED_DIR/eegpt_mcae_58chs_4s_large4E.ckpt"

echo "=== checking weights in $PRETRAINED_DIR ==="
ok=1
if [ -s "$LABRAM_PT" ]; then
    echo "  ok  $LABRAM_PT ($(du -h "$LABRAM_PT" | cut -f1))"
else
    echo "  --  $LABRAM_PT missing"
    ok=0
fi
if [ -s "$EEGPT_SLIM" ]; then
    echo "  ok  $EEGPT_SLIM ($(du -h "$EEGPT_SLIM" | cut -f1))"
else
    echo "  --  $EEGPT_SLIM missing"
    ok=0
fi
[ "$ok" = "1" ] && { echo; echo "All required weights present. Nothing to do."; exit 0; }

if [ ! -s "$LABRAM_PT" ]; then
    echo
    echo "=== fetching LaBraM-base from HuggingFace (23 MB) ==="
    curl -L -o "$LABRAM_PT" \
        https://huggingface.co/braindecode/Labram-Braindecode/resolve/main/braindecode_labram_base.pt
    ls -lh "$LABRAM_PT"
fi

if [ ! -s "$EEGPT_SLIM" ]; then
    echo
    echo "=== EEGPT slim ckpt missing (98 MB) ==="
    echo "Option A — rebuild from the original Lightning ckpt:"
    echo "    1. download from https://figshare.com/s/e37df4f8a907a866df4b in a"
    echo "       browser (WAF blocks curl), unzip, copy the .ckpt to:"
    echo "         $EEGPT_FULL"
    echo "    2. sbatch sbatch/figures/slim_eegpt.sbatch  # 30 sec on a cpu node"
    echo "       — strips Lightning + drops reconstructor/encoder + fp16 cast"
    echo "       -> writes ${EEGPT_SLIM##*/}"
    echo
    echo "Option B — re-pull the repo with the ckpts/ tree (normal git clone)."
    exit 1
fi

echo
echo "=== ready. point sbatch at these via env if not using defaults ==="
echo "  export LABRAM_CKPT=$LABRAM_PT"
echo "  export EEGPT_CKPT=$EEGPT_SLIM"
