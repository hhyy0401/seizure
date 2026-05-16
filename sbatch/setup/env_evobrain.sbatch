#!/bin/bash
#SBATCH -J evo_env
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -p gpu-v100
#SBATCH -o /storage/scratch1/3/hkim3239/logs/evo_env_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/evo_env_%j.err

set -euo pipefail

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/src
mkdir -p /storage/scratch1/3/hkim3239/logs

echo "Host: $(hostname)"
echo "Start: $(date)"

# Conda init
source $(conda info --base)/etc/profile.d/conda.sh
nvidia-smi || true
nvcc --version 2>/dev/null || echo "(no nvcc on PATH; using torch's bundled CUDA runtime)"

# 1) Create env
if [[ ! -d "$ENV_PATH" ]]; then
    conda create -y -p "$ENV_PATH" -c conda-forge --override-channels python=3.11
fi
conda activate "$ENV_PATH"
python -V
which python

PIP="$ENV_PATH/bin/pip"

# 2) Torch 2.5.1 + cu121 (matches mamba-ssm 2.2.4 wheel naming)
$PIP install --upgrade pip
# Install numpy first so torch's numpy interop check doesn't warn.
$PIP install "numpy<2"
$PIP install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Sanity: torch + CUDA visible
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# 4) causal-conv1d 1.5.4 + mamba-ssm 2.2.6.post3 wheels for torch2.5 + cu12 + cp311
# (2.2.4 specified in EvoBrain's requirements.txt pairs with causal-conv1d 1.4.x
#  for which no torch2.5 wheel exists; 2.2.6.post3 is the latest 2.2.x with a
#  torch2.5 wheel and is ABI-compatible with causal-conv1d 1.5.x.)
$PIP install --force-reinstall --no-deps \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
$PIP install --force-reinstall --no-deps \
    https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 5) PyG 2.3.1 + scatter/sparse for cu121/torch2.5
$PIP install torch-geometric==2.3.1
$PIP install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html || true

# 6) Remaining requirements. Drop very old strict pins that conflict with
# Python 3.11 (numpy 1.23.5, scipy 1.9.1, scikit-learn 1.1.2, etc.), keep
# the rest at the pinned versions.
$PIP install \
    "numpy<2" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "pandas>=1.5" \
    h5py==3.11.0 \
    pyedflib==0.1.42 \
    tqdm==4.64.1 \
    tensorboardX==2.6.2.2 \
    transformers==4.39.3 \
    matplotlib==3.8.4 \
    graphviz==0.18.2 \
    dotted-dict==1.1.3 \
    networkx==3.1 \
    linear_attention_transformer==0.19.1 \
    more_itertools==8.12.0 \
    opt_einsum==3.4.0 \
    pytorch_lightning==2.5.0.post0

# 7) Validate end-to-end import
python - <<'PY'
import torch
from mamba_ssm import Mamba
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import pyedflib, h5py, scipy
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 8, 64, device="cuda")
y = m(x)
print("Mamba forward OK", y.shape)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("pyg", torch_geometric.__version__)
print("scipy", scipy.__version__, "pyedflib", pyedflib.__version__)
PY

# 8) Smoke-import our model wrapper
cd "$REPO"
python - <<'PY'
import sys, importlib
sys.path.insert(0, ".")
import argparse
from model.EvoBrain import EvoBrain_classification
from model.NodeMambaGCN import NodeMambaGCN_classification
print("EvoBrain + NodeMambaGCN imports OK")
PY

echo "End: $(date)"
du -sh "$ENV_PATH"
