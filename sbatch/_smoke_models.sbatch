#!/bin/bash
#SBATCH -J smoke_models
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:20:00
#SBATCH -p gpu-rtx6000
#SBATCH -o /storage/scratch1/3/hkim3239/logs/smoke_models_%j.out
#SBATCH -e /storage/scratch1/3/hkim3239/logs/smoke_models_%j.err

set -euo pipefail

ENV_PATH=/storage/scratch1/3/hkim3239/.conda/envs/evobrain
REPO=/storage/project/r-nimam6-0/hkim3239/disease/evobrain

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
cd "$REPO"

# Forward + backward smoke test for each of the 4 models with CHB-shaped dummy
# input. Mirrors the dataloader output convention: x is (B, T, N, D), labels
# are scalar binary.
python - <<'PY'
import sys, time, types
sys.path.insert(0, ".")
import torch
import torch.nn as nn

device = torch.device("cuda")
B, T, N, D = 4, 12, 22, 100  # batch, time_steps, channels, FFT feature dim
RNN_UNITS = 64

def make_args():
    a = types.SimpleNamespace()
    a.num_nodes = N
    a.input_dim = D
    a.rnn_units = RNN_UNITS
    a.num_classes = 1
    a.dropout = 0.0
    a.agg = "max"
    a.num_rnn_layers = 2
    a.dcgru_activation = "tanh"
    a.max_diffusion_step = 2
    a.cl_decay_steps = 3000
    a.use_curriculum_learning = False
    return a

def run(model_name, build_fn):
    args = make_args()
    print(f"\n=== {model_name} ===")
    model = build_fn(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    x = torch.randn(B, T, N, D, device=device, requires_grad=False)
    seq_lengths = torch.full((B,), T, device=device, dtype=torch.long)
    adj = torch.rand(B, T, N, N, device=device)
    y = torch.randint(0, 2, (B,), device=device, dtype=torch.float)

    t0 = time.time()
    logits, _ = model(x, seq_lengths, adj)
    if logits.dim() > 1 and logits.shape[-1] == 1:
        logits = logits.view(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    dt = time.time() - t0
    print(f"forward+backward OK  logits.shape={tuple(logits.shape)}  loss={loss.item():.4f}  time={dt*1000:.0f}ms")

# 1) EvoBrain
def _evobrain(args):
    from model.EvoBrain import EvoBrain_classification
    return EvoBrain_classification(args=args, num_classes=args.num_classes, device=device)
run("evobrain", _evobrain)

# 2-4) Light variants
def _light(edge_type):
    def _b(args):
        from model.LightEvoBrain import LightEvoBrain_classification
        return LightEvoBrain_classification(args=args, num_classes=args.num_classes,
                                            device=device, edge_type=edge_type)
    return _b
run("light_dot",       _light("dot"))
run("light_bilinear",  _light("bilinear"))
run("light_attention", _light("attention"))
print("\nALL 4 MODELS PASSED SMOKE TEST")
PY

echo "End: $(date)"
