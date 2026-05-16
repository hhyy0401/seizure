# Phoenix Cluster Setup

GA Tech PACE Phoenix — GPU partitions, conda envs, sbatch templates.

## GPU partition cheat-sheet

| Partition | GPU | VRAM | SM | Walltime cap |
|---|---|---|---|---|
| `gpu-rtx6000` | Quadro RTX 6000 (Turing) | 24 GB | sm_75 | 3 d |
| `gpu-l40s` | L40S | 48 GB | sm_89 | 3 d |
| `gpu-a100` | A100 | 80 GB | sm_80 | 3 d |
| `gpu-rtxpro-blackwell` | RTX PRO 6000 Blackwell | 96 GB | sm_120 | 3 d |

QOS cap: max 32 concurrent GPUs per user. CPU/GPU ratio max 12:1 on rtx6000.

## Conda environments

| Env | Purpose | Torch | CUDA | mamba_ssm | Notes |
|---|---|---|---|---|---|
| `/storage/scratch1/3/hkim3239/.conda/envs/evobrain` | Training | 2.5.1 | cu121 | 2.2.6 | Works on sm_75, sm_80, sm_89. **NOT** sm_120 |
| `~/miniconda3/envs/fastenv` | Preprocess | — | — | — | No GPU; for FFT/h5 generation |

For Blackwell (`gpu-rtxpro-blackwell`, sm_120): need a fresh cu128 venv —
existing cu121 evobrain env raises kernel errors. See appendix below.

Re-build evobrain env (if ever needed):
```bash
sbatch sbatch/setup_env_evobrain.sbatch
```

## Standard sbatch header

```
#SBATCH -A gts-nimam6-paid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        # max 12 on rtx6000
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -p gpu-rtx6000
```

Bad node `atl1-1-03-007-1-0` has SIGBUS issues — always `--exclude` it:
```bash
sbatch --exclude=atl1-1-03-007-1-0 your.sbatch
```

## Batch size guidance (24 GB rtx6000)

| Model | clip | TRAIN_BS | TEST_BS |
|---|---|---|---|
| evobrain (full DCRNN + Mamba) | 12s | 32 | 64 |
| light_st_hyper | 12s | 128 | 256 |
| light_st_hyper | 60s | 64 | 128 |

Halve all batch sizes for 60s clips.

## Paths

```
Code              : /storage/project/r-nimam6-0/hkim3239/disease/src
sbatch            : /storage/project/r-nimam6-0/hkim3239/disease/sbatch
Run outputs       : /storage/scratch1/3/hkim3239/eeg/runs (heavy, regenerable)
CHB-MIT raw       : /storage/scratch1/3/hkim3239/data/chb_mit
CHB-MIT h5        : /storage/scratch1/3/hkim3239/data/chb_mit_resampled
TUSZ raw          : /storage/scratch1/3/hkim3239/eeg/tusz/v2.0.6
TUSZ h5           : /storage/scratch1/3/hkim3239/eeg/tusz_resampled
TUSZ FFT preproc  : /storage/scratch1/3/hkim3239/eeg/tusz_preproc/clipLen12_timeStepSize1
Graph cache       : src/graph_cache → scratch symlink
```

Logs: `/storage/scratch1/3/hkim3239/logs/*.{out,err}`

## Quick-start training

```bash
# Main model (LightSTHyper) on TUSZ 12s
sbatch --exclude=atl1-1-03-007-1-0 sbatch/train_light_st_hyper.sbatch

# Parameter sweep (45 jobs, ~3h wall)
sbatch --exclude=atl1-1-03-007-1-0 sbatch/sweep_main.sbatch
sbatch --exclude=atl1-1-03-007-1-0 sbatch/sweep_reg.sbatch

# EvoBrain reproduction
sbatch --exclude=atl1-1-03-007-1-0 \
    --export=ALL,MODEL_NAME=evobrain,NUM_EPOCHS=100 \
    sbatch/train_evobrain.sbatch

# Aggregate paper-standard table
python src/scripts/build_results_table.py
```

## Blackwell (rtxpro) one-time setup

The existing cu121 env doesn't support sm_120. Build a separate cu128 venv:

```bash
module load cuda/12.8
python -m venv ~/blackwell_env
source ~/blackwell_env/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu128
pip install mamba_ssm --no-build-isolation
pip install <rest of evobrain reqs>
```

GCP-style throughput (TRAIN_BS=128, 60s clips fit) since VRAM == 96 GB.
