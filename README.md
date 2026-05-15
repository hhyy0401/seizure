# EvoBrain / light_st_hyper — EEG Seizure Detection

Spatio-temporal hyperedge models for seizure detection on CHB-MIT and TUSZ.

Built on top of [Kotoge/EvoBrain](https://github.com/Kotoge/EvoBrain) (MIT license — `evobrain/LICENSE`),
extended with new model variants (`light_st_hyper`, `light_mamba_band_plv`,
`ada_mshyper`, etc.) and Phoenix HPC training infrastructure.

---

## Repo layout

```
disease/
├── evobrain/                      # Core code (forked from Kotoge/EvoBrain, modified)
│   ├── main.py                    # Training entry point
│   ├── model/                     # All model variants (EvoBrain, LightEvoBrain, …)
│   ├── data/                      # Dataloaders, preprocess scripts, file_markers
│   ├── args.py, constants.py, utils.py
│   └── ...
├── sbatch/                        # SLURM scripts for Phoenix
│   ├── train_light_st_hyper.sbatch
│   ├── train_evobrain.sbatch
│   ├── train_chb.sbatch
│   ├── preprocess_chb.sbatch
│   ├── preprocess_tusz.sbatch
│   ├── download_chb_mit.sbatch
│   ├── rsync_tusz_nedc.sbatch
│   └── setup_env_evobrain.sbatch  # one-shot env builder
├── PHOENIX_SETUP_rtx6000.txt      # Setup notes for gpu-rtx6000
├── PHOENIX_SETUP_rtxpro.txt       # Setup notes for gpu-rtxpro-blackwell
├── light_evobrain_pseudocode.md   # Model design sketch
├── figures/experiments_summary.md # Methodology summary
└── evobrain.pdf                   # Project writeup
```

Heavy artifacts (`runs/`, `graph_cache/`, `external_baselines/`, `backup/`) are
gitignored — regenerate them by training/preprocessing.

---

## Quick start (training the current model)

The **current canonical model** is `light_st_hyper` (Light EvoBrain with
spatio-temporal hyperedge block, ~80K params).

### 1) Environment (one-time)

Build the conda env (Phoenix `gts-nimam6-paid` account):

```bash
sbatch sbatch/setup_env_evobrain.sbatch
```

This creates `<scratch>/.conda/envs/evobrain` with:
- python 3.11, torch 2.5.1+cu121
- mamba-ssm 2.2.6.post3 + causal-conv1d 1.5.4
- torch-geometric 2.3.1, pyg-scatter
- numpy, scipy, h5py, pyedflib, tensorboardX, ...

Existing env path expected: `/storage/scratch1/3/hkim3239/.conda/envs/evobrain`
(edit `ENV_PATH=` in sbatch headers to match your setup).

### 2) Data

- **CHB-MIT**: `sbatch sbatch/download_chb_mit.sbatch` then `sbatch sbatch/preprocess_chb.sbatch`
- **TUSZ v2.0.6** (requires NEDC SSH key `~/.ssh/id_ed25519_nedc`):
  `sbatch sbatch/rsync_tusz_nedc.sbatch` then `sbatch sbatch/preprocess_tusz.sbatch`

Output layout (scratch by convention):
```
<scratch>/data/chb_mit/              # CHB-MIT raw EDF
<scratch>/data/chb_mit_resampled/    # 200 Hz h5
<scratch>/eeg/tusz/v2.0.6/           # TUSZ raw EDF
<scratch>/eeg/tusz_resampled/        # 200 Hz h5
<scratch>/eeg/tusz_preproc/          # FFT 12s/60s clip h5
```

### 3) Train (current model)

```bash
# CHB-MIT, light_st_hyper, 100 epochs
sbatch -p gpu-l40s \
    --export=ALL,MODEL_NAME=light_st_hyper,TAG=run1,SEED=123 \
    sbatch/train_light_st_hyper.sbatch
```

Available `MODEL_NAME` values:
- `light_st_hyper` ← **current canonical**
- `light_dot`, `light_bilinear`, `light_attention` (LightEvoBrain edge variants)
- `light_dyn_hyper`, `light_static_hyper`
- `light_mamba_band_plv`, `light_attn_band_gated`
- `light_st_hyper_band`, `light_st_hyper_band_mamba`
- `light_st_hyper_norm`, `light_st_hyper_mscale`
- `ada_mshyper`, `st_hyper`, `mshyper`
- `evobrain` (original full model — use `sbatch/train_evobrain.sbatch`)
- `BIOT`, `dcrnn`, `evolvegcn`, `graphs4mer`, `gru_gcn`, `lstm`, `cnnlstm`

All overridable env vars:
- `MODEL_NAME` (default: `light_st_hyper`)
- `CLIP_LEN` (12 | 60, default 12)
- `NUM_EPOCHS` (default 100)
- `TRAIN_BS`, `TEST_BS` (default 128 / 256)
- `LR` (default 1e-4)
- `SEED` (default 123)
- `TAG` (run name suffix, default = `$MODEL_NAME`)
- `EVAL_EVERY` (default 5)
- `NUM_WORKERS` (DataLoader workers, default 4)

---

## Recommended Phoenix partition

`gpu-l40s` (48 GB VRAM, Ada sm_89) is the best balance of speed vs queue depth
vs cost for these models. cu121 env works on all partitions except
`gpu-rtxpro-blackwell` (sm_120 requires cu128+).

| Partition | VRAM | Compatible? |
|---|---|---|
| gpu-l40s | 48 GB | ✅ (recommended) |
| gpu-a100 | 80 GB | ✅ |
| gpu-h100 / gpu-h200 | 80 / 141 GB | ✅ |
| gpu-v100 | 32 GB | ✅ (slower, queue often long) |
| gpu-rtx6000 | 24 GB | ✅ (24 GB tight for full EvoBrain, queue often 200+) |
| gpu-rtxpro-blackwell | 96 GB | ❌ needs cu128 env |

---

## Path overrides

The sbatch scripts contain hardcoded paths under `/storage/scratch1/3/hkim3239/`
and `/storage/project/r-nimam6-0/hkim3239/`. To run on your own Phoenix
account:

1. Search-replace `hkim3239` and `r-nimam6-0` to match your account.
2. Or override at submit time: edit `REPO=`, `ENV_PATH=`, `SAVE_BASE=` near the
   top of each training sbatch.

The expected dir layout (mirrored from this repo) for friend's account:
```
~/r-nimam6-0/disease/                    # this repo
/storage/scratch1/<id>/.conda/envs/evobrain   # conda env (build once)
/storage/scratch1/<id>/data/chb_mit*          # CHB-MIT data
/storage/scratch1/<id>/eeg/tusz*              # TUSZ data
/storage/scratch1/<id>/eeg/runs/              # training outputs (heavy)
```

---

## Attribution

- `evobrain/` is forked from [Kotoge/EvoBrain](https://github.com/Kotoge/EvoBrain)
  (MIT license, see `evobrain/LICENSE`). New models and pipeline are our
  extension. Original git history backed up at `backup/.git_kotoge_evobrain_backup/`
  (not tracked in this repo).
- `external_baselines/Ada-MSHyper` and `MSHyper` are vendored from their
  respective upstreams (gitignored — clone separately if needed).
