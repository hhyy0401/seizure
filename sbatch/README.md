# SLURM scripts — execution order

```
setup/    → data + env  (run once)
train/    → baseline + final sweep
report/   → aggregate sweep into table + refresh shipped checkpoints
figures/  → interpretability plots from the shipped checkpoint
```

All scripts assume Phoenix paths under `/storage/scratch1/3/hkim3239/`.
Edit the top of each file (env path, data dirs) for a different filesystem.

## 1. setup/ — env + data (one-time)

| script | output |
|---|---|
| `env_evobrain.sbatch` | conda env at `.conda/envs/evobrain` |
| `download_chb_mit.sbatch` | raw CHB-MIT EDFs |
| `preprocess_chb.sbatch` | resampled HDF5 cache |
| `preprocess_tusz.sbatch` | TUSZ resample + FFT cache (serial, 24 h) |
| `parallel_tusz_resample.sbatch` | TUSZ resample only, 8-way array (faster) |
| `parallel_tusz_preproc12.sbatch` | TUSZ 12 s FFT clips, 8-way array |

Recommended order: env → both raw data → preproc. For TUSZ, prefer the
two `parallel_*` arrays over the serial `preprocess_tusz.sbatch`.

## 2. train/ — baseline + final sweep

| script | jobs | what it trains |
|---|---|---|
| `evobrain.sbatch` | 1 | Paper EvoBrain on CHB-MIT (override `--export=ALL,CLIP_LEN=60,…` for other settings) |
| `final_tusz12.sbatch` | array 0–8 | LightSTHyper uni-Mamba on TUSZ 12 s × E_h ∈ {1, 2, 3} × 3 seeds |
| `final_tusz60.sbatch` | array 0–8 | same on TUSZ 60 s |
| `final_chb12.sbatch` | array 0–8 | same on CHB-MIT 12 s |

Final-sweep launch:

```bash
J1=$(sbatch --parsable sbatch/train/final_tusz12.sbatch)
J2=$(sbatch --parsable sbatch/train/final_tusz60.sbatch)
J3=$(sbatch --parsable sbatch/train/final_chb12.sbatch)
sbatch --dependency=afterany:$J1:$J2:$J3 sbatch/report/build_final_table.sbatch
```

## 3. report/ — aggregate + refresh shipped checkpoints

| script | output |
|---|---|
| `build_final_table.sbatch` | `FINAL_RESULTS.md` (paper EvoBrain + ours × {E_h=1,2,3} on 3 datasets) |
| `refresh_ckpts.sbatch` | copies `(E_h=1, seed=123)` best ckpts into `ckpts/{tusz12,tusz60,chb12}/`. Override `E_H` / `SEED` via `--export`. |

## 4. figures/ — interpretability plots (uses shipped TUSZ 12 s ckpt)

| script | output |
|---|---|
| `dump_membership.sbatch` | `figures/membership_tusz12.npz` (hyperedge soft membership) |
| `viz_channel_focus.sbatch` | `figures/channel/*.png` (scalp topomap, bar, temporal) |
| `viz_node_emb.sbatch` | `figures/node_emb/*.png` (t-SNE, cosine) |

```bash
JD=$(sbatch --parsable sbatch/figures/dump_membership.sbatch)
sbatch --dependency=afterok:$JD sbatch/figures/viz_channel_focus.sbatch
sbatch                          sbatch/figures/viz_node_emb.sbatch
```

`viz_node_emb` reads `ckpts/tusz12/best.pth.tar` directly (no dump
needed); the other two need `membership_tusz12.npz` first.
