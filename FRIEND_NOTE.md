# LightSTHyper — final results & how to reproduce

## Results — 3-seed mean ± std vs paper EvoBrain

| Dataset | Window | Paper AUROC | **Ours AUROC** | Paper F1 | **Ours F1** |
|---|---|---|---|---|---|
| TUSZ | 12s | 0.877 ±0.005 | **0.884 ±0.007** | 0.539 ±0.009 | 0.466 ±0.084 |
| TUSZ | 60s | 0.865 ±0.009 | **0.878 ±0.009** | 0.483 ±0.006 | **0.531 ±0.050** |
| CHB-MIT | 12s | 0.940 | 0.905 ±0.001 | — | 0.158 ±0.013 |

Reporting protocol: test AUROC at dev-AUROC-best ckpt; F1@τ\* with
τ\* = F1-best τ on dev. Seeds {123, 456, 789}. **Bold** beats paper σ band.

- AUROC: 2 / 3 beat paper, CHB-MIT 0.035 below.
- F1: TUSZ 60s beats; TUSZ 12s has high seed variance (see note); CHB-MIT
  F1 calibration breaks under within-patient split (AUROC is the meaningful
  number there).

## Final config (`E=3, d=128, lr=1e-3`, single setting across all 3 datasets)

| Param | Value | CLI flag |
|---|---|---|
| Backbone | BiMamba, 2 layers | `--model_name light_st_hyper --bidirectional` |
| d_model | 128 | `--rnn_units 128` |
| Hyperedges (E_h) | 3 | `--n_hyperedges 3` |
| Hyperedge layers | 2 | `--n_hyper_layers 2` |
| Node embedding | on | `--use_node_emb` |
| Aux head | per-edge BCE, w=0.3 | `--aux_type bce --aux_weight 0.3` |
| Dropout | 0.0 | `--dropout 0.0` |
| Weight decay | 5e-4 | `--l2_wd 5e-4` |
| LR | 1e-3 (Adam) | `--lr_init 1e-3` |
| Input | FFT magnitudes, learned graph | `--use_fft --graph_type none` |
| Early-stop | dev AUROC, patience 10 (TUSZ) / 15 (CHB) | `--metric_name auroc` |
| Epoch cap | 80 (12s) / 100 (60s, CHB) | `--num_epochs 80` |
| Train BS | 128 (12s, CHB) / 32 (60s) | `--train_batch_size 128` |

Architecture + pseudocode: [`docs/MODEL.md`](docs/MODEL.md). Cluster /
batch-size details: [`docs/PHOENIX_SETUP.md`](docs/PHOENIX_SETUP.md).

## How we picked it (search procedure)

1. **Coarse grid** on TUSZ 12s (`sbatch/sweep_main.sbatch`, 36 jobs):
   E_h ∈ {2,3,4,5} × d_model ∈ {64,96,128} × lr ∈ {3e-4, 5e-4, 1e-3}.
2. **Regularization** at the best anchor (`sbatch/sweep_reg.sbatch`,
   9 jobs): dropout × l2_wd grid → confirmed dropout=0, wd=5e-4 best.
3. **Multi-seed validation** of top-3 distinct cfgs on all 3 datasets
   (`sbatch/multiseed_*.sbatch`, 27 jobs, seeds {123, 456, 789}) →
   E=3 d=128 lr=1e-3 is top-1 or near-top-1 on each dataset, so adopt
   as final unified config.

## TUSZ 12s F1 variance note

At fixed cfg, F1 across seeds: 0.396 / 0.442 / 0.559 (mean 0.466, σ 0.084).
AUROC moves only 0.007 — the model's ranking is robust. F1 swings come
from argmax-τ\* on a plateau-flat dev PR curve; small τ shifts amplify
into ≥0.1 F1 swings on the ~5%-pos test set. Paper's F1 σ=0.009 likely
uses a smoother threshold-selection convention. Fix candidates (no
retraining needed, re-aggregates from existing npz): median-over-plateau,
smoothed argmax, or bootstrap-stabilized τ\*.

## Checkpoints

Best-dev-AUROC checkpoints (seed=123, the cfg above) are committed:

```
ckpts/tusz12/{best.pth.tar, args.json}     # TUSZ 12s, dev AU 0.875, test AU 0.891
ckpts/tusz60/{best.pth.tar, args.json}     # TUSZ 60s, dev AU 0.866, test AU 0.871
ckpts/chb12/{best.pth.tar, args.json}      # CHB-MIT 12s, test AU 0.904
```

`args.json` has absolute paths to our scratch — update `raw_data_dir /
input_dir / preproc_dir` if reproducing on a different filesystem.

## Reproducing

**Eval from shipped ckpts** (fastest, ~5–15 min each on rtx6000):

```bash
sbatch --exclude=atl1-1-03-007-1-0 sbatch/eval_ckpts.sbatch
# array 0-2 → dumps {dev,test}_results.npz alongside each ckpt
python src/scripts/aggregate_multiseed.py
```

**Full retrain** (27 runs, ~6h wall total):

```bash
J1=$(sbatch --parsable --exclude=atl1-1-03-007-1-0 sbatch/multiseed_tusz12.sbatch)
J2=$(sbatch --parsable --exclude=atl1-1-03-007-1-0 sbatch/multiseed_tusz60.sbatch)
J3=$(sbatch --parsable --exclude=atl1-1-03-007-1-0 sbatch/multiseed_chb12.sbatch)
sbatch --dependency=afterany:$J1:$J2:$J3 sbatch/aggregate_multiseed.sbatch
```

GitHub: `hhyy0401/seizure`, branch `main`.
