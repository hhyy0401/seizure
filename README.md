# Seizure Detection — LightSTHyper

EEG seizure detection on TUSZ and CHB-MIT. Built on top of
[Kotoge/EvoBrain](https://github.com/Kotoge/EvoBrain) (NeurIPS 2025 Spotlight,
MIT license — `src/LICENSE`), with our main model **LightSTHyper** that
replaces the paper's dynamic-adjacency graph with a learnable spatio-temporal
hyperedge module and a bi-directional Mamba backbone.

## Main model — LightSTHyper

```
EEG (B, T, N, FFT)
  → BiMamba (forward + reverse, 2-layer, per channel)
  → + learnable node embedding
  → 2× SpatioTemporalHyperedgeBlock  (E_h soft hyperedges, swept)
  → mean over time
  → PMA readout (Set Transformer seed query)
  → BCE classifier
```

Aux head: per-edge BCE deep supervision on the last hyperedge block
(`--aux_type bce --aux_weight 0.3`).

Full architecture + pseudocode: [docs/MODEL.md](docs/MODEL.md).

## What's different from paper EvoBrain

| Aspect | Paper EvoBrain | Ours |
|---|---|---|
| Temporal | Mamba (uni-directional) | **Bi-Mamba** |
| Spatial | Dynamic xcorr graph + top-k | **Learnable hypergraph** |
| Readout | DCRNN diffusion + max pool | PMA (Set Transformer) |
| Deep supervision | none | Optional per-edge BCE |
| 1 epoch on TUSZ 12s | ~5–10 min | **~1.5 min** |

Paper EvoBrain reproduction is included (`src/model/EvoBrain.py`) for
direct comparison.

## Repo layout

```
disease/
├── src/                            # All code (renamed from evobrain/)
│   ├── main.py                     # Training entry point
│   ├── args.py                     # CLI (ablation switches preserved)
│   ├── model/
│   │   ├── light_dyn_hyper.py      # ★ LightSTHyper (main)
│   │   ├── mamba_backbone.py       # BiMamba backbone
│   │   ├── temporal_backbones.py   # dwsep backbone (ablation)
│   │   ├── EvoBrain.py             # paper baseline
│   │   ├── DCRNN.py, gru_gcn.py, EGCN.py, …
│   │   └── BIOT.py, lstm.py, cnnlstm.py, graphs4mer.py
│   ├── data/                       # Loaders, preproc, file_markers
│   │   ├── dataloader_detection.py # TUSZ
│   │   ├── dataloader_chb.py       # CHB-MIT
│   │   └── build_file_markers_*.py
│   └── scripts/
│       ├── aggregate_grid.py       # paper-standard F1@τ* aggregation
│       ├── finalize_runs.py        # rebuild test_results.npz from ckpt
│       └── build_results_table.py  # cross-dataset comparison table
├── sbatch/                         # SLURM scripts for Phoenix
│   ├── sweep_main.sbatch           # 36-job grid (E × d × lr)
│   ├── sweep_reg.sbatch            # 9-job reg sweep (dropout × wd)
│   ├── train_light_st_hyper.sbatch
│   ├── train_evobrain.sbatch       # paper reproduction
│   └── …
├── docs/
│   ├── MODEL.md                    # architecture + pseudocode
│   └── PHOENIX_SETUP.md            # cluster config, batch sizes, paths
├── README.md
├── FRIEND_NOTE.md                  # status + paper-vs-ours table
├── evobrain.pdf                    # paper
└── external_baselines/             # third-party (gitignored)
```

## Quick start

```bash
# 1. Activate env (Phoenix or compatible)
conda activate /storage/scratch1/3/hkim3239/.conda/envs/evobrain

# 2. Train main model (TUSZ 12s)
cd src/
python main.py \
    --dataset TUSZ --task detection \
    --model_name light_st_hyper \
    --aux_type bce --aux_weight 0.3 \
    --n_hyperedges 3 --use_node_emb --bidirectional \
    --rnn_units 64 --dropout 0.0 --l2_wd 5e-4 \
    --lr_init 3e-4 --num_epochs 80 --patience 10 \
    --train_batch_size 128 --test_batch_size 256 \
    --eval_every 1 --metric_name auroc \
    --raw_data_dir /path/to/tusz/edf \
    --input_dir /path/to/tusz_resampled \
    --preproc_dir /path/to/tusz_preproc/clipLen12_timeStepSize1 \
    --save_dir /path/to/out/run1 --data_augment

# 3. Or via SLURM
sbatch sbatch/train_light_st_hyper.sbatch
```

## Paper-standard reporting

Always report:
1. Test AUROC at the **dev-AUROC-best checkpoint**
2. F1 at **τ\* = F1-best τ tuned on dev**, applied to test

Aggregate after training:
```bash
python src/scripts/build_results_table.py
```

`main.py` automatically dumps `dev_results.npz` / `test_results.npz` from
the best ckpt on natural termination (early stop or end of epochs).
`scripts/finalize_runs.py` is the fallback for runs killed by walltime.

## Datasets

- **TUSZ v2.0.6** detection (official patient-level split): 12s + 60s clips
- **CHB-MIT** detection (paper protocol — same-patient random 15% test
  split; see [docs/MODEL.md](docs/MODEL.md))

Train uses balanced 1:1 sz:nosz subsampling. Dev/test use full sets.

## Setup

See [docs/PHOENIX_SETUP.md](docs/PHOENIX_SETUP.md) for cluster details,
batch-size guidance, and the Blackwell variant.

## License

Code: MIT (inherited from EvoBrain).
