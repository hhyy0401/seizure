# Seizure Detection — LightSTHyper

EEG seizure detection on TUSZ and CHB-MIT. Built on top of
[Kotoge/EvoBrain](https://github.com/Kotoge/EvoBrain) (NeurIPS 2025 Spotlight,
MIT license — `src/LICENSE`), with our main model **LightSTHyper**:
a learnable spatio-temporal hyperedge module on top of a uni-directional
Mamba backbone.

```
EEG (B, T, N, FFT)
  → uni-Mamba (2-layer, per channel)
  → + learnable node embedding
  → 2× SpatioTemporalHyperedgeBlock  (E_h soft hyperedges)
  → mean over time
  → PMA readout (Set Transformer seed query)
  → BCE classifier
```

Architecture + pseudocode: [docs/MODEL.md](docs/MODEL.md). Cluster
details: [docs/PHOENIX_SETUP.md](docs/PHOENIX_SETUP.md).

## Final config

Single hyperparameter setting (other than E_h) across all three benchmarks.

| Param | Value | CLI flag |
|---|---|---|
| Backbone | uni-Mamba, 2 layers (per channel) | `--model_name light_st_hyper --no_bidirectional` |
| d_model | 128 | `--rnn_units 128` |
| Hyperedges (E_h) | {1, 2, 3} swept | `--n_hyperedges {1,2,3}` |
| Hyperedge layers | 2 | `--n_hyper_layers 2` |
| Node embedding | on | `--use_node_emb` |
| Aux head | off | `--aux_type none` |
| Dropout | 0.0 | `--dropout 0.0` |
| Weight decay | 5e-4 | `--l2_wd 5e-4` |
| LR | 1e-3 (Adam) | `--lr_init 1e-3` |
| Input | FFT magnitudes, no fixed adjacency | `--use_fft --graph_type none` |
| Early-stop | dev AUROC, patience 10 (TUSZ) / 15 (CHB-MIT) | `--metric_name auroc` |
| Epoch cap | 80 (TUSZ 12s) / 100 (TUSZ 60s, CHB-MIT) | `--num_epochs ...` |
| Train batch | 128 (12s, CHB-MIT) / 32 (60s) | `--train_batch_size ...` |

Reporting protocol: test AUROC at dev-AUROC-best ckpt; F1@τ\* with
τ\* = F1-best τ on dev. Seeds {123, 456, 789}.

Numbers from the 27-run sweep are written to
[`FINAL_RESULTS.md`](FINAL_RESULTS.md) by the aggregator
(`sbatch/report/build_final_table.sbatch`).

## Repo layout

```
disease/
├── src/                            # All code
│   ├── main.py                     # Training entry
│   ├── args.py                     # CLI
│   ├── model/
│   │   ├── light_dyn_hyper.py      # ★ LightSTHyper (main)
│   │   ├── mamba_backbone.py       # Mamba backbone
│   │   ├── EvoBrain.py             # paper baseline
│   │   └── (other baselines: DCRNN, EGCN, BIOT, GraphS4mer, …)
│   ├── data/                       # Loaders + preproc + file_markers
│   └── scripts/
│       ├── build_final_table.py    # aggregate sweep → FINAL_RESULTS.md
│       ├── refresh_ckpts.py        # copy best run ckpts into ckpts/
│       ├── finalize_runs.py        # rebuild test_results.npz from ckpt
│       ├── dump_membership.py      # dump hyperedge M from ckpt
│       ├── viz_channel_focus.py    # channel topomaps
│       └── viz_node_emb.py         # t-SNE + cosine of node embeddings
├── sbatch/                         # SLURM scripts (see sbatch/README.md)
│   ├── setup/    env, data download, preproc
│   ├── train/    evobrain baseline + final_{tusz12,tusz60,chb12}
│   ├── report/   build_final_table, refresh_ckpts
│   └── figures/  dump_membership, viz_channel_focus, viz_node_emb
├── ckpts/                          # 3 final checkpoints (E_h=1, seed=123)
│   ├── tusz12/{best.pth.tar, args.json}
│   ├── tusz60/{best.pth.tar, args.json}
│   └── chb12/{best.pth.tar,  args.json}
├── docs/
│   ├── MODEL.md
│   └── PHOENIX_SETUP.md
├── figures/                        # Interpretability outputs
├── README.md  ·  FINAL_RESULTS.md  (auto-generated)
└── evobrain.pdf                    # paper
```

`args.json` records absolute scratch paths — update
`raw_data_dir / input_dir / preproc_dir` when reproducing elsewhere.

## Reproducing

```bash
# 0. Env
conda activate /storage/scratch1/3/hkim3239/.conda/envs/evobrain

# 1. Final sweep (27 runs: 3 datasets × E_h ∈ {1,2,3} × 3 seeds)
J1=$(sbatch --parsable sbatch/train/final_tusz12.sbatch)
J2=$(sbatch --parsable sbatch/train/final_tusz60.sbatch)
J3=$(sbatch --parsable sbatch/train/final_chb12.sbatch)
sbatch --dependency=afterany:$J1:$J2:$J3 sbatch/report/build_final_table.sbatch
# → FINAL_RESULTS.md

# 2. Paper EvoBrain baseline (CHB-MIT)
sbatch sbatch/train/evobrain.sbatch

# 3. Interpretability figures (uses ckpts/tusz12/)
sbatch sbatch/figures/dump_membership.sbatch     # → figures/membership_tusz12.npz
sbatch sbatch/figures/viz_channel_focus.sbatch   # → figures/channel/*.png
sbatch sbatch/figures/viz_node_emb.sbatch        # → figures/node_emb/*.png
```

Single-run override (TUSZ 12s, E_h=1, seed=123):

```bash
cd src/
python main.py --dataset TUSZ --task detection --model_name light_st_hyper \
    --num_nodes 19 --max_seq_len 12 --time_step_size 1 \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units 128 --n_hyper_layers 2 --n_hyperedges 1 \
    --aux_type none --dropout 0.0 --l2_wd 5e-4 \
    --num_epochs 80 --train_batch_size 128 --test_batch_size 256 \
    --lr_init 1e-3 --patience 10 --eval_every 1 --metric_name auroc \
    --rand_seed 123 --data_augment \
    --raw_data_dir /path/to/tusz/edf \
    --input_dir /path/to/tusz_resampled \
    --preproc_dir /path/to/tusz_preproc/clipLen12_timeStepSize1 \
    --save_dir /path/to/out/run1
```

## Datasets

- **TUSZ v2.0.6** detection (official patient-level split): 12 s and 60 s clips
- **CHB-MIT** detection (paper protocol — same-patient random 15 % test split)

Train uses balanced 1:1 sz:nosz subsampling. Dev/test use full sets.

## License

Code: MIT (inherited from EvoBrain).
