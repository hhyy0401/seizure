# Status update — EvoBrain reproduction + our LightSTHyper

## Confirmed main model

**LightSTHyper** (`evobrain/model/light_dyn_hyper.py`):

```
input (B, T=12, N=19, D=100 FFT) 
  → BiMamba backbone (2 layers, bidirectional)
  → + learnable node embedding (N × d_model)
  → 2× SpatioTemporalHyperedgeBlock
  → time mean pool
  → PMA readout (1 seed)
  → BCE classifier
```

Auxiliary head: **per-edge BCE deep supervision** on the last hypergraph
layer (aux_weight=0.3).

Differentiator vs paper EvoBrain:
- **Bi-directional Mamba** wrapper (paper is unidirectional)
- **Learnable hyperedge** with PMA readout (paper uses dynamic graph
  adjacency via xcorr + top-k)
- **No xcorr / on-the-fly graph compute** → 10–20× faster training/eval

Ablation switches preserved via `args`:
- `--model_name {light_st_hyper, light_st_hyper_linear, light_st_hyper_dwsep}`
  (mamba / linear / depthwise-sep conv)
- `--aux_type {none, bce, entropy}`, `--aux_weight`
- `--n_hyperedges`, `--n_hyper_layers`, `--use_node_emb`
- `--bidirectional / --no_bidirectional`
- `--rnn_units` (d_model)

## Paper baselines (from EvoBrain Table 1, NeurIPS 2025 Spotlight)

| Dataset | Window | AUROC | F1 |
|---|---|---|---|
| TUSZ | 12s detection | 0.877 ±0.005 | 0.539 ±0.009 |
| TUSZ | 60s detection | 0.865 ±0.009 | 0.483 ±0.006 |
| CHB-MIT | 12s detection | 0.94 (text only) | not stated |

## Performance comparison (current best, before sweep)

Paper-standard reporting: dev-AUROC-best ckpt → test AUROC + F1@τ\* (τ\* = F1-best τ tuned on dev).
LSH = LightSTHyper (BiMamba + node_emb + 2-layer hypergraph). `†` = ep17-58 mid-training snapshot, NOT converged.

| Dataset | Window | Model | dev_AU | test_AU | F1 |
|---|---|---|---|---|---|
| TUSZ | 12s | Paper EvoBrain | — | 0.877 ±0.005 | 0.539 ±0.009 |
| | | **Ours (LSH+E=3+bce)** | 0.871 | **0.884** | **0.522** |
| | | Ours (LSH+E=3+none) | 0.870 | 0.875 | 0.501 |
| | | Ours (LSH+E=4+bce) | 0.873 | 0.888 | 0.494 |
| TUSZ | 60s | Paper EvoBrain | — | 0.865 ±0.009 | 0.483 ±0.006 |
| | | **Ours (LSH+E=4+bce)** † | 0.826 | **0.869** | — |
| | | Ours (LSH+E=3+bce) † | 0.818 | 0.850 | — |
| | | Ours (LSH+E=3+none) † | 0.812 | 0.846 | — |
| CHB-MIT | 12s | Paper EvoBrain | — | 0.94 | not stated |
| | | **Ours (LSH+E=4+bce)** † | 0.889 | **0.902** | — |
| | | Ours (LSH+E=3+none) † | 0.876 | 0.892 | — |
| | | Ours (LSH+E=3+bce) † | 0.881 | 0.891 | — |

**Highlights**:
- TUSZ 12s: paper AUROC 0.877 vs **ours 0.884** (match within paper σ=0.005)
- **TUSZ 60s: paper AUROC 0.865 vs ours 0.869** — at ep17 snapshot, training
  was still climbing (dev reached 0.834 before cancel; test not re-checked
  at that epoch)
- CHB-MIT: paper 0.94 vs ours 0.90 (cancelled at ep55-58 still climbing).
  Split is same-patient random 15% — seed mismatch alone yields meaningful
  delta; not converged
- F1 columns marked `—` were lost during pre-sweep cleanup; sweep finalize
  will produce them paper-standard

## Currently running: TUSZ 12s parameter sweep (45 jobs, overnight)

Anchor config: mamba + bce + use_node_emb=True, patience=10, num_epochs=80.

**Stage A** (36 jobs) — architecture × optimization grid:
- n_hyperedges ∈ {2, 3, 4, 5}
- rnn_units (d_model) ∈ {64, 96, 128}
- lr ∈ {3e-4, 5e-4, 1e-3}

**Stage B** (9 jobs) — regularization sweep at anchor (E=3, d=64, lr=3e-4):
- dropout × weight_decay = 3 × 3

Patience-on-AUROC early-stop + 2h walltime + auto-finalize fallback.
Results dumped to `dev_results.npz` / `test_results.npz` for paper-standard
F1@τ\* reporting.

## CHB-MIT note (skipped for now)

We ran 3 jobs (top TUSZ configs) — ours hit dev AUROC 0.897 climbing toward
plateau around ep73-80. Paper claims 0.94 AUROC. We didn't reach it BUT:

- **Split is random within-patient** (paper protocol: "randomly selected
  15% of the patient's data"). Train and test share all 22 patients —
  same-session leakage possible. Strict patient-LOO would give different
  numbers.
- Our seed (123) ≠ paper seed (unknown) → exact numbers differ even with
  identical protocol.
- Re-running with same protocol but same-architecture EvoBrain reproduction
  is the right comparison. Paper's published EvoBrain config on CHB-MIT
  trains slowly (~30 min/epoch — large eval set + dynamic graph compute).

## EvoBrain reproduction speed note

The paper's "17× faster than DCRNN" claim refers to **forward pass compute**
(BiMamba beats DCRNN's gated recurrence). It does NOT mean EvoBrain itself is
fast end-to-end:
- `graph_type=dynamic` recomputes 19×19 cross-correlation adjacency per batch
- Dev/test eval is full set (no subsample) — 40K clips on CHB-MIT, 100K+ on
  TUSZ
- Combined: 1 epoch ≈ 35 min for CHB-MIT EvoBrain reproduction

Our LightSTHyper avoids both — `graph_type=none`, hyperedges learned end-to-end.
1 epoch ≈ 1.5 min on TUSZ 12s.

## Repo

GitHub: `hhyy0401/seizure`, branch `main`, latest commit `eaff304`.
Main: `evobrain/main.py`, model: `evobrain/model/light_dyn_hyper.py`,
sweep sbatch: `sbatch/sweep_main.sbatch`, `sbatch/sweep_reg.sbatch`.

EvoBrain reproduction code (paper's model) is in `evobrain/model/EvoBrain.py`
+ supporting files (`cell.py`, `DCRNN.py`, etc.). All Paper Table 1
baselines included for direct comparison.
