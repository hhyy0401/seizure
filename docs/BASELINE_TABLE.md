# Baseline table — fill-in tracking

> Auto-updated by hourly monitor. dev-AUROC-best ckpt → test AUROC + F1 at τ\*
> (τ\* = F1-best on dev). Bold = filled, *(n=k)* = seeds aggregated so far.

## Last updated


2026-05-18 22:39 (local)

## Source of seeds

- TUSZ 12s / 60s: this Phoenix project, runs under `/storage/scratch1/3/hkim3239/eeg/runs/`
- CHB-MIT 12s **s123** for LSTM / BIOT / DCRNN: ran elsewhere by user — values pasted manually below (do NOT overwrite from auto-pull)
- CHB-MIT 12s s456 / s789: this Phoenix project

## CHB-MIT 12s — externally-supplied s123 (do not overwrite)

| Model | AUROC (s123) | F1 (s123) | Source |
|---|---|---|---|
| LSTM  | 0.884 | 0.061 | external machine, reported by user 2026-05-18 |
| BIOT  | 0.904 | 0.000 | external machine, reported by user 2026-05-18 |
| DCRNN | 0.883 | 0.138 | external machine, reported by user 2026-05-18 |

## Current table (TUSZ + CHB-MIT)



| Method | TUSZ 12s AUROC | TUSZ 12s F1 | TUSZ 60s AUROC | TUSZ 60s F1 | CHB-MIT 12s AUROC | CHB-MIT 12s F1 |
|---|---|---|---|---|---|---|
| LSTM       | 0.839±0.016 *(n=3)* | 0.403±0.065 *(n=3)* | — | — | 0.823±0.075 *(n=3, w/ ext s123)* | 0.064±0.007 *(n=3, w/ ext s123)* |
| CNN-LSTM   | 0.819±0.017 *(n=2)* | 0.395±0.001 *(n=2)* | — | — | — | — |
| BIOT       | — | — | — | — | 0.903±0.018 *(n=3, w/ ext s123)* | 0.054±0.094 *(n=3, w/ ext s123)* |
| LaBraM | — | — | — | — | — | — |
| EEGPT | — | — | — | — | — | — |
| EvolveGCN  | 0.812±0.006 *(n=3)* | 0.356±0.018 *(n=3)* | 0.752±0.010 *(n=3)* | 0.333±0.025 *(n=3)* | — | — |
| DCRNN      | — | — | — | — | 0.877±0.006 *(n=3, w/ ext s123)* | 0.125±0.018 *(n=3, w/ ext s123)* |
| GRAPHS4MER | 0.884±0.002 *(n=3)* | 0.461±0.001 *(n=3)* | 0.856±0.017 *(n=3)* | 0.474±0.046 *(n=3)* | 0.899±0.015 *(n=3)* | 0.185±0.046 *(n=3)* |
| GRU-GCN    | — | — | — | — | — | — |
| EvoBrain | — | — | — | — | — | — |
| **LightSTHyper (E_h=1)** | **0.898±0.003** | **0.519±0.019** | **0.877±0.017** | **0.569±0.017** | 0.898±0.006 | 0.142±0.007 |
| **LightSTHyper (E_h=2)** | 0.892±0.006 | 0.440±0.023 | 0.848±0.019 | 0.463±0.040 | 0.898±0.006 | **0.154±0.025** |
| **LightSTHyper (E_h=3)** | 0.889±0.006 | 0.504±0.029 | **0.877±0.005** | 0.537±0.039 | **0.904±0.007** | 0.134±0.021 |

## Aggregated CHB-MIT 12s detail



| Model | s123 (ext) | s456 | s789 | mean±std |
|---|---|---|---|---|
| LSTM  | 0.884 / 0.061 | 0.845 / 0.072 | 0.739 / 0.059 | **0.823±0.075 / 0.064±0.007** |
| BIOT  | 0.904 / 0.000 | 0.885 / 0.000 | 0.921 / 0.162 | **0.903±0.018 / 0.054±0.094** |
| DCRNN  | 0.883 / 0.138 | 0.877 / 0.132 | 0.872 / 0.105 | **0.877±0.006 / 0.125±0.018** |

## Known mismatches with prior table (need user resolution)

- EvolveGCN TUSZ 12s: prior 0.757±0.004 vs new 0.812±0.006
- EvolveGCN TUSZ 60s: prior 0.670±0.017 vs new 0.752±0.010
- GRAPHS4MER TUSZ 12s: prior 0.833±0.005 vs new 0.884±0.002
- GRAPHS4MER TUSZ 60s: prior 0.778±0.021 vs new 0.856±0.017

The prior numbers are kept in the rendered figure; the new numbers above come from
the latest Phoenix runs (job IDs 8531xxx, 8539xxx — see `/storage/scratch1/3/hkim3239/eeg/runs/`).

## Job board (snapshot, auto-updated)



### Running
- `8605394` upd_basetbl (0:04)
- `8605341` regen_evobrain (0:41)
- `8583940_2` base_chb (3:41:47)
- `8583940_1` base_chb (3:44:23)
- `8583940_0` base_chb (3:44:35)
- `8605124_1` base_fnd (1:23)
- `8605124_0` base_fnd (2:59)

### Pending
- `8605125_[0-2]` base_fnd (0:00)
- `8605126_[0-2]` base_fnd (0:00)
- `8605132_[0-2]` base_fnd (0:00)
- `8605136_[0-2]` base_fnd (0:00)
- `8605140_[0-2]` base_fnd (0:00)
- `8605124_[2]` base_fnd (0:00)

## Not started (manual, no sbatch yet)

- LaBraM TUSZ/CHB
- EEGPT TUSZ/CHB
- EvoBrain TUSZ/CHB
- GRU-GCN CHB-MIT s789 (only s123/s456 running)
