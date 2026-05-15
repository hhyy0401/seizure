# CHB-MIT Sweep Plan — Beat EvoBrain

## Target
Beat EvoBrain (Kotoge et al., NeurIPS 2025 Spotlight) on CHB-MIT seizure detection
(22-channel bipolar, 12s clips, AUROC/AUPRC primary metrics).

## Final variant set (5 runs)

| # | model_name           | Role                          | Pseudocode                          |
|---|----------------------|-------------------------------|-------------------------------------|
| 1 | `evobrain`           | Baseline (original)           | (existing `model/EvoBrain.py`)      |
| 2 | `light_attention`    | Light baseline (best of 3)    | (existing `model/LightEvoBrain.py`) |
| 3 | `light_dyn_hyper`    | **NOVEL headline**            | `light_dyn_hyper.md`                |
| 4 | `light_static_hyper` | Ablation for C+               | (section in `light_dyn_hyper.md`)   |
| 5 | `light_mamba_band_plv` | **NOVEL secondary**         | `light_mamba_band_plv.md`           |

**모델 비교 한 페이지**: [sweep_models.md](sweep_models.md)

**공통 Mamba 백본**: `model/mamba_backbone.py` — Bi-Mamba(양방향) 적용. 단,
baseline 2종(`evobrain`, `light_attention`)은 fairness 위해 원본 그대로 단방향.

## 실행

```bash
cd /home/hkim3239/eeg/evobrain
bash sbatch/sweep_chb.sh                              # 5개 모두 (50 epoch each)
NUM_EPOCHS=10 bash sbatch/sweep_chb.sh                # 빠르게 검증
MODELS="light_dyn_hyper" bash sbatch/sweep_chb.sh     # 한 개만 재실행
```

진행 로그: `logs/<model>_<timestamp>.log`, 결과: `/home/hkim3239/eeg/runs/`

## Variants explicitly DROPPED (duplicate to prior art)

| Idea                | Rejected by                                              |
|---------------------|----------------------------------------------------------|
| GATv2 + FC prior    | DTS-GAN (Sci Rep 2025); GAT-Epilepsy (arXiv 2507.15118)  |
| Multi-view fusion   | MGCNA (Frontiers 2024); Hypercube-S4GNN (IEEE 2024)      |
| Anatomical/LapPE    | EvoBrain (LapPE on dist); NeuroGNN (10-20 coords)        |
| Static band-PLV     | Multi-Band GNN Neonatal (Appl Sci 2024); STGAT-PLV       |

## Story for the main result table

- **EvoBrain vs Light-attn**: Is the heavy edge-stream Mamba necessary?
- **Light-attn vs C+**: Does higher-order (hypergraph) > pairwise?
- **C+ vs C-ab**: Does Mamba-state-driven dynamic evolution > static learned queries?
- **C+ vs E+**: Higher-order channel synchrony vs frequency-band adaptation — which captures seizure dynamics better?
- **E+ vs Light-attn**: Does Mamba-state-gated band fusion > single learned attention?

## Time/cost budget
- 5 runs × ~3.5h (50 epochs, 12s clip, batch 128) ≈ 18 GPU-hours
- g4-standard-48 @ $2.85/hr → **~$50** total

## Common setup (all 5)
- Dataset: CHB-MIT, 22 bipolar channels, 200Hz
- Clip: 12s, time_step_size=1s
- Patient set: chb01–chb22 (655 EDFs found, 624 successfully preprocessed)
- Split: same as `data/file_markers_chb/*.txt` (regenerated to drop 31 failed files)
- Optimizer/LR/schedule: same as `sbatch/train_chb.sbatch` defaults

## Implementation phases (in order)

1. **`model/fc_compute.py`** — bandpass / Hilbert / PLV on GPU (shared by E+)
2. **`model/light_dyn_hyper.py`** — C+ (and C-ab as a flag)
3. **`model/light_mamba_band_plv.py`** — E+
4. **`args.py`** — add `model_name` choices
5. **`main.py`** — model dispatch
6. **`data/build_file_markers_chb.py`** — regenerate markers (drop 31 failed files)
7. **`sbatch/sweep_chb.sh`** — sequential driver over 5 models
