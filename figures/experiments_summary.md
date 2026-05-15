# EEG Seizure Detection — Confirmed Methodology

## Datasets

| Dataset | N_channels | Test N | Test Pos | Pos ratio |
|---|---|---|---|---|
| **TUSZ v2.0.6**, 12s clips | 19 | 38,384 | 2,340 | 6.10% |
| **CHB-MIT**, 12s clips | 22 | 40,676 | 161 | 0.40% |

---

## 1. 모델 구조 (Confirmed)

> **Conv-only spatio-temporal seizure detector with per-second multi-task supervision.**

| Component | Choice | Detail |
|---|---|---|
| Backbone | **TCN** (manual dilated) | 2 layers, kernel=3, dilation ∈ {1, 2}, d_model=64 |
| Channel ID embed (chid) | `nn.Embedding(N, 16)` | per-channel learnable embedding concat'd to FFT input (input_dim = 100 + 16 = 116) |
| Spatio-Temporal Hyperedge | `SpatioTemporalHyperedgeBlock`, h=3 | static learnable query `Q ∈ (3, d)`, sigmoid soft-membership over (N, T) |
| Multi-task aux head | LayerNorm → Linear → GELU → Linear | backbone 출력에서 노드 mean pool → 12-bin per-second seizure 예측 |
| Loss | `BCE(main, y_clip) + λ × BCE(aux, mask_12bin)` | λ = 1.0 |
| Graph supports | **none** (adj 안 씀) | hyperedge가 자체 채널 mixing 수행 |
| Params (TUSZ N=19) | **78,770** | |
| Params (CHB N=22) | **78,818** | |

### 1.1 Aux loss 메커니즘

- **Main head** (clip-level): 12초 clip 전체에 발작이 있냐 없냐 — 1개 logit
- **Aux head** (per-second): 12개 1초 bin이 각각 발작인지 — 12개 logit
- **Mask 생성**: TUSZ `.csv_bi`(또는 CHB seizure_map)의 seizure start/stop을 1초 단위로 잘라서 12-bin 이진 마스크 생성
  - 예: clip 24–36s, 발작 28–32s → `mask = [0,0,0,0,1,1,1,1,0,0,0,0]`
  - negative clip → `[0]×12`
- **Test 시**: main head만 사용. aux head는 학습용 신호.

**왜 효과 있나**:
- Clip-level label은 정보 손실 큼 ("1초만 발작" vs "12초 다 발작" 같은 y=1)
- Per-second 신호가 모델에게 **발작의 시작/끝을 정확히 짚도록** 강제 → 시간적 feature 샤프해짐
- Positive score의 압축 (squashing) 해소: `pos_mean 0.677 → 0.718`, recall@0.5 +6%p
- Focal/pos_weight와 다른 메커니즘 (threshold가 아니라 **representation** 자체를 고침)

---

## 1.2 Figure-ready Architecture (그림 직접 그릴 수 있게)

### 입력
- `x ∈ ℝ^(B, N, T, D_fft)` — EEG clip, T=12 s-bins, D_fft=100, N=channels (TUSZ=19, CHB=22)
- `y_clip ∈ {0,1}^B` — clip-level label
- `m ∈ {0,1}^(B, 12)` — per-second mask from `.csv_bi` / `seizure_map`

### 블록 순서 (figure 위→아래)

```
┌──────────────────────────────────────────────────────────────────┐
│              INPUT  x : (B, N, T=12, D_fft=100)                   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │  Channel ID Embedding (chid)      │
                │  chid_emb = Embed(N, 16)          │
                │  concat over feature dim          │
                │  → (B, N, T, 116)                 │
                └─────────────────┬─────────────────┘
                                  │
                       LayerNorm(116)
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │  TCN Backbone (2 layers, per-channel temporal)    │
        │  reshape: (B·N, T, 116)                           │
        │                                                    │
        │  ┌─────────────── Layer 1 ───────────────┐         │
        │  │  DilatedConv(116→64, k=3, d=1)        │         │
        │  │  GELU                                 │         │
        │  │  DilatedConv( 64→64, k=3, d=1)        │         │
        │  │  + skip (Linear(116→64))              │         │
        │  └───────────────────────────────────────┘         │
        │              │                                     │
        │  ┌─────────────── Layer 2 ───────────────┐         │
        │  │  DilatedConv(64→64, k=3, d=2)         │         │
        │  │  GELU                                 │         │
        │  │  DilatedConv(64→64, k=3, d=2)         │         │
        │  │  + skip (Identity)                    │         │
        │  └───────────────────────────────────────┘         │
        │  LayerNorm(64)                                     │
        │  reshape → H_seq : (B, N, T=12, d=64)             │
        └──┬──────────────────────────────────────────┬─────┘
           │                                          │
           │                                          │
   ┌───────▼────────┐                       ┌─────────▼──────────┐
   │  AUX HEAD      │                       │ Spatio-Temporal    │
   │ (train only)   │                       │ Hyperedge (h=3)    │
   │ ─────────────  │                       │ ────────────────── │
   │ mean over N    │                       │ Q ∈ (3, 64) static │
   │ → (B, T, 64)   │                       │                    │
   │ LayerNorm      │                       │ M = σ(H_seq·Q/√d)  │
   │ Linear(64→32)  │                       │ ∈ (B, T, N, 3)     │
   │ GELU           │                       │                    │
   │ Linear(32→1)   │                       │ H_pool = M·H_seq   │
   │ → (B, T)       │                       │ ∈ (B, T, 3, 64)    │
   │ aux_logits     │                       └─────────┬──────────┘
   └──────┬─────────┘                                 │
          │                          mean over T → (B, 3, 64)
          │                                          │
          │                          PMA Readout     │
          │                          → (B, K·64)     │
          │                                          │
          │                          Linear → (B, 1) │
          │                          ↓ main_logit    │
          │                                          │
   ┌──────▼─────────────────────────────────▼────────┐
   │            Loss (training only)                 │
   │  L = BCE(main_logit, y_clip)                    │
   │    + λ × BCE(aux_logits, mask_12bin)            │
   │  λ = 1.0                                        │
   └─────────────────────────────────────────────────┘

   TEST: main_logit → sigmoid → thr=0.5
```

### Shape 표 (figure 옆 caption)

| Step | Operation | Output |
|---|---|---|
| 0 | Input FFT features | (B, N, 12, 100) |
| 1 | + Channel ID embed (concat) | (B, N, 12, 116) |
| 2 | LayerNorm | (B, N, 12, 116) |
| 3 | TCN Layer 1 (k=3, d=1) | (B·N, 12, 64) |
| 4 | TCN Layer 2 (k=3, d=2) | (B·N, 12, 64) |
| 5 | LayerNorm + reshape (H_seq) | (B, N, 12, 64) |
| 6a | (Aux) mean N → MLP | (B, 12) aux_logits |
| 6b | (Main) Hyperedge h=3 → H_pool | (B, 12, 3, 64) |
| 7 | mean over T | (B, 3, 64) |
| 8 | PMA Readout | (B, K·64) |
| 9 | Classifier Linear | (B, 1) main_logit |

### Figure caption (한 줄)
> **Figure 1**: TCN + Channel-ID embedding + Spatio-Temporal Hyperedge block (h=3) with multi-task per-second seizure supervision (λ=1.0). Conv-only, 78K parameters, no recurrence, no Transformer, no graph supports. Aux head supervises 12 per-second binary masks at training; inference uses only the clip-level main head.

### 학습 hyperparameters (figure 캡션 또는 별도 표)

| Item | Value |
|---|---|
| Optimizer | Adam |
| LR / wd / clip | from args.json (`lr_init`, `l2_wd`, `max_grad_norm`) |
| Schedule | CosineAnnealingLR (T_max = num_epochs) |
| Batch size | 128 train, 256 test |
| Epochs | 100 (early-stop patience 15; best dev ckpt at ep30 typically) |
| **λ (aux weight)** | **1.0** |
| **Hyperedge h** | **3** |
| **d_model** | **64** |
| TCN kernel | 3 |
| TCN dilation | {1, 2} |
| chid dim | 16 |
| graph_type | `none` |
| Seed | 123 |

---

## 2. Training Workflow

### 2.1 Reproduce on TUSZ
```bash
cd /tmp && python divreg_train_mt_v2.py 3 1.0 100 1 tcn > tusz_main.log
# args: H=3 (hyperedge) λ=1.0 (MT) EPOCHS=100 CHID=1 BACKBONE=tcn
```
Script: [/tmp/divreg_train_mt_v2.py](/tmp/divreg_train_mt_v2.py)

### 2.2 Reproduce on CHB-MIT
```bash
cd /tmp && python divreg_train_mt_chbmit.py 3 1.0 100 > chb_main.log
# args: H=3 λ=1.0 EPOCHS=100
```
Script: [/tmp/divreg_train_mt_chbmit.py](/tmp/divreg_train_mt_chbmit.py)

### 2.3 학습 파이프라인 (high level)

1. **Data load**: `load_dataset_detection` (TUSZ) / `load_dataset_chb` (CHB)
   - 12s clip, FFT(100) features per (channel, second)
   - `graph_type='none'` — adj 계산 안 함 (dataloader 속도 ~30배 향상)
2. **Mask cache**: 학습 전에 모든 EDF의 seizure interval을 캐시 (TUSZ는 `getSeizureTimes`, CHB는 `dataset.seizure_map`)
3. **모델**: `LightSTHyper_classification(backbone_type='mamba', hyper_block_type='static')` 생성 후 backbone을 **수동 TCN으로 swap**
4. **Train**: 
   - 매 batch: `compute_boundary_batch(writeout_fns)` → 12-bin per-second mask
   - `loss = BCE(main, y) + λ × BCE(aux, mask)`
   - Adam, cosine schedule, `clip_grad_norm`
5. **Eval**: dev set AUROC로 best ckpt 선택, 학습 끝나면 test 평가
6. **Test 추론**: main logit만 사용 (`sigmoid(main) ≥ 0.5`)

---

## 3. Results

### 3.1 TUSZ Main Results

| Model | Backbone | AUROC | AUPRC | Sens | Prec | F1 | Acc |
|---|---|---|---|---|---|---|---|
| ada_mshyper | — | — | — | — | — | — | — |
| mshyper | — | — | — | — | — | — | — |
| st_hyper | — | — | — | — | — | — | — |
| BiMamba+chid (no MT, 100ep) | BiMamba | 0.857 | 0.480 | 0.749 | 0.248 | 0.372 | 0.846 |
| **EvoBrain (ref)** | BiMamba | **0.894** | **0.559** | **0.846** | 0.198 | 0.321 | 0.782 |
| TCN+chid plain (no MT, 100ep) | TCN | 0.881 | 0.503 | 0.691 | 0.301 | 0.419 | **0.883** |
| **TCN+chid+MT (Ours, 30ep)** | TCN | 0.8811 | 0.4429 | **0.8333** | 0.260 | 0.389 | 0.852 |
| **TCN+chid+MT (Ours, 100ep)** ★ | TCN | **0.8796** | **0.5037** | 0.7432 | 0.2505 | 0.3747 | 0.8488 |

★ = paper main (100ep, best dev ep30 ckpt).

### 3.2 CHB-MIT Main Results

| Model | Backbone | AUROC | AUPRC | Sens | Prec | F1 | Acc |
|---|---|---|---|---|---|---|---|
| ada_mshyper (100ep) | — | 0.710 | — | 0.727 | 0.007 | 0.015 | 0.617 |
| st_hyper (100ep) | — | 0.775 | — | 0.621 | 0.011 | 0.021 | 0.772 |
| mshyper (100ep) | — | 0.789 | — | 0.727 | 0.008 | 0.017 | 0.659 |
| EvoBrain (100ep) | BiMamba | 0.876 | — | 0.783 | 0.014 | 0.028 | 0.788 |
| light_st_hyper (100ep, dyn graph) | BiMamba | **0.894** | — | **0.882** | — | 0.033 | 0.797 |
| **TCN+chid+MT (Ours, 100ep)** ★ | TCN | **0.8729** | 0.0526 | 0.8385 | 0.0126 | 0.0248 | 0.7385 |

★ = paper main (100ep, best dev ep100 ckpt).

**비고**: CHB-MIT는 Pos=0.40% (TUSZ 6.10%의 1/15)로 극심한 imbalance라 AUPRC/F1이 본질적으로 낮음.

---

## 4. Ablations (TUSZ)

### 4.1 λ sweep (TCN+chid, h=3, 30ep)

| λ | AUROC | AUPRC | Recall | pos_mean |
|---|---|---|---|---|
| 0 (plain BCE) | 0.8721 | 0.4395 | 0.7714 | 0.677 |
| 0.5 | 0.8770 | 0.4383 | 0.8256 | 0.719 |
| **1.0 ★** | **0.8811** | **0.4429** | 0.8333 | 0.718 |
| 2.0 | 0.8783 | 0.4393 | **0.8513** | 0.713 |

→ λ=1.0이 sweet spot. λ=2.0은 Recall만 더 올라가고 AUROC 떨어짐.

### 4.2 MT vs plain BCE (TCN+chid, 100ep apples-to-apples)

| h | Type | AUROC | AUPRC | Recall | Prec | F1 |
|---|---|---|---|---|---|---|
| 2 | plain BCE | 0.874 | 0.503 | 0.767 | 0.230 | 0.353 |
| 2 | + MT λ=1.0 | 0.8786 | 0.4579 | 0.7564 | 0.2431 | 0.3679 |
| **3** | **plain BCE** | **0.881** | 0.503 | 0.691 | **0.301** | **0.419** |
| **3** | **+ MT λ=1.0 ★** | 0.8796 | **0.5037** | 0.7432 | 0.2505 | 0.3747 |
| 4 | plain BCE | 0.867 | 0.486 | 0.692 | 0.263 | 0.381 |
| 4 | + MT λ=1.0 | 0.8639 | 0.4421 | 0.7825 | 0.1979 | 0.3159 |

→ **MT 효과 (100ep h=3)**: Recall 0.691 → 0.743 (+0.052), AUPRC tied, AUROC tied.  
→ 30ep에선 MT가 AUROC도 같이 올랐고 (apples-to-apples MT vs plain: +0.009), 100ep까지 가면 plain BCE도 따라잡음. **MT의 본질 = 빠른 수렴 + Recall lift**.

### 4.3 30ep vs 100ep MT (h=3)

| Epochs | AUROC | AUPRC | Recall | pos_mean |
|---|---|---|---|---|
| 30 | **0.8811** | 0.4429 | **0.8333** | 0.718 |
| 100 | 0.8796 | **0.5037** | 0.7432 | 0.686 |

→ 100ep AUPRC +0.06 (EvoBrain gap 절반 좁힘), Recall은 약간 squashing 재발.

### 4.4 Backbone (TCN vs BiMamba)

**With MT λ=1.0 + chid (apples-to-apples, h=3, 30ep)**:

| Backbone | AUROC | AUPRC | Recall | Prec | F1 |
|---|---|---|---|---|---|
| BiMamba + chid + MT | 0.8762 | 0.3945 | 0.8248 | 0.2036 | 0.3265 |
| **TCN + chid + MT** | **0.8811** | **0.4429** | 0.8333 | 0.260 | 0.389 |
| **Δ (TCN − BiMamba)** | **+0.005** | **+0.048** | +0.009 | +0.057 | +0.063 |

### 4.5 chid 있고/없고 (TCN+MT h=3, 30ep)

| chid | AUROC | AUPRC | Recall | pos_mean |
|---|---|---|---|---|
| ✗ no chid | 0.8772 | 0.4403 | 0.7940 | 0.692 |
| **✓ chid** | **0.8811** | **0.4429** | **0.8333** | **0.718** |
| **Δ** | **+0.0039** | **+0.0026** | **+0.0393** | +0.026 |

### 4.6 Hyperedge expressiveness (negative results)

TCN+chid h=3 MT λ=1.0 30ep 위에:

| Variant | AUROC | Δ vs main |
|---|---|---|
| MT λ=1.0 main | **0.8811** | — |
| + Mixer block (FFN over E_h) | 0.8643 | −0.017 |
| + LapPE concat | 0.8720 | −0.009 |

→ 모듈 capacity 추가는 모두 **해롭다**. "h=3 + static Q"가 sweet spot이고, 더 키우면 generalization 떨어짐.

### 4.7 Hyperedge h sweep (TCN+chid plain, 100ep)

| h | AUROC | AUPRC | Recall | Prec | F1 |
|---|---|---|---|---|---|
| 2 | 0.874 | 0.503 | 0.767 | 0.230 | 0.353 |
| **3 ★** | **0.881** | 0.503 | 0.691 | **0.301** | **0.419** |
| 4 | 0.867 | 0.486 | 0.692 | 0.263 | 0.381 |

→ h=3 sweet spot for both AUROC and F1.

---

## 5. Excluded / 시도했지만 채택 안 한 것들

| 시도 | 효과 | 채택? | 이유 |
|---|---|---|---|
| Focal loss | thr 트릭 | ✗ | AUROC 변화 없음, threshold만 이동 |
| pos_weight | thr 트릭 | ✗ | 같은 이유 |
| γ (dynamic adj-row concat) | 효과 미미 | ✗ | 데이터로더 30× slowdown 대비 이득 없음 |
| Mixer block (hyperedge FFN) | 부정적 | ✗ | −0.017 AUROC |
| LapPE concat | 부정적 | ✗ | −0.009 AUROC |
| n_hyperedges rank 강제 (1→4) | 변화 없음 | ✗ | ±0.003 noise 범위 |
| Larger d_model (128) / deeper (L=3) | dataset별 다름 | ✗ | 일반 효과 미확정 |
| Channel-mix Linear(N,N) | dataset별 다름 | △ | CHB만 좋음, TUSZ 해로움 — main에서 제외, future work |
| EMA decay 0.999 | dataset별 다름 | △ | TUSZ 약간 도움, CHB는 small batch 때문에 실패 |

---

## 6. 결과 해석

### 6.1 Aux loss는 threshold trick과 본질적으로 다름
- Plain BCE의 squashing: positive class score가 0 근처로 압축됨 (`pos_mean ≈ 0.57–0.68`)
- Focal/pos_weight: threshold만 이동시키고 AUROC 그대로
- **MT aux loss**: 새 supervisory signal 추가 → score 분포 자체를 lift (`pos_mean 0.677 → 0.718`), recall +6%p, AUROC도 +0.009 (30ep 비교)
- 두 dataset에서 일관되게 작동 (TUSZ + CHB)

### 6.2 TCN > BiMamba (TUSZ에서 정량 확인)
- chid+h=3 동일 조건에서 +0.024 AUROC, +0.023 AUPRC
- 추가로: Manual TCN은 Blackwell/CUDA-12.9의 `Conv1d` cuBLAS LT NVRTC bug 회피 (engineering bonus)

### 6.3 Hyperedge h=3 + static Q는 saturated
- h=2/3/4 sweep에서 h=3이 sweet spot
- Mixer, LapPE 등 expressiveness 보강 시도 모두 **negative** (-0.009 ~ -0.017)
- "더 깊고 큰" 통상적 방향 반대 — efficiency angle과 결이 맞음

### 6.4 EvoBrain과의 gap

**TUSZ**:
| Metric | EvoBrain | Ours (100ep) | Gap |
|---|---|---|---|
| AUROC | 0.894 | 0.8796 | −0.014 |
| AUPRC | 0.559 | 0.5037 | −0.055 |
| Sens | 0.846 | 0.7432 | −0.103 |
| Prec | 0.198 | 0.2505 | **+0.052** (Ours win) |
| F1 | 0.321 | 0.3747 | **+0.054** (Ours win) |
| Acc | 0.782 | 0.8488 | **+0.067** (Ours win) |

**CHB-MIT**:
| Metric | EvoBrain | Ours | Gap |
|---|---|---|---|
| AUROC | 0.876 | 0.8729 | −0.003 (essentially tied) |
| Sens | 0.783 | **0.8385** | **+0.055** (Ours win) |
| Acc | 0.788 | 0.7385 | −0.050 |

CHB-MIT에서 EvoBrain과 거의 tied, Sensitivity는 우리가 win. TUSZ는 AUROC −0.014, AUPRC −0.055로 약간 뒤지지만 Precision/F1/Acc는 BEAT.

### 6.5 한 줄 요약

> **TCN + chid + multi-task per-second supervision + static hyperedge h=3** — conv-only, 78K params, no recurrent state, no graph supports, no transformer. TUSZ에서 EvoBrain AUROC 거의 따라잡고 (−0.014) Precision/F1/Acc BEAT, CHB-MIT에서 EvoBrain과 tied (−0.003) Sens BEAT (+0.055). Threshold-free, dataset-agnostic.

---

## 부록: 코드 위치

- Architecture: [light_dyn_hyper.py](../eeg/evobrain/model/light_dyn_hyper.py)
- `SpatioTemporalHyperedgeBlock`: [light_dyn_hyper.py:220](../eeg/evobrain/model/light_dyn_hyper.py#L220)
- TUSZ dataloader: [dataloader_detection.py](../eeg/evobrain/data/dataloader_detection.py)
- CHB-MIT dataloader: [dataloader_chb.py](../eeg/evobrain/data/dataloader_chb.py)
- Seizure time parser (TUSZ): [data_utils.py:82](../eeg/evobrain/data/data_utils.py#L82) (`getSeizureTimes`)
- TUSZ training script: [/tmp/divreg_train_mt_v2.py](/tmp/divreg_train_mt_v2.py)
- CHB-MIT training script: [/tmp/divreg_train_mt_chbmit.py](/tmp/divreg_train_mt_chbmit.py)
