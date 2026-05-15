# Baseline Candidates for CIKM 2026 Submission

작성: 2026-05-13. 우리 main contribution: `light_st_hyper` (Bi-Mamba + spatio-temporal hyperedge, (N,T) joint soft prototype pooling). 가장 위협적인 prior work들의 정직한 분석.

## 우리 surviving 5 model + evobrain (이미 실험됨)

| Model | Backbone | Hyperedge | CHB-MIT (1:1, 100ep) AUROC | Params |
|---|---|---|---|---|
| evobrain (baseline) | dual Mamba | pairwise + LapPE | 0.876 | 184K |
| light_st_hyper_linear | Linear (no SSM) | ST hyperedge | 0.878 | 16K |
| light_st_hyper_uni | uni-Mamba | ST hyperedge | 0.889 | 82K |
| light_st_hyper_mscale | Bi-Mamba | ST + multi-scale temporal | 0.889 | 147K |
| **light_st_hyper** (BEST) | **Bi-Mamba** | **ST hyperedge** | **0.894** | 147K |

TUSZ 결과는 진행 중 (2026-05-13 sweep).

---

## P0 — 비교 안 하면 reject 위험

### 1. UHM (IJNS 2026) — 최위협
- **Title**: "A Unified Hypergraph-Mamba Framework for Adaptive EEG Modeling in Multi-view Seizure Prediction"
- **Authors**: Changxu Dong, Dengdi Sun, Zejing Zhang, Bin Luo (Anhui University ICSP lab)
- **DOI**: 10.1142/S012906572550056X
- **Venue**: International Journal of Neural Systems 36(1), 2026
- **Threat**: ⚠️ **이름부터 같음** — Hypergraph + Mamba for seizure prediction. 우리 main contribution과 직접 충돌 가능성.
- **차별점 (paper 본문 받기 전 미확정)**: "Multi-view" — 어떤 view 분해인지 불명
- **Code**: ❌ 없음. paywall.
- **Action**: 학교 access → ResearchGate → 저자 이메일 → 최후수단 $45 결제

### 2. STHAT (BSPC 2025)
- **Title**: "EEG-based patient-specific seizure prediction based on Spatial–Temporal Hypergraph Attention Transformer"
- **Authors**: Changxu Dong, Dengdi Sun, Zejing Zhang, Bin Luo (Anhui Univ ICSP)
- **DOI**: 10.1016/j.bspc.2024.107075
- **Venue**: Biomedical Signal Processing and Control, Volume 100, 2025
- **Threat**: 높음 (EEG + ST hypergraph 명명)
- **차별점**: **Decomposed design** — Swin Transformer (temporal) + KNN/K-means dynamic hypergraph (spatial). Hyperedge는 채널만 묶음. 우리(joint (N,T))와 디자인 다름.
- **Code**: ❌ 없음. paywall.
- **Effort**: 7-10일 (Swin Transformer + KNN dynamic hyperedge 재구현)

### 3. ST-Hyper (CIKM 2025)
- **Title**: "ST-Hyper: Learning High-Order Dependencies Across Multiple Spatial-Temporal Scales for Multivariate Time Series Forecasting"
- **Authors**: Binqing Wu, Jianlong Huang, Zongjiang Shang, Ling Chen (Zhejiang Univ Ling Chen lab)
- **arxiv**: 2509.02217
- **Venue**: **CIKM 2025** (우리와 같은 venue prior)
- **Threat**: 매우 높음 (concurrent + (N,T) joint hyperedge + 같은 venue → reviewer 알 가능성↑)
- **차별점**:
  - ST-Hyper: multi-scale ST nodes (N₁+…+Nⱼ)×K, tri-phase conv 𝓔₁=ϕ(𝑼𝚲̃ᵀ𝓧)+𝚲̃ᵀ𝓧
  - 우리: single-scale T×N, soft prototype Q[e] (sigmoid attention)
  - ST-Hyper는 forecasting → node feature 출력. 우리는 classification → hyperedge embedding 출력
- **Code**: ❌ 직접 없음. **✅ 같은 lab의 Ada-MSHyper (NeurIPS 2024) repo로 70-80% 재현 가능**
- **Effort**: 3-5일 (Ada-MSHyper 코드 활용)

---

## P1 — 강하게 권장 (자체 baseline + ST-Hyper 구현 base)

### 4. Ada-MSHyper (NeurIPS 2024)
- **Title**: "Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting"
- **Authors**: Zongjiang Shang, Ling Chen, Binqing Wu, Dongliang Cui
- **arxiv**: 2410.23992
- **Code**: ✅ https://github.com/shangzongjiang/Ada-MSHyper
- **Threat**: 중 (forecasting domain, NeurIPS 2024 prior to ST-Hyper)
- **차별점**: Multi-scale temporal hypergraph (시간축만, joint (N,T) 아님)
- **Effort**: 1-2일 (공식 코드)
- **역할**: ST-Hyper 구현의 핵심 base (AHL module 공유)

### 5. MSHyper (2024)
- **Title**: "MSHyper: Multi-Scale Hypergraph Transformer for Long-Range Time Series Forecasting"
- **Authors**: Zongjiang Shang, Ling Chen, Binqing Wu, Dongliang Cui (같은 lab)
- **arxiv**: 2401.09261
- **Code**: ✅ https://github.com/shangzongjiang/MSHyper
- **Threat**: 중-약 (Ada-MSHyper의 이전 버전)
- **Effort**: 1-2일

---

## P2 — 시간 있으면 추가

### 6. DBHCN (Frontiers 2024)
- **Title**: "An epilepsy detection method based on multi-dimensional feature extraction and dual-branch hypergraph convolutional network"
- **Authors**: Liu, Yang, Li, Luo J (다른 그룹)
- **Open access**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11047041/
- **Threat**: 중 (EEG seizure, hypergraph)
- **차별점**: Channel-only hyperedge (KNN by Euclidean distance), spatial-only. LOOCV + Accuracy metric (setting 다름 → 직접 비교 어려움)
- **Reported**: TUH 96.9% Acc, CHB-MIT 94.4% Acc (LOOCV, subject-independent)
- **Code**: ❌ 없음. PMC 무료 access만.
- **Effort**: 3-5일

### 7. ST Dynamic Hypergraph IB (IJNS 2024)
- **Title**: "Spatial-Temporal Dynamic Hypergraph Information Bottleneck for Brain Network Classification"
- **Authors**: 같은 Anhui ICSP group (Dong et al.)
- **DOI**: 10.1142/S0129065724500539
- **Threat**: 중-높음 (이름이 정확히 ST hypergraph + brain network)
- **차별점 (미확정)**: Information bottleneck 도입. Brain network classification (EEG-related)
- **Code**: ❌ 없음. paywall.

---

## P3 — 다른 도메인, skip 가능

| Method | Domain | 이유 |
|---|---|---|
| Hyper-STTN (arxiv 2401.06344) | 보행자 trajectory | EEG와 무관 |
| STH-SepNet (arxiv 2505.19620) | Traffic forecasting | Decoupled (spatial-only hyperedge), 우리와 다름 |
| Hypergraph ST anomaly (arxiv 2410.22256) | Time series anomaly | Task 다름 |

---

## 이미 구현된 baseline (additional 작업 불필요)

`args.py` choices: `evobrain`, `dcrnn`, `evolvegcn`, `BIOT`, `gru_gcn`, `graphs4mer`, `lstm`, `cnnlstm`

---

## 구현 우선순위 (CIKM 2026 target)

| Phase | Action | ETA |
|---|---|---|
| **P1-first** | Ada-MSHyper + MSHyper 공식 코드 활용 → 우리 pipeline에 plug-in | 2-3일 |
| **P0-ST-Hyper** | Ada-MSHyper base + ST 확장 (paper 수식대로 수정) | +2일 |
| **P0-STHAT** | best-effort 재구현 (paper 디테일 부족 명시) | 7-10일 |
| **P0-UHM** | paper 본문 받은 후 결정 | TBD |

---

## Open issue (paper 본문 필요)

- **UHM 2026 IJNS** 본문 → 우리 contribution과 정확한 차이 확인 P0
- **STHAT 2025 BSPC** 본문 → 구현 디테일
- **ST Dynamic Hypergraph IB 2024 IJNS** 본문 (있으면 보너스)

## 학습 setting (모든 baseline 동일)
- Dataset: CHB-MIT (sampling_ratio=1, 1:1) 및 TUSZ v2.0.6 (sampling_ratio=1)
- num_epochs=100, eval_every=10, fix_threshold=0.5
- train_batch=128, test_batch=256, lr=1e-4, weight_decay=5e-4
- num_workers=4 (병렬 4 모델 launch 시), 16 (단독)
- graph_type=none for hypergraph models (ours 포함)
- Metric: AUROC (primary), F1/Acc/Pre/Recall (보조)

---

## Implemented baselines — exact forecasting → classification adaptation

### `ada_mshyper` (model/ada_mshyper_eeg.py)
- **Source**: Ada-MSHyper (Shang et al., NeurIPS 2024) — official repo
  https://github.com/shangzongjiang/Ada-MSHyper. Same author group as
  ST-Hyper (CIKM 2025).
- **Ported verbatim**: `Bottleneck_Construct` (CSCM), `MultiAdaptiveHypergraph` (AHL),
  `HypergraphConv` (PyG MessagePassing + constraint loss), `SelfAttentionLayer`,
  `get_mask`. Normalization (mean/std), AHL invocation, CSCM, inter-scale
  edge_sums loop, intra-scale HGNN result_tensor concat, padding to 80,
  hyperedge attention, 3-path summation (`x_id + x_out + x_out_inter`),
  `Linear_Tran` projection — **all 100% identical to original Model.forward**.
- **Only departures from original** (documented in code):
  1. EEG pre-processing: `(B, N, T, F)` STFT input → `feat_proj: F → d_inner`
     → reshape to `(B, T, N*d_inner)` so model receives the original's
     `(B, seq_len, enc_in)` contract with `seq_len=T`, `enc_in=N*d_inner`.
  2. Removed denormalization (`x * std_enc + mean_enc`) because we're not
     forecasting a value at the original scale.
  3. **Final FF head replaces forecasting projection**: forecasting output
     `(B, pred_len, channels)` → `Flatten → LayerNorm → Linear(pred_len*channels,
     pred_len*4) → GELU → Dropout → Linear(pred_len*4, num_classes)`. The
     `pred_len` is re-purposed as an internal embedding hidden dim.
- **EEG config** (default): `d_inner=8, window_size=[2,2,3], hyper_num=[16,12,8,4],
  topk=10, hidden_len=16`. all_size=[12,6,3,1], Ms_length=22.
- **Params**: ~700K (CHB-MIT N=22) / ~580K (TUSZ N=19).

### `mshyper` (model/mshyper_eeg.py)
- **Source**: MSHyper (Shang et al., 2024, arxiv:2401.09261) — official repo
  https://github.com/shangzongjiang/MSHyper. Predecessor of Ada-MSHyper.
- **Key delta from Ada-MSHyper**: FIXED (non-learnable) hypergraph from
  `get_mask` — uses intra-scale window hyperedges (`inner_size` length)
  + dilated `khop`-step skip hyperedges. No AHL.
- **Ported verbatim**: `build_mshyper_hyperedge_index` (translation of
  Layers.get_mask's active intra-scale parts) + reuse of
  `Bottleneck_Construct`, `HypergraphConv`, `SelfAttentionLayer`,
  3-path summation, `Linear_Tran` from `ada_mshyper_eeg`.
- **EEG / FF head**: same as `ada_mshyper`.
- **EEG config**: `inner_size=4, khop=2, window_size=[2,2,3]`. Produces
  12 fixed hyperedges over Ms_length=22 multi-scale node axis.
- **Params**: ~410K (CHB-MIT) / ~307K (TUSZ).

### `st_hyper` (model/st_hyper_eeg.py)
- **Source**: ST-Hyper (Wu et al., CIKM 2025, arxiv:2509.02217). **Official
  code NOT public** — we use the same author group's Ada-MSHyper repo as
  the authoritative implementation of the core machinery.
- **ST-Hyper-specific delta from Ada-MSHyper**: per ST-Hyper §3
  "we treat each feature of 𝓧 as a node in a hypergraph. Thus, the
  total number of nodes in a hypergraph α is (N₁+…+Nⱼ)×K", we change
  the node axis from temporal-only to **joint (N×T)**: input is reshaped
  to `(B, N*T, d_inner)` so `seq_len = N*T` becomes the joint
  spatio-temporal node axis. AHL learns hyperedges over (N×T) nodes;
  CSCM pools over the joint axis. All other components (normalization,
  CSCM, HypergraphConv, SelfAttentionLayer, 3-path summation,
  Linear_Tran) are **byte-identical to Ada-MSHyper**.
- **Final FF head**: same classification FF head as `ada_mshyper`.
- **EEG config** (default): `d_inner=8, window_size=[2,2,2], hyper_num=[32,16,8,4],
  topk=10, hidden_len=16`. For CHB-MIT (N=22, T=12): seq_len=264,
  all_size=[264,132,66,33], Ms_length=495.
- **Params**: ~29K (CHB-MIT) — much smaller because channels=d_inner=8.
- **Caveat (clearly stated in code docstring + paper writing)**: Without
  access to ST-Hyper's official paper code, this is a *best-effort*
  faithful adaptation of the joint (N×T) idea on the same author group's
  authoritative codebase. Will be flagged in the paper.
