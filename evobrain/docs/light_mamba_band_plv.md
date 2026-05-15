# `light_mamba_band_plv` — Mamba-State-Gated Band-Specific PLV Fusion

## 1. 모델이 풀려는 문제

발작 시 채널 간 functional connectivity (FC)는 **주파수 밴드마다 다르게**
나타난다:

- **δ (1-4 Hz)**: 느린 발작파, 후반 ictal 단계에서 광범위 동기화
- **θ (4-8 Hz)**: 측두엽 발작의 hallmark, 이상 양상이 자주 시작되는 곳
- **α (8-13 Hz)**: 정상 휴식 리듬, **발작 중 *붕괴*하는 패턴**이 특징
- **β (13-30 Hz)**: 이차성 일반화 발작의 전조
- **γ (30-50 Hz)**: 발작 초기 tonic onset의 빠른 활동, **일찍 붕괴**

즉 "어느 밴드의 FC가 가장 정보적인가"는 **환자별·발작유형별·발작 진행
시점별로 다르다**. 그런데 기존 multi-band GNN 연구(Multi-Band GNN
Neonatal, STGAT-PLV, Multi-frequency FC-GCN)들은 다 **static하게
concat 또는 fixed weight로 합친다**. 모든 샘플에 같은 가중치를 강제.

**제안**: 가중치를 **Mamba state로 sample-별 결정**. 같은 모델이지만
'이 샘플은 γ 신호가 강하니 γ 위주로 본다' '이 샘플은 α 붕괴가 두드러지니
α-band가 중요하다' 같은 적응이 가능해진다.

## 2. PLV가 뭐고 왜 쓰나

PLV (Phase-Locking Value): 두 채널의 **순간 위상**이 얼마나 일정한
차이를 유지하는지의 측정.

```
PLV(i, j) = | E_t[exp(i(φ_i(t) - φ_j(t)))] |
          ∈ [0, 1]   (1 = 완벽한 phase lock, 0 = 완전 무관)
```

- **상관계수와 차이**: PLV는 진폭 무시, 위상만. 발작의 핵심은 phase
  synchrony — 같은 리듬으로 같이 진동하는지. 진폭 차이는 채널별
  임피던스/위치 노이즈가 많음.
- **밴드별 PLV가 표준**: 신경과학에서 PLV는 항상 특정 밴드로 필터링한
  뒤 계산. 광대역 PLV는 다양한 리듬이 섞여 해석 불가.

## 3. 전체 흐름

```
입력1: STFT 특징 x ∈ R^{B, T, N, D_in}      (Mamba용)
입력2: 원시 신호 raw ∈ R^{B, N, L}          (PLV용, L=2400 for 12s @ 200Hz)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[A] Mamba   [B] Band-PLV 계산 (model/fc_compute.py)
   │         │
   │         │  for band in {δ, θ, α, β, γ}:
   │         │      s_band = bandpass_FFT(raw, band)      ◄── FFT mask
   │         │      φ_band = hilbert_phase(s_band)        ◄── 해석 신호 phase
   │         │      A_band = |E_t[e^(iΔφ)]|               ◄── PLV pairwise
   │         │
   │         │  → A ∈ R^{B, 5, N, N}     5개 밴드별 인접행렬
   │         │
   │         │  Top-k 희소화 + row-normalize
   ▼         ▼
   H ∈ R^{B, T, N, d}     A ∈ R^{B, 5, N, N}
        │                       │
        ▼                       │
[1] 5개 병렬 GCN branch          │
        │                       │
        │   for k in range(5):  │
        │       H_k = GCN(H, A[k])    각 밴드 그래프로 메시지 패싱
        │                              shape: (B, T, N, d)
        ▼
   stack: H_stack ∈ R^{B, 5, T, N, d}
        │
        ▼
[2] Mamba-state 기반 band gate
        │
        │   s_global = mean_n(H[:, -1, :, :])     ◄── 마지막 시점, 채널 평균
        │   g = softmax(MLP_gate(s_global))       ◄── 5-차원 밴드 가중치
        │   shape: g ∈ R^{B, 5}, 각 행이 1로 합산
        │
        │   의미: "이 샘플은 어떤 밴드의 FC를 얼마나 신뢰할지"
        │   (uniform/static gate는 ablation)
        ▼
[3] Gated Fusion
        │   H_fused = Σ_k g[k] * H_stack[k]       ◄── shape: (B, T, N, d)
        ▼
[4] 마지막 시점 추출: (B, N, d)
        ▼
[5] Attention readout (learnable query)
        │   w_n = softmax(<q, H_n>)
        │   z = Σ_n w_n * H_n                     ◄── shape: (B, d)
        ▼
[6] FC → logits (B, num_classes)
```

## 4. 핵심 인수 — Mamba state가 gate를 결정하는 이유

다른 logical 선택지가 있다. 왜 Mamba state?

| 선택지 | 비판 |
|---|---|
| Learnable scalar α_k (Multi-Band GNN Neonatal 식) | 모든 샘플에 같은 가중치 → 환자/유형별 적응 불가 |
| 입력 신호에서 직접 gate (예: STFT band power) | Mamba state는 이미 신호의 시간 진화 정보 통합 → 더 정보적 |
| Cross-attention over band branches | 더 expressive지만 파라미터 폭발, 22채널엔 과함 |
| **Mamba state → MLP_gate (본 선택)** | 시간 통합 + 채널 평균이 자연스러운 sample-level 요약 |

Mamba는 어차피 모든 모델에서 첫 단계인데, 그 state를 어차피 다른
용도(분류 readout)로도 쓴다. 추가 비용 사실상 0.

## 5. Gate Mode 별 차이 (sweep 시 ablation)

| `gate_mode` | g 정의 | 학습 가능? | sample-conditional? |
|---|---|---|---|
| `uniform` | g = [0.2, 0.2, 0.2, 0.2, 0.2] | ✗ | ✗ |
| `static` | g = softmax(learnable_logits) | ✓ | ✗ |
| `mamba` (main) | g = softmax(MLP_gate(s_global)) | ✓ | ✓ |

`uniform vs static`: "어떤 밴드가 평균적으로 더 중요한가?" — 만약
유의미한 차이 없으면 모든 밴드가 비슷하게 기여한다는 뜻.

`static vs mamba`: "**같은 균일 가중치 대신 sample-별로 가중하는 게
의미 있는가?**" ← **본 contribution의 진짜 검증**. mamba >> static이면
sample-adaptive band weighting이 contribution이라는 증거.

## 6. 주요 텐서 shape

| 이름 | shape | 비고 |
|---|---|---|
| `raw` | (B, N=22, L=2400) | 12s @ 200Hz |
| `A_plv` | (B, 5, 22, 22) | 5밴드, top-k 희소화 |
| `H` | (B, T=12, 22, d=64) | Mamba 출력 |
| `H_stack` | (B, 5, 12, 22, 64) | 밴드별 GCN 출력 |
| `g` | (B, 5) | 밴드 가중치 |
| `H_fused` | (B, 12, 22, 64) | gated 결합 |
| `z` | (B, 64) | readout |

## 7. 신규성 vs 선행 연구

| Prior | Band fusion 방식 |
|---|---|
| Multi-Band GNN Neonatal (Appl Sci 2024) | static concat |
| STGAT-PLV (CHB-MIT) | static per-band GAT, 평균 |
| Multi-frequency FC-GCN Alzheimer's (2025) | static |
| **E+ (본 모델)** | **Mamba-state로 sample-별 gated** |

## 8. 잠재적 위험

- **Gate가 한 밴드로 collapse**: 학습 초기에 한 밴드가 약간 좋으면
  gate가 거기로 쏠리고 다른 4개 branch는 학습이 안 됨.
  → entropy regularizer on `g`, 또는 warmup 동안 uniform 강제.
- **Bandpass + Hilbert 비용**: 한 forward에 5번 FFT. 22ch × 2400샘플
  기준 ~5ms per batch (GPU). 50 epoch에서 누적 ~수 분, 무시 가능.
- **PLV가 진폭 무시**: 진폭 정보가 분류에 중요한 발작 유형(예: 진폭
  급증 spike-wave)은 못 잡을 수 있음. 보완으로 `A_corr`(진폭 포함)을
  6번째 view로 추가 가능 (sweep 변종).
