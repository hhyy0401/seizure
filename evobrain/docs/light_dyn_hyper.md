# `light_dyn_hyper` — Dynamic Hyperedges Driven by Mamba State

## 1. 모델이 풀려는 문제

발작(seizure)은 **여러 채널이 동시에 같이 움직이는 현상**이다. 시작은 한
영역에서지만 곧 양반구로 퍼지고(generalize) 후엔 다중 채널 동기화가
지속된다. 그래프 모델(EvoBrain 포함)이 이걸 잘 잡는다고들 하는데,
**문제는 "그래프 = pairwise edge"라는 점**. 채널 i, j 사이 edge로는
"세 채널 (i, j, k)가 동시에 같이 움직인다"는 사실을 *원자적*으로 표현
할 수 없다. 세 쌍 (i,j), (j,k), (i,k)이 다 강하다는 *부산물*로만 추론
될 뿐.

Hypergraph는 이걸 직접 모델링할 수 있다. **하나의 hyperedge가 3+개의
채널을 한 묶음으로 묶음**. 발작 전파의 자연스러운 표현이다.

기존 hypergraph EEG 연구(STHAT, UHM)들은 K-means / KNN 같은 **non-
differentiable 클러스터링**으로 hyperedge를 만들었다. 즉 학습 안 되고,
시간에 따라 진화하지도 못한다. 우리는 이 두 한계를 모두 푼다.

## 2. 전체 흐름

```
입력: STFT 특징 x ∈ R^{B, T, N, D_in}                       (시간×채널×주파수)
        │
        ▼
[1] Per-channel Mamba           ◄── 채널별 시계열 표현 학습
        │                          (LightEvoBrain과 동일)
        ▼
    H ∈ R^{B, T, N, d}             각 시점 t에서 각 채널 n의 d차원 표현
        │
        ▼
[2] Dynamic Hyperedge Queries Q_t ◄── 본 모델의 핵심
        │
        │  매 시점 t마다:
        │   g_t = mean_n(H[:, t, :])     모든 채널의 평균 = 전역 상태
        │   Q_t = MLP_query(g_t)         E_h개의 "hyperedge prototype" 생성
        │
        │  Q_t의 의미: "이 시점에서는 이런 채널 그룹들이 중요해"
        │  Mamba state가 시간 따라 변하면 Q_t도 진화 → 동적 hyperedge
        │
        ▼
[3] Soft Membership M_t ∈ R^{B, T, N, E_h}
        │
        │  M_t[n, e] = sigmoid(<H_t[n], Q_t[e]> / √d)
        │            = "채널 n이 hyperedge e에 얼마나 속하는지" (soft)
        │
        │  - 0~1 사이 연속값. discrete clustering과 달리 미분 가능
        │  - 한 채널이 여러 hyperedge에 동시에 속할 수 있음 (multi-membership)
        ▼
[4] Hyperedge Embedding h_e
        │  
        │  h_e[t, e] = Σ_n M_t[n, e] * H_t[n] / (Σ_n M_t[n, e] + ε)
        │           = membership 가중평균
        │
        │  의미: "hyperedge e의 대표 표현" = 그 hyperedge에 속한 채널들의 평균
        ▼
[5] Node Update H'
        │
        │  H'_t[n] = Σ_e M_t[n, e] * h_e[t, e]
        │         = "내가 속한 모든 hyperedge의 정보를 합쳐서 내 표현 갱신"
        │
        │  여기서 채널 간 정보 교환이 일어남 (단, hyperedge를 경유해서).
        │  Residual + LayerNorm으로 안정화.
        ▼
   2회 반복 (n_hyper_layers=2)
        ▼
[6] Temporal Aggregation: H[:, -1, :, :]    마지막 시점만 사용
                                            (Mamba가 이미 시간 통합)
        ▼
[7] PMA Readout (Set Transformer 식)
        │
        │  Learnable seed query q ∈ R^d → 채널에 attention pool
        │  z = Σ_n softmax(<q, H_n>) * H_n
        │
        │  의미: "어떤 채널이 분류에 중요한지" 학습형 가중평균
        │       (max pool보다 더 부드럽게 정보 보존)
        ▼
[8] FC → logits (B, num_classes)
```

## 3. 주요 텐서 shape

| 이름 | shape | 의미 |
|---|---|---|
| `x` | (B, T, N, D_in) | STFT 입력 |
| `H` | (B, T, N, d) | Mamba 출력 |
| `g_t` | (B, T, d) | 시점별 전역 채널 풀링 |
| `Q_t` | (B, T, E_h, d) | 시변 hyperedge query |
| `M_t` | (B, T, N, E_h) | 채널→hyperedge 소속도 |
| `h_e` | (B, T, E_h, d) | hyperedge 표현 |
| `z` | (B, d) | PMA readout |

`N=22, T=12, d=64, E_h=8` 기본. `M_t` 메모리 ≈ 270K floats / batch (가벼움).

## 4. 왜 SOTA(EvoBrain)를 이길 가능성이 있나

| 측면 | EvoBrain | C+ (본 모델) |
|---|---|---|
| Edge | pairwise (N×N) | hyperedge (N×E_h, higher-order) |
| Edge 진화 | Mamba on edge feature 시계열 | Mamba state → query MLP |
| Edge 표현력 | 두 채널 간 강도 | 다채널 동기 그룹 |
| 학습 | end-to-end | end-to-end |

**가설**: EvoBrain은 "어떤 두 채널이 강하게 연결됐는가"는 잘 잡지만,
"세-네 채널이 동시에 떼지어 발화한다"는 발작 패턴은 부산물로만 추론.
C+는 그걸 *직접* hyperedge로 모델링.

## 5. 위험 요소 / 실패 시 가능성

- **N=22는 hypergraph가 빛나기에 작음**: hypergraph 표현 이득은 보통
  N≥100에서 큼. 22채널에선 pairwise로도 잡힐 수 있음.
- **M_t collapse**: 모든 채널이 같은 hyperedge에 몰리면 의미 상실.
  → entropy regularizer 또는 `M_t` 행 합 제약 추가 고려.
- **Q_t 불안정**: 시간 따라 너무 출렁이면 학습 불안정. 필요시
  smoothing `Q_t ← (1-α) Q_{t-1} + α MLP(g_t)`로 stabilize.

## 6. Ablation: `light_static_hyper`

동일한 아키텍처에서 `Q_t`만 **시간 무관 static parameter Q**로 바꿈
(`static_queries=True` 플래그).

테스트하는 가설: "동적 진화가 정말 도움이 되나, 아니면 학습형
soft hyperedge만으로 충분한가?"

이 ablation이 C+와 큰 차이를 보이면 → **Mamba-state-conditioning이
contribution의 핵심**임을 입증. 차이 없으면 → soft hyperedge 자체가
key, 진화는 부차적.

## 7. 하이퍼파라미터

| 이름 | 기본값 | 비고 |
|---|---|---|
| `E_h` | 8 | hyperedge 개수 (4, 8, 16 sweep 가능) |
| `n_hyper_layers` | 2 | 깊이 |
| `d` | 64 | LightEvoBrain과 동일 |
| `n_pma_seeds` | 1 | readout query 수 (multi-aspect classification 시 ↑) |

## 8. 신규성 vs 선행 연구

| Prior | Hyperedge 구성 | 시변? | end-to-end? |
|-------|---------------|------|-------|
| STHAT (BSPC 2024) | KNN + K-means on FC | 부분 | ✗ |
| UHM (2025) | Hypergraph + Mamba, fixed topology | ✗ | ✗ |
| SoftHGNN (IJCV 2026, CV) | Soft membership via attention | ✗ | ✓ |
| **C+ (본 모델)** | Mamba-state-conditioned query | ✓ | ✓ |

**Claim**: "EvoBrain의 dynamic pairwise graph를 dynamic *hyper*graph로
확장하고, hyperedge query를 SSM state로 진화시킨다."
