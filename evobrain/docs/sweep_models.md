# Sweep 모델 5종 — 한 페이지 정리

각 행은 `bash sbatch/sweep_chb.sh` sweep에서 하나의 `--model_name`에 해당.
모든 모델은 같은 하이퍼파라미터(50 epoch, 12s clip, batch 128, lr 1e-4)로 돌아가고
**graph 부분만 다르다**. Per-channel Mamba(또는 Bi-Mamba)는 공통.

---

## 1. `evobrain` — Baseline (원본, NeurIPS 2025 Spotlight)

**역할**: SOTA reference. 이걸 이기는 게 목표.

**구조 요약**:
```
STFT → Node Mamba (per-channel) → H
        Dynamic correlation adj (dataloader) → Edge Mamba → edge weight
        H + LapPE on weighted graph → GCN
        Max pool → FC
```

**핵심 특징**:
- **두 개의 Mamba** (node + edge). 시변 채널 표현과 시변 edge weight를 모두 SSM으로
- **LapPE** (Laplacian Positional Encoding) on weighted graph: 그래프 구조 정보 주입
- pairwise edge만 사용 (higher-order 없음)

**약점 (우리가 노리는 지점)**:
- 두 번째 Mamba가 무겁고, 학습 안정성 의문
- pairwise만으로는 다채널 동기화를 *원자적*으로 표현 못 함

**구현**: `model/EvoBrain.py` (1076줄, 원본)

---

## 2. `light_attention` — Baseline (LightEvoBrain 최선 변종)

**역할**: "EvoBrain의 무거운 두 번째 Mamba가 정말 필요한가?" 의 대조군.

**구조 요약**:
```
STFT → Per-channel Mamba → H
        Edge = multi-head attention softmax(QK^T) over H
        GCN (2-layer) on this attention adj
        Max pool → FC
```

**핵심 특징**:
- Edge Mamba 제거. Edge weight는 H에서 직접 attention으로 계산
- LapPE도 제거
- 가장 가벼움 (≈ 230K params)

**약점**:
- Per-channel Mamba가 채널 간 정보 없이 독립 → 정보가 그래프 layer에 다 떠넘겨짐
- 그래프 layer 한 번만 통과 → mixing이 얕음

**구현**: `model/LightEvoBrain.py` with `edge_type="attention"`

---

## 3. `light_dyn_hyper` (C+) — **NOVEL Headline**

**역할**: 이 sweep의 메인 contribution 후보.

**구조 요약**:
```
STFT → Bi-Mamba (per-channel, 양방향) → H (B, T, N, d)
       시점별 hyperedge query Q_t = MLP_query(mean_n(H_t))   ← 시변 (Mamba state 기반)
       Soft membership M_t = sigmoid(<H_t, Q_t> / √d)        ← 채널→hyperedge 소속도
       Hyperedge embedding h_e = weighted_mean(H_t, M_t)
       Node update H' = aggregate over incident hyperedges
       2층 반복 + residual + LayerNorm
       마지막 시점 → PMA readout → FC
```

**핵심 특징 (novelty)**:
- **Hyperedge가 시간 따라 진화**: Q_t가 Mamba state로 매 timestep 변함. EvoBrain의
  dynamic pairwise graph → dynamic *hypergraph*로 확장
- **end-to-end learnable**: 기존 STHAT/UHM처럼 K-means/KNN으로 hyperedge 만드는 게
  아니라 attention으로 soft membership 학습
- **higher-order 표현**: 한 hyperedge가 여러 채널을 묶음 → 발작 시 다채널 동기화를
  *원자적*으로 모델링

**Hyperparameters**:
- E_h = 8 (hyperedge 개수)
- 2층 hypergraph
- PMA seed query 1개

**구현**: `model/light_dyn_hyper.py` (~190줄). 자세한 흐름은
[light_dyn_hyper.md](light_dyn_hyper.md) 참조.

---

## 4. `light_static_hyper` (C-ab) — C+ Ablation

**역할**: C+의 "Mamba state로 진화"가 정말 도움이 되는가? 검증.

**구조 요약**:
C+와 **완전히 동일**한데, Q_t를 시간 무관 static parameter Q로 교체.
즉 `Q_static ∈ R^{E_h × d}`가 단일 learnable parameter. 시간 따라 안 변함.

**테스트 가설**:
- C+ ≈ C-ab → "soft hyperedge 자체가 핵심, 진화는 부차적"
- C+ >> C-ab → "**Mamba-state-conditioning이 contribution의 핵심**"

후자가 나오면 논문 contribution 정당화. 전자가 나오면 design 단순화 가능.

**구현**: `model/light_dyn_hyper.py`에 `static_queries=True` 플래그.

---

## 5. `light_mamba_band_plv` (E+) — **NOVEL Secondary**

**역할**: 두 번째 contribution 후보. PLV 도메인 지식 + Mamba-state gating.

**구조 요약**:
```
STFT → Bi-Mamba (per-channel) → H
       Raw signal → 5개 밴드 PLV adj (δθαβγ, 미분 가능 GPU 계산)
       H → 5개 GCN branch (각 branch가 한 밴드의 PLV graph로 메시지 패싱)
       g = softmax(MLP_gate(mean_n(H_last)))   ← 5차원 밴드 가중치
       H_fused = Σ_k g_k * H_band_k
       Attention pool over channels → FC
```

**핵심 특징 (novelty)**:
- **PLV가 진폭 무시 phase 동기 측정**: 발작 신경과학에서 표준 FC. EvoBrain의
  correlation graph는 진폭이 섞임 → PLV가 더 깨끗
- **밴드별 분리**: δ는 ictal late, γ는 onset 등 발작 stage마다 다른 밴드가 dominant.
  단일 grand-average correlation은 이걸 collapse
- **Mamba-state-gated fusion**: 기존 multi-band GNN(Multi-Band GNN Neonatal,
  STGAT-PLV)은 static concat. E+는 **sample-별** Mamba state로 band 가중치 결정 →
  환자/발작유형별 적응

**Hyperparameters**:
- 5 밴드: δ(1-4) θ(4-8) α(8-13) β(13-30) γ(30-50) Hz
- 2-layer GCN per band
- gate_mode = "mamba" (main) | "static" (ablation) | "uniform" (ctrl)

**구현**: `model/light_mamba_band_plv.py` (~165줄). 자세한 흐름은
[light_mamba_band_plv.md](light_mamba_band_plv.md) 참조. PLV 계산은
`model/fc_compute.py`에 GPU 미분가능 구현.

---

## 공통 — Bi-Mamba (Boost)

C+, C-ab, E+는 모두 **Bi-Mamba**를 씁니다 (vanilla per-channel Mamba가 약하다는
관찰을 반영한 보강):

```
forward Mamba(x) + reverse Mamba(x.flip(time)).flip(time)
→ 평균 → LayerNorm + residual
```

**왜?** Per-channel Mamba는 각 채널 시계열을 *과거 → 미래*로만 본다. 발작은
짧은 onset 패턴 이후 진행되는 현상이라 미래 정보(같은 clip 내)도 정상 분류에
중요. 양방향이 거의 항상 도움이 됨 (Brain-Go-Brr v4도 채택).

`evobrain`과 `light_attention` baseline은 원본대로 단방향 (불공정해 보이지만,
**baseline은 원본 그대로**가 reviewer 방어에 유리).

**구현**: `model/mamba_backbone.py` → `BiMambaBackbone`

---

## 비교 표 (예상)

| Model | Params | Edge | Conv | Readout | Bi-Mamba |
|---|---|---|---|---|---|
| `evobrain` | ~1.5M | Edge Mamba on corr | GCN+LapPE | Max | ✗ |
| `light_attention` | ~230K | softmax(QK^T) | GCN | Max | ✗ |
| `light_dyn_hyper` (C+) | ~295K | Mamba-evolved hyperedge | HypergraphConv | PMA | ✓ |
| `light_static_hyper` (C-ab) | ~150K | static learnable hyperedge | HypergraphConv | PMA | ✓ |
| `light_mamba_band_plv` (E+) | ~185K | 5-band PLV (fixed) | 5×GCN gated | Attention pool | ✓ |

## 핵심 비교 차원

- **EvoBrain vs Light-attn**: "두 번째 Mamba가 정말 필요한가?"
- **Light-attn vs C+**: "Higher-order(hypergraph) > pairwise?"
- **C+ vs C-ab**: "Mamba-state-driven 진화가 핵심인가?"
- **Light-attn vs E+**: "PLV + band-specific FC가 단일 attention graph보다 나은가?"
- **C+ vs E+**: 두 novelty 중 어느 쪽이 더 큰 lift?

각 비교가 paper의 한 ablation row가 됨.
