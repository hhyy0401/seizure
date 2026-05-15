# Lightweight EvoBrain: Node Mamba + Learnable Edge + GCN

## 변경 요약
- **제거**: Step 1 (cross-correlation), Edge stream Mamba, LapPE
- **추가**: Learnable edge weight (3가지 비교)
- **유지**: STFT 전처리, Node Mamba, GCN, Max pooling + classification

```
Input: Raw EEG (N channels × time)
  │
  ▼
[STFT] → X ∈ R^{N × T × d}
  │
  ▼
[Node Mamba] → H ∈ R^{N × h}
  │
  ▼
[Learnable Edge] → A ∈ R^{N × N}
  │
  ▼
[GCN] → Z ∈ R^{N × d_gcn}
  │
  ▼
[Max Pool → FC → Softmax] → prediction
```

---

## Pseudocode (PyTorch-style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class LightEvoBrain(nn.Module):
    def __init__(self,
                 d_input,         # STFT freq bins
                 d_model,         # Mamba hidden dim
                 d_gcn=64,        # GCN hidden dim
                 n_nodes=19,      # EEG channels
                 n_classes=2,
                 n_mamba_layers=2,
                 n_gcn_layers=2,
                 edge_type='dot', # 'dot', 'bilinear', 'attention'
                 n_heads=4,
                 ):
        super().__init__()
        self.n_nodes = n_nodes
        self.edge_type = edge_type

        # --- Node Mamba ---
        self.input_proj = nn.Linear(d_input, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_mamba_layers)
        ])

        # --- Learnable Edge ---
        if edge_type == 'bilinear':
            self.W_edge = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        elif edge_type == 'attention':
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)

        # --- GCN ---
        self.gcn_layers = nn.ModuleList()
        for i in range(n_gcn_layers):
            in_dim = d_model if i == 0 else d_gcn
            self.gcn_layers.append(nn.Linear(in_dim, d_gcn))

        # --- Classifier ---
        self.classifier = nn.Linear(d_gcn, n_classes)


    def node_mamba(self, x):
        """
        x: (B, N, T, d_input)
        Returns: (B, N, d_model)
        """
        B, N, T, D = x.shape
        x = x.view(B * N, T, D)           # (B*N, T, d_input)
        x = self.input_proj(x)             # (B*N, T, d_model)
        for mamba in self.mamba_layers:
            x = mamba(x)                   # (B*N, T, d_model)
        h = x[:, -1, :]                   # last step: (B*N, d_model)
        return h.view(B, N, -1)            # (B, N, d_model)


    def compute_edge(self, H):
        """
        H: (B, N, d_model)
        Returns: A (B, N, N)
        """
        if self.edge_type == 'dot':
            A = torch.bmm(H, H.transpose(1, 2))       # (B, N, N)
            A = F.softmax(A, dim=-1)

        elif self.edge_type == 'bilinear':
            HW = torch.matmul(H, self.W_edge)          # (B, N, d_model)
            A = torch.bmm(HW, H.transpose(1, 2))       # (B, N, N)
            A = F.softmax(A, dim=-1)

        elif self.edge_type == 'attention':
            B, N, _ = H.shape
            Q = self.W_q(H).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            K = self.W_k(H).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            # (B, n_heads, N, N)
            attn = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_head ** 0.5)
            attn = F.softmax(attn, dim=-1)
            A = attn.mean(dim=1)                        # (B, N, N)

        return A


    def gcn_forward(self, H, A):
        """
        H: (B, N, d_model)
        A: (B, N, N)
        """
        for layer in self.gcn_layers:
            H = torch.bmm(A, H)    # message passing
            H = layer(H)           # linear
            H = F.relu(H)
        return H                   # (B, N, d_gcn)


    def forward(self, x):
        """
        x: (B, N, T, d_input)
        Returns: (B, n_classes)
        """
        # 1. Node Mamba
        H = self.node_mamba(x)          # (B, N, d_model)

        # 2. Learnable Edge
        A = self.compute_edge(H)        # (B, N, N)

        # 3. GCN
        H = self.gcn_forward(H, A)      # (B, N, d_gcn)

        # 4. Classification
        z = H.max(dim=1)[0]             # max pool: (B, d_gcn)
        out = self.classifier(z)        # (B, n_classes)
        return out
```

---

## 파라미터 수 비교 (d_model=64, d_gcn=64 기준)

| 모듈 | dot | bilinear | attention (4-head) |
|------|-----|---------|-------------------|
| Node Mamba (2-layer) | ~50K | ~50K | ~50K |
| Edge weight | 0 | 4K (64²) | 8K (W_q + W_k) |
| GCN (2-layer) | 8K | 8K | 8K |
| Classifier | 130 | 130 | 130 |
| **합계** | **~58K** | **~62K** | **~66K** |

기존 EvoBrain 114K 대비 **약 50% 경량화**.

---

## 실험 계획

| 실험 | 목적 |
|------|------|
| dot vs bilinear vs attention | edge 방식 비교 |
| LightEvoBrain vs EvoBrain | 성능 유지하면서 경량화 되는지 |
| LightEvoBrain vs EvoBrain (no Step1) | Step1 제거 자체의 효과 |
| 학습된 A 시각화 | 어떤 채널 연결을 학습하는지 |
| Inference time 비교 | 실제 속도 차이 |
