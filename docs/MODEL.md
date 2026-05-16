# LightSTHyper — Architecture & Pseudocode

The main model in this repo (`src/model/light_dyn_hyper.py:LightSTHyper`).
Designed as a lightweight, learnable-hypergraph alternative to the dynamic-
adjacency EvoBrain reproduction (`src/model/EvoBrain.py`).

## High-level pipeline

```
Input: raw EEG → resample 200 Hz → STFT (1 s windows) → FFT magnitudes
  X ∈ R^(B × T × N × D)                  T=12, N=19, D=100 for TUSZ 12s

  ▼
Per-channel BiMamba (forward + reverse, two-layer, d_model)
  H_seq ∈ R^(B × T × N × d_model)

  ▼ + learnable node embedding (N × d_model)
  ▼

SpatioTemporalHyperedgeBlock ×2
  - Project each (t, n) into hyperedge logits M ∈ R^(B × T × N × E_h)
  - Soft-assign nodes/timesteps to hyperedges: M̂ = softmax(M, dim=node)
  - Hyperedge representations: h_edge ∈ R^(B × E_h × d)
  - Update node features by aggregating from edges
  → H_seq' ∈ R^(B × T × N × d)

  ▼  (mean over time → B × N × d)
  ▼

PMA readout (Set Transformer): seed vector queries node tokens
  z ∈ R^(B × d × n_seeds)

  ▼

Linear → sigmoid → BCE loss (binary seizure detection)

Optional: aux head on the last hyperedge block
  - "bce":     per-edge classifier supervises h_edge with the same target
  - "entropy": minimize membership entropy on M̂ (encourage hard assignments)
```

## Why LightSTHyper differs from EvoBrain (paper)

| Aspect | Paper EvoBrain | Ours LightSTHyper |
|---|---|---|
| Temporal backbone | Mamba (uni-directional, causal) | **Bi-Mamba** (forward + reverse, mean) |
| Spatial structure | Dynamic xcorr graph + top-k edges | **Learnable hypergraph** (no xcorr at runtime) |
| Spatial module | DCRNN-style diffusion conv | SpatioTemporalHyperedgeBlock (set-style) |
| Readout | Per-node MLP + max pool | **PMA readout** (Set Transformer) |
| Deep supervision | none | Optional **per-edge BCE** (aux head) |
| Eval-time graph cost | O(N²) xcorr per batch | O(N · E_h) projection |
| 1 epoch on TUSZ 12s | ~5–10 min | ~1.5 min |

## Pseudocode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class BiMambaBackbone(nn.Module):
    """forward + reverse Mamba per channel, mean-combined."""

    def __init__(self, d_input, d_model, n_layers=2, bidirectional=True):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.fwd = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])
        self.bwd = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_layers)]
        ) if bidirectional else None
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x):  # x: (B, N, T, d_input)
        B, N, T, D = x.shape
        h = self.input_proj(x.reshape(B * N, T, D))           # (B*N, T, d)
        for i, (f, ln) in enumerate(zip(self.fwd, self.norms)):
            h_f = f(h)
            if self.bwd is not None:
                h_b = self.bwd[i](h.flip(dims=[1])).flip(dims=[1])
                h_new = 0.5 * (h_f + h_b)
            else:
                h_new = h_f
            h = ln(h + h_new)
        return h.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()  # (B, T, N, d)


class SpatioTemporalHyperedgeBlock(nn.Module):
    """Learnable hyperedge: cluster (t, n) tokens into E_h soft groups."""

    def __init__(self, d_in, d_out, n_hyperedges):
        super().__init__()
        self.proj_logits = nn.Linear(d_in, n_hyperedges)
        self.edge_to_node = nn.Linear(d_in, d_out)
        self.node_proj = nn.Linear(d_in, d_out)

    def forward(self, x):  # x: (B, T, N, d_in)
        logits = self.proj_logits(x)                          # (B, T, N, E_h)
        M = F.softmax(logits, dim=-2)                          # soft assignment over nodes
        # Hyperedge representations (mean over (t, n) weighted by M)
        h_edge = torch.einsum("btne,btnd->bed", M, x)          # (B, E_h, d)
        # Send edge info back to nodes
        msg = torch.einsum("btne,bed->btnd", M, self.edge_to_node(h_edge))
        out = self.node_proj(x) + msg                          # residual update
        self.last_M = M
        self.last_h_edge = h_edge
        return out, M


class PMAReadout(nn.Module):
    """Set Transformer–style readout with seed queries."""

    def __init__(self, d_model, n_seeds=1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_pool):  # (B, N, d)
        B, N, d = H_pool.shape
        seeds = self.seeds.unsqueeze(0).expand(B, -1, -1)
        attn = torch.einsum("bsd,bnd->bsn", seeds, H_pool) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bsn,bnd->bsd", attn, H_pool)
        return self.norm(out).reshape(B, -1)


class LightSTHyper(nn.Module):
    def __init__(self, d_input, d_model, d_hidden, n_nodes, n_classes,
                 n_mamba_layers=2, n_hyper_layers=2, n_hyperedges=3,
                 n_pma_seeds=1, bidirectional=True, dropout=0.0,
                 use_node_emb=True, aux_type="bce"):
        super().__init__()
        self.backbone = BiMambaBackbone(d_input, d_model, n_mamba_layers, bidirectional)
        self.node_emb = (
            nn.Parameter(torch.randn(n_nodes, d_model) * 0.02) if use_node_emb else None
        )
        self.hyper_layers = nn.ModuleList()
        for i in range(n_hyper_layers):
            d_in = d_model if i == 0 else d_hidden
            self.hyper_layers.append(SpatioTemporalHyperedgeBlock(d_in, d_hidden, n_hyperedges))
        self.pma = PMAReadout(d_hidden, n_pma_seeds)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden * n_pma_seeds, n_classes)
        # Aux head (BCE deep supervision) on the last hyperedge block
        self.aux_type = aux_type
        if aux_type == "bce":
            self.aux_classifier = nn.Linear(d_hidden, n_classes)

    def forward(self, x):  # (B, T, N, d_input) -> permuted internally
        x = x.permute(0, 2, 1, 3).contiguous()               # (B, N, T, d_input)
        H_seq = self.backbone(x)                             # (B, T, N, d_model)
        if self.node_emb is not None:
            H_seq = H_seq + self.node_emb.unsqueeze(0).unsqueeze(0)
        for layer in self.hyper_layers:
            H_seq, _ = layer(H_seq)
        H_pool = H_seq.mean(dim=1)                           # (B, N, d_hidden)
        z = self.pma(H_pool)
        return self.classifier(self.dropout(z))

    def compute_aux_loss(self, y):
        """Per-edge BCE deep supervision on last layer's h_edge."""
        last = self.hyper_layers[-1]
        if self.aux_type == "bce":
            edge_logits = self.aux_classifier(last.last_h_edge).squeeze(-1)  # (B, E_h)
            y_b = y.float().unsqueeze(-1).expand_as(edge_logits)
            return F.binary_cross_entropy_with_logits(edge_logits, y_b)
        if self.aux_type == "entropy":
            M = last.last_M
            p = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return -(p * p.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return y.new_zeros(())
```

## Ablation switches (CLI)

```bash
# Backbone choice (default = mamba):
--model_name {light_st_hyper, light_st_hyper_linear, light_st_hyper_dwsep}

# Bi-Mamba toggle (default = bidirectional):
--bidirectional / --no_bidirectional

# Hyperedge config:
--n_hyperedges 3        # E_h
--n_hyper_layers 2
--n_pma_seeds 1
--use_node_emb

# Aux head:
--aux_type {none, bce, entropy}
--aux_weight 0.3

# Standard hyper:
--rnn_units 64          # d_model
--dropout 0.0
--lr_init 3e-4
--l2_wd 5e-4
```
