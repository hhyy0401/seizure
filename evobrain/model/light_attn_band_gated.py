"""
`light_attn_band_gated` — Ablation of `light_mamba_band_plv` (E+).

Replaces the band-specific PLV adjacencies with **5 learnable attention heads**.
Everything else stays identical: Bi-Mamba backbone, 5 parallel GCN branches,
Mamba-state gate, attention readout.

This isolates the contribution of PLV (neuroscience-grounded phase synchrony
prior) vs purely learned attention edges in the band-branch design.

E+        : edges from band-PLV (raw signal → bandpass → Hilbert → PLV)
This file : edges from softmax((W_q^k H) (W_k^k H)^T)   for k in 1..5

If E+ > this → PLV prior matters.
If E+ ≈ this → the band-branch + Mamba gating structure is the contribution,
               not the prior.
If E+ < this → learned edges beat PLV, PLV is the wrong inductive bias here.
"""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_backbone import BiMambaBackbone

# Reuse BandGCN from the PLV model (same per-band conv module).
from model.light_mamba_band_plv import BandGCN, GateMode


class LightAttnBandGated(nn.Module):
    """5 learnable attention heads (one per "band slot"), each feeding a GCN
    branch; Mamba-state-gated fusion; attention pool readout.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_hidden: int = 64,
        n_nodes: int = 22,
        n_classes: int = 1,
        n_mamba_layers: int = 2,
        n_gcn_layers: int = 2,
        n_heads: int = 5,                 # equivalent to "n_bands" in PLV model
        gate_mode: GateMode = "mamba",
        topk: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_heads = n_heads
        self.gate_mode = gate_mode
        self.topk = topk

        # --- Per-channel (Bi-)Mamba backbone (same as E+) ---
        self.backbone = BiMambaBackbone(
            d_input=d_input, d_model=d_model,
            n_layers=n_mamba_layers, bidirectional=bidirectional,
        )

        # --- 5 learnable attention heads for edge construction ---
        # We keep heads independent (no shared W_q / W_k) so each branch can
        # learn its own "view" of the channel graph, paralleling 5 frequency
        # bands in the PLV version.
        self.W_q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_heads)])

        # --- One GCN branch per head ---
        self.gcn_branches = nn.ModuleList([
            BandGCN(d_model, d_hidden, n_layers=n_gcn_layers)
            for _ in range(n_heads)
        ])

        # --- Gate (same modes as E+) ---
        if gate_mode == "uniform":
            self.register_buffer(
                "_uniform_gate", torch.full((n_heads,), 1.0 / n_heads)
            )
        elif gate_mode == "static":
            self.static_gate_logits = nn.Parameter(torch.zeros(n_heads))
        elif gate_mode == "mamba":
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, n_heads),
            )
        else:
            raise ValueError(f"unknown gate_mode: {gate_mode}")

        # --- Attention readout over channels (same as E+) ---
        self.readout_query = nn.Parameter(torch.randn(d_hidden) * 0.02)
        self.readout_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden, n_classes)

    def node_mamba(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def compute_head_adj(self, H_last: torch.Tensor) -> torch.Tensor:
        """H_last: (B, N, d) → (B, n_heads, N, N) normalized adjacency."""
        B, N, d = H_last.shape
        outs = []
        for k in range(self.n_heads):
            Q = self.W_q[k](H_last)                                # (B, N, d)
            K = self.W_k[k](H_last)                                # (B, N, d)
            A = torch.softmax(
                torch.einsum("bnd,bmd->bnm", Q, K) / (d ** 0.5),
                dim=-1,
            )                                                      # (B, N, N)
            # Optional top-k sparsify per row to mirror PLV's top-k.
            if self.topk < N:
                vals, idx = A.topk(self.topk, dim=-1)
                A_sparse = torch.zeros_like(A)
                A_sparse.scatter_(-1, idx, vals)
                A = A_sparse / A_sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            outs.append(A)
        return torch.stack(outs, dim=1)                            # (B, K, N, N)

    def compute_gate(self, H_last_pool: torch.Tensor, B: int) -> torch.Tensor:
        if self.gate_mode == "uniform":
            return self._uniform_gate.unsqueeze(0).expand(B, -1)
        if self.gate_mode == "static":
            return F.softmax(self.static_gate_logits, dim=0).unsqueeze(0).expand(B, -1)
        return F.softmax(self.gate_mlp(H_last_pool), dim=-1)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        q = self.readout_query.unsqueeze(0).expand(H.shape[0], -1)
        scores = (H * q.unsqueeze(1)).sum(dim=-1) / (H.shape[-1] ** 0.5)
        w = torch.softmax(scores, dim=-1)
        z = (w.unsqueeze(-1) * H).sum(dim=1)
        return self.readout_norm(z)

    def forward(self, x: torch.Tensor):
        H_seq = self.node_mamba(x)                                 # (B, T, N, d)

        # Build attention adjacencies from the LAST timestep's H (cheap, stable).
        H_last = H_seq[:, -1, :, :]                                # (B, N, d)
        A_heads = self.compute_head_adj(H_last)                    # (B, K, N, N)

        # Per-head GCN
        H_per_head = []
        for k, gcn in enumerate(self.gcn_branches):
            H_per_head.append(gcn(H_seq, A_heads[:, k]))
        H_stack = torch.stack(H_per_head, dim=1)                   # (B, K, T, N, d)

        # Mamba-state gate
        H_last_pool = H_seq[:, -1, :, :].mean(dim=1)               # (B, d)
        g = self.compute_gate(H_last_pool, B=x.shape[0])           # (B, K)
        H_fused = (g.view(-1, self.n_heads, 1, 1, 1) * H_stack).sum(dim=1)

        H_last2 = H_fused[:, -1, :, :]                             # (B, N, d)
        z = self.readout(H_last2)                                  # (B, d)
        return self.classifier(self.dropout(z)), H_last2


class LightAttnBandGated_classification(nn.Module):
    """main.py-compatible wrapper. Does NOT need raw_signal."""

    def __init__(self, args, num_classes, device=None,
                 gate_mode: GateMode = "mamba"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightAttnBandGated(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_gcn_layers=2,
            n_heads=5,
            gate_mode=gate_mode,
            topk=getattr(args, "top_k", 3),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        x = input_seq.permute(0, 2, 1, 3).contiguous()             # (B, N, T, D)
        logits, hidden = self.model(x)
        return logits, hidden
