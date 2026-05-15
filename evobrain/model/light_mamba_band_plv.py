"""
`light_mamba_band_plv` — Mamba-State-Gated Band-Specific PLV Fusion.

See docs/light_mamba_band_plv.md for the full pseudocode and motivation.

Architecture:
    Per-channel Mamba → H (B, T, N, d)
    Raw signal → band-specific PLV adjacencies (5 bands)
    H → 5 GCN branches, each fed its band's PLV graph
    Mamba state → per-sample band gate g ∈ R^5
    Gated fusion → attention pool over channels → FC → logits

`gate_mode`:
    'uniform' — g = [0.2]*5 (no learning, control)
    'static'  — g is a learnable parameter, NOT data-conditional (ablation)
    'mamba'   — main contribution: g = softmax(MLP(mean-pooled H_last))
"""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fc_compute import band_plv, row_normalize, BANDS_DEFAULT
from model.mamba_backbone import BiMambaBackbone


GateMode = Literal["uniform", "static", "mamba"]


class BandGCN(nn.Module):
    """A single 2-layer GCN over a fixed adjacency.

    Forward expects:
        H: (B, T, N, d_in)
        A: (B, N, N) — normalized adjacency
    Returns:
        H': (B, T, N, d_out)
    """

    def __init__(self, d_in: int, d_out: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            din = d_in if i == 0 else d_out
            self.layers.append(nn.Linear(din, d_out))

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # A: (B, N, N), H: (B, T, N, d). Apply A across the channel dim, per t.
        for layer in self.layers:
            # AH per timestep: einsum b n m, b t m d -> b t n d
            H = torch.einsum("bnm,btmd->btnd", A, H)
            H = layer(H)
            H = F.gelu(H)
        return H


class LightMambaBandPLV(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_hidden: int = 64,
        n_nodes: int = 22,
        n_classes: int = 1,
        n_mamba_layers: int = 2,
        n_gcn_layers: int = 2,
        fs: float = 200.0,
        bands=BANDS_DEFAULT,
        gate_mode: GateMode = "mamba",
        topk: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.fs = fs
        self.bands = list(bands)
        self.n_bands = len(self.bands)
        self.gate_mode = gate_mode
        self.topk = topk

        # --- Per-channel (Bi-)Mamba backbone ---
        self.backbone = BiMambaBackbone(
            d_input=d_input, d_model=d_model,
            n_layers=n_mamba_layers, bidirectional=bidirectional,
        )

        # --- One GCN branch per band ---
        self.gcn_branches = nn.ModuleList([
            BandGCN(d_model, d_hidden, n_layers=n_gcn_layers)
            for _ in range(self.n_bands)
        ])

        # --- Band gate ---
        if gate_mode == "uniform":
            self.register_buffer(
                "_uniform_gate", torch.full((self.n_bands,), 1.0 / self.n_bands)
            )
        elif gate_mode == "static":
            self.static_gate_logits = nn.Parameter(torch.zeros(self.n_bands))
        elif gate_mode == "mamba":
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, self.n_bands),
            )
        else:
            raise ValueError(f"unknown gate_mode: {gate_mode}")

        # --- Attention readout over channels ---
        self.readout_query = nn.Parameter(torch.randn(d_hidden) * 0.02)
        self.readout_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden, n_classes)

    def node_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, d_input) → (B, T, N, d_model)."""
        return self.backbone(x)

    def compute_band_adj(self, raw_signal: torch.Tensor) -> torch.Tensor:
        """raw_signal: (B, N, L) → (B, K, N, N) normalized adjacency per band."""
        A = band_plv(raw_signal, fs=self.fs, bands=self.bands)         # (B, K, N, N)
        # Sparsify top-k per row then row-normalize (with self-loops).
        B, K, N, _ = A.shape
        A_flat = A.reshape(B * K, N, N)
        # local topk then renormalize
        if self.topk < N:
            vals, idx = A_flat.topk(self.topk, dim=-1)
            sparse = torch.zeros_like(A_flat)
            sparse.scatter_(-1, idx, vals)
            A_flat = sparse
        A_flat = row_normalize(A_flat)
        return A_flat.reshape(B, K, N, N)

    def compute_gate(self, H_last_pool: torch.Tensor, B: int) -> torch.Tensor:
        """Return gate g of shape (B, K)."""
        if self.gate_mode == "uniform":
            return self._uniform_gate.unsqueeze(0).expand(B, -1)
        if self.gate_mode == "static":
            return F.softmax(self.static_gate_logits, dim=0).unsqueeze(0).expand(B, -1)
        # mamba: data-conditional, per-sample
        return F.softmax(self.gate_mlp(H_last_pool), dim=-1)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        """Attention pool over the channel dim. H: (B, N, d) → (B, d)."""
        q = self.readout_query.unsqueeze(0).expand(H.shape[0], -1)
        scores = (H * q.unsqueeze(1)).sum(dim=-1) / (H.shape[-1] ** 0.5)
        w = torch.softmax(scores, dim=-1)
        z = (w.unsqueeze(-1) * H).sum(dim=1)
        return self.readout_norm(z)

    def forward(self, x: torch.Tensor, raw_signal: torch.Tensor):
        """
        x:         (B, N, T, d_input)  — STFT features for Mamba
        raw_signal:(B, N, L)           — 200Hz time-domain clip for PLV
        """
        H_seq = self.node_mamba(x)                                     # (B, T, N, d)
        A_bands = self.compute_band_adj(raw_signal)                    # (B, K, N, N)

        # Per-band GCN
        H_per_band = []
        for k, gcn in enumerate(self.gcn_branches):
            H_per_band.append(gcn(H_seq, A_bands[:, k]))               # (B, T, N, d)
        H_stack = torch.stack(H_per_band, dim=1)                       # (B, K, T, N, d)

        # Gate from Mamba state at last timestep, mean over channels.
        H_last_pool = H_seq[:, -1, :, :].mean(dim=1)                   # (B, d)
        g = self.compute_gate(H_last_pool, B=x.shape[0])               # (B, K)
        H_fused = (g.view(-1, self.n_bands, 1, 1, 1) * H_stack).sum(dim=1)  # (B, T, N, d)

        H_last = H_fused[:, -1, :, :]                                  # (B, N, d)
        z = self.readout(H_last)                                       # (B, d)
        return self.classifier(self.dropout(z)), H_last


class LightMambaBandPLV_classification(nn.Module):
    """main.py-compatible wrapper.

    Call signature:
        logits, hidden = model(x, seq_lengths, adj, raw_signal=raw)

    `x`   from CHB dataloader: (B, T, N, D_stft).
    `raw_signal` (B, N, L_samples) must be passed in by the trainer; the CHB
    dataloader is being extended to surface this alongside `x`.
    """

    def __init__(self, args, num_classes,
                 device=None, gate_mode: GateMode = "mamba"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightMambaBandPLV(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_gcn_layers=2,
            fs=getattr(args, "fs", 200.0),
            gate_mode=gate_mode,
            topk=getattr(args, "top_k", 3),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None, raw_signal=None):
        if raw_signal is None:
            raise ValueError(
                "light_mamba_band_plv requires raw_signal kwarg "
                "(modify dataloader to return raw clip)."
            )
        x = input_seq.permute(0, 2, 1, 3).contiguous()                 # (B, N, T, D)
        logits, hidden = self.model(x, raw_signal)
        return logits, hidden
