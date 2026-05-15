"""
Lightweight EvoBrain variants per docs/light_evobrain_pseudocode.md.

Architecture (same backbone for all 3 variants):
    STFT features ──► Node Mamba (per-channel sequence) ──► H (B, N, d)
                                                           │
              Learnable Edge (3 variants) ────────────────►│ A (B, N, N)
                                                           │
                                              GCN(2-layer) │
                                                           ▼
                                    Max pool → FC → logits (B, C)

What is dropped vs EvoBrain:
    * dynamic correlation graph construction ("Step 1"  in the user's spec)
    * edge stream Mamba
    * Laplacian Positional Encoding

What is added:
    * one of three learnable edge-weight schemes — selected via `edge_type`:
        - 'dot'       : softmax(H Hᵀ)
        - 'bilinear'  : softmax(H W Hᵀ),  W ∈ R^{d×d} learnable
        - 'attention' : multi-head scaled-dot-product attention, averaged over heads
"""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


EdgeType = Literal["dot", "bilinear", "attention"]


class LightEvoBrain(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_gcn: int = 64,
        n_nodes: int = 18,
        n_classes: int = 1,
        n_mamba_layers: int = 2,
        n_gcn_layers: int = 2,
        edge_type: EdgeType = "dot",
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.edge_type = edge_type
        self.d_model = d_model

        # --- Node Mamba ---
        self.input_proj = nn.Linear(d_input, d_model)
        self.mamba_layers = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )

        # --- Learnable Edge ---
        if edge_type == "bilinear":
            self.W_edge = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        elif edge_type == "attention":
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
        elif edge_type != "dot":
            raise ValueError(f"Unknown edge_type: {edge_type}")

        # --- GCN (simple matrix form: H ← ReLU(Linear(A H)))  ---
        self.gcn_layers = nn.ModuleList()
        for i in range(n_gcn_layers):
            in_dim = d_model if i == 0 else d_gcn
            self.gcn_layers.append(nn.Linear(in_dim, d_gcn))

        # --- Classifier (after max pool over nodes) ---
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_gcn, n_classes)

    # ------------------------------------------------------------------

    def node_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, d_input) → (B, N, d_model). Take the last time step."""
        B, N, T, D = x.shape
        x = x.reshape(B * N, T, D)
        x = self.input_proj(x)
        for mamba in self.mamba_layers:
            x = mamba(x)
        h = x[:, -1, :]
        return h.view(B, N, -1)

    def compute_edge(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, N, d_model) → A: (B, N, N) with rows summing to 1."""
        if self.edge_type == "dot":
            A = torch.bmm(H, H.transpose(1, 2))
            A = F.softmax(A, dim=-1)
        elif self.edge_type == "bilinear":
            HW = torch.matmul(H, self.W_edge)
            A = torch.bmm(HW, H.transpose(1, 2))
            A = F.softmax(A, dim=-1)
        else:  # attention
            B, N, _ = H.shape
            Q = self.W_q(H).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            K = self.W_k(H).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            attn = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_head ** 0.5)
            attn = F.softmax(attn, dim=-1)
            A = attn.mean(dim=1)
        return A

    def gcn_forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        for layer in self.gcn_layers:
            H = torch.bmm(A, H)
            H = layer(H)
            H = F.relu(H)
        return H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.node_mamba(x)            # (B, N, d_model)
        A = self.compute_edge(H)           # (B, N, N)
        H = self.gcn_forward(H, A)         # (B, N, d_gcn)
        z = H.max(dim=1)[0]                # (B, d_gcn)
        return self.classifier(self.dropout(z))  # (B, n_classes)


class LightEvoBrain_classification(nn.Module):
    """Adapter that matches the call signature used in main.py:

        logits, hidden = model(x, seq_lengths, adj)

    where `x` from the EvoBrain dataloader is (B, T, N, D) and `adj` is ignored.
    """

    def __init__(self, args, num_classes, device=None, edge_type: EdgeType = "dot"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightEvoBrain(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_gcn=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_gcn_layers=2,
            edge_type=edge_type,
            n_heads=4,
            dropout=args.dropout,
        )

    def forward(self, input_seq, seq_lengths, adj_unused):
        # (B, T, N, D) -> (B, N, T, D)
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        B, N, T, D = x.shape
        # Inline expansion of LightEvoBrain.forward so we can expose the
        # pre-classifier features as `hidden` (main.py's evaluate() does
        # hidden.cpu().reshape(B, -1), so None breaks it).
        H = self.model.node_mamba(x)
        A = self.model.compute_edge(H)
        H = self.model.gcn_forward(H, A)             # (B, N, d_gcn)
        z = H.max(dim=1)[0]                          # (B, d_gcn)
        logits = self.model.classifier(self.model.dropout(z))
        return logits, H
