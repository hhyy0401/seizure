"""
`light_dense_hyper` — Per-second (seq2seq) seizure detection.

Uses the **per-timestep `HyperedgeBlock`** (time-conditioned queries)
rather than the spatio-temporal hyperedge block, because for per-second
output we want each t to form its own hyperedges from the channel state
at that moment. This makes the per-second logit a function of "what
hyperedge pattern is active at t" rather than a global clip pattern
broadcast back.

Differences vs `LightSTHyper` (the clip-level variant):
  1. Hyperedge block = `HyperedgeBlock` (per-t Q, per-t hyperedge embedding),
     NOT `SpatioTemporalHyperedgeBlock` (which collapses time).
  2. No temporal mean before readout — keep (B, T, N, d).
  3. PMA over channels is applied per-timestep (parameters shared across t).
  4. Aux head (aux_type='bce') uses h_edge that is (B, T, E_h, d), so we
     do per-(t, e) BCE against the per-second label y[b, t].
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_backbone import BiMambaBackbone
from model.light_dyn_hyper import (
    HyperedgeBlock,
    InputInstanceNorm,
    LinearBackbone,
)


class DenseHyperedgeBlock(HyperedgeBlock):
    """Same as `HyperedgeBlock`, but stashes the per-t hyperedge embedding
    so the dense aux head can run per-(t, e) BCE supervision.

    Stashed attribute:
        last_h_edge: (B, T, E_h, d)
    """

    def forward(self, H_seq: torch.Tensor):
        B, T, N, d = H_seq.shape
        E_h = self.E_h

        if self.static_queries:
            Q_seq = self.Q_static.unsqueeze(0).unsqueeze(0).expand(B, T, E_h, d)
        else:
            g_seq = H_seq.mean(dim=2)                                    # (B, T, d)
            Q_seq = self.query_mlp(g_seq).view(B, T, E_h, d)             # (B, T, E_h, d)

        logits = torch.einsum("btnd,bted->btne", H_seq, Q_seq) / (d ** 0.5)
        M = torch.sigmoid(logits)                                        # (B, T, N, E_h)

        w_sum = M.sum(dim=2).clamp_min(1e-6).unsqueeze(-1)               # (B, T, E_h, 1)
        h_edge = torch.einsum("btne,btnd->bted", M, H_seq) / w_sum       # (B, T, E_h, d)
        # Stash per-t hyperedge embedding for dense aux head.
        self.last_h_edge = h_edge
        self.last_M = M

        H_upd = torch.einsum("btne,bted->btnd", M, h_edge)               # (B, T, N, d)

        out = self.out_proj(H_upd)
        out = F.gelu(out)
        out = self.norm(out + self.res_proj(H_seq))
        return out, M


class DenseLightHyper(nn.Module):
    """Per-channel Mamba -> 2x per-t HyperedgeBlock -> per-t PMA -> per-t FC.

    Forward output:
        logits: (B, T, n_classes)
        H_seq:  (B, T, N, d)
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_hidden: int = 64,
        n_nodes: int = 22,
        n_classes: int = 1,
        n_mamba_layers: int = 2,
        n_hyper_layers: int = 2,
        n_hyperedges: int = 8,
        n_pma_seeds: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        backbone_type: str = "mamba",
        static_queries: bool = False,
        use_input_norm: bool = False,
        use_node_emb: bool = False,
        aux_type: str = "none",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.backbone_type = backbone_type
        assert aux_type in ("none", "bce", "entropy")
        self.aux_type = aux_type

        self.input_norm = InputInstanceNorm() if use_input_norm else nn.Identity()

        if backbone_type == "mamba":
            self.backbone = BiMambaBackbone(
                d_input=d_input, d_model=d_model,
                n_layers=n_mamba_layers, bidirectional=bidirectional,
            )
        elif backbone_type == "linear":
            self.backbone = LinearBackbone(d_input=d_input, d_model=d_model)
        elif backbone_type == "dwsep":
            from model.temporal_backbones import DepthwiseSeparable1DBackbone
            self.backbone = DepthwiseSeparable1DBackbone(
                d_input=d_input, d_model=d_model, n_layers=n_mamba_layers)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        self.use_node_emb = use_node_emb
        if use_node_emb:
            self.node_emb = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)

        layers = []
        for i in range(n_hyper_layers):
            d_in = d_model if i == 0 else d_hidden
            layers.append(DenseHyperedgeBlock(d_in, d_hidden,
                                              n_hyperedges=n_hyperedges,
                                              static_queries=static_queries))
        self.hyper_layers = nn.ModuleList(layers)

        # PMA readout, applied per-timestep (parameters time-shared).
        self.pma_seeds = nn.Parameter(torch.randn(n_pma_seeds, d_hidden) * 0.02)
        self.pma_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden * n_pma_seeds, n_classes)

        self.aux_classifier = (
            nn.Linear(d_hidden, n_classes) if aux_type == "bce" else None
        )

    def pma_readout_per_t(self, H_seq: torch.Tensor) -> torch.Tensor:
        """Per-timestep PMA. (B, T, N, d) -> (B, T, n_seeds * d)."""
        B, T, N, d = H_seq.shape
        s = self.pma_seeds.shape[0]
        H_bt = H_seq.reshape(B * T, N, d)
        seeds = self.pma_seeds.unsqueeze(0).expand(B * T, -1, -1)
        attn = torch.einsum("bsd,bnd->bsn", seeds, H_bt) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bsn,bnd->bsd", attn, H_bt)
        out = self.pma_norm(out)
        return out.reshape(B, T, s * d)

    def compute_aux_loss(self, y: torch.Tensor) -> torch.Tensor:
        """Per-(t, e) BCE deep supervision on last hypergraph layer."""
        if self.aux_type == "none":
            return y.new_zeros(()).float()
        last = self.hyper_layers[-1]
        if self.aux_type == "bce":
            if self.aux_classifier is None or not hasattr(last, "last_h_edge"):
                return y.new_zeros(()).float()
            h_edge = last.last_h_edge                                    # (B, T, E_h, d)
            edge_logits = self.aux_classifier(h_edge)                    # (B, T, E_h, n_cls)
            if y.dim() == 1:
                y_t = y.float().unsqueeze(1).expand(-1, edge_logits.size(1))
            else:
                y_t = y.float()
            if edge_logits.shape[-1] == 1:
                edge_logits = edge_logits.squeeze(-1)                    # (B, T, E_h)
                y_te = y_t.unsqueeze(-1).expand_as(edge_logits)
                return F.binary_cross_entropy_with_logits(edge_logits, y_te)
            B, T, E_h, C = edge_logits.shape
            return F.cross_entropy(
                edge_logits.reshape(B * T * E_h, C),
                y_t.long().unsqueeze(-1).expand(B, T, E_h).reshape(B * T * E_h),
            )
        if self.aux_type == "entropy":
            if not hasattr(last, "last_M"):
                return y.new_zeros(()).float()
            M = last.last_M
            p = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
            return ent.mean()
        return y.new_zeros(()).float()

    def forward(self, x: torch.Tensor):
        # x: (B, N, T, d_input)
        x = self.input_norm(x)
        H_seq = self.backbone(x)                                         # (B, T, N, d)
        if self.use_node_emb:
            H_seq = H_seq + self.node_emb.unsqueeze(0).unsqueeze(0)
        for layer in self.hyper_layers:
            H_seq, _ = layer(H_seq)
        # No temporal pooling — keep per-t representation.
        z_seq = self.pma_readout_per_t(H_seq)                            # (B, T, s*d)
        logits = self.classifier(self.dropout(z_seq))                    # (B, T, n_classes)
        return logits, H_seq


class DenseLightHyper_classification(nn.Module):
    """main.py-compatible wrapper."""

    def __init__(self, args, num_classes, device=None,
                 backbone_type: str = "mamba",
                 static_queries: bool = False,
                 use_input_norm: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = DenseLightHyper(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_hyper_layers=getattr(args, "n_hyper_layers", 2),
            n_hyperedges=getattr(args, "n_hyperedges", 8),
            n_pma_seeds=getattr(args, "n_pma_seeds", 1),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
            backbone_type=backbone_type,
            static_queries=static_queries,
            use_input_norm=use_input_norm,
            use_node_emb=getattr(args, "use_node_emb", False),
            aux_type=getattr(args, "aux_type", "none"),
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        # (B, T, N, D) -> (B, N, T, D)
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        logits, hidden = self.model(x)
        return logits, hidden

    def compute_aux_loss(self, y):
        return self.model.compute_aux_loss(y)
