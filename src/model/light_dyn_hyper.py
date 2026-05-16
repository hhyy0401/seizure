"""
`light_dyn_hyper` — Dynamic Hyperedges Driven by Mamba State.

See docs/light_dyn_hyper.md for the full pseudocode and motivation.

Replaces LightEvoBrain's (edge construction + GCN) block with a hypergraph
layer whose hyperedge queries Q_t evolve across timesteps, driven by a
global pooled summary of the Mamba node embeddings at time t.

Ablation flag `static_queries=True` collapses Q_t to a single static
parameter, which reproduces a SoftHGNN-style soft-membership hypergraph
without temporal evolution. This is the variant we name
`light_static_hyper`.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_backbone import BiMambaBackbone


class LinearBackbone(nn.Module):
    """Per-timestep Linear projection — no temporal or cross-channel mixing.

    Same input/output convention as `BiMambaBackbone`. Used when the downstream
    module (e.g. spatio-temporal hyperedge) is responsible for all temporal
    integration. Drop-in replacement to ablate the SSM backbone.
    """

    def __init__(self, d_input: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, d_input) → (B, T, N, d_model)
        h = self.proj(x)                                              # (B, N, T, d_model)
        return h.permute(0, 2, 1, 3).contiguous()


class InputInstanceNorm(nn.Module):
    """Per-recording, per-channel instance normalization.

    For input x of shape (B, N, T, D), normalizes over the (T, D) axes for each
    (B, N). Removes per-channel amplitude / baseline drift across patients so
    downstream prototypes see a more uniform distribution. Stateless, 0 params.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, D)
        mu = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (x - mu) / std


class AdaptiveSpatioTemporalHyperedgeBlock(nn.Module):
    """ST hyperedge with patient-adaptive prototype via cross-attention.

    Replaces the static `Q ∈ (E_h, d)` parameter with a sample-specific
        Q_b = softmax(Q_seeds @ K^T / √d) @ V    where K, V are projections of H
    so prototypes are constructed from the current sample's own tokens.
    Everything after Q_b (sigmoid assignment, ST pool, broadcast, FFN) is
    identical to `SpatioTemporalHyperedgeBlock`.
    """

    def __init__(self, d_in: int, d_out: int, n_hyperedges: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.E_h = n_hyperedges

        self.Q_seeds = nn.Parameter(torch.randn(n_hyperedges, d_in) * 0.02)
        self.q_proj = nn.Linear(d_in, d_in)
        self.k_proj = nn.Linear(d_in, d_in)
        self.v_proj = nn.Linear(d_in, d_in)

        self.out_proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.res_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, H_seq: torch.Tensor):
        B, T, N, d = H_seq.shape
        H_tok = H_seq.reshape(B, T * N, d)

        Q0 = self.q_proj(self.Q_seeds).unsqueeze(0).expand(B, -1, -1)   # (B, E_h, d)
        K = self.k_proj(H_tok)                                          # (B, T*N, d)
        V = self.v_proj(H_tok)                                          # (B, T*N, d)
        attn = (Q0 @ K.transpose(-2, -1)) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)                              # (B, E_h, T*N)
        Q_b = attn @ V                                                  # (B, E_h, d)

        logits = torch.einsum("btnd,bed->btne", H_seq, Q_b) / (d ** 0.5)
        M = torch.sigmoid(logits)                                       # (B, T, N, E_h)
        w_sum = M.sum(dim=(1, 2)).clamp_min(1e-6).unsqueeze(-1)         # (B, E_h, 1)
        h_edge = torch.einsum("btne,btnd->bed", M, H_seq) / w_sum       # (B, E_h, d)
        H_upd = torch.einsum("btne,bed->btnd", M, h_edge)               # (B, T, N, d)

        out = self.out_proj(F.gelu(H_upd))
        out = self.norm(out + self.res_proj(H_seq))
        return out, M


class IterativeSpatioTemporalHyperedgeBlock(nn.Module):
    """ST hyperedge with patient-adaptive prototype via soft k-means (EM).

    Q_b is initialized from learnable seeds, then refined for `n_iters`
    iterations of (E) assign + (M) re-center on the sample's own H. The
    final M and Q_b are used for the broadcast / FFN. 0 extra params over
    the static block (only Q_seeds, like before). Orthogonal seed init
    discourages mode collapse since the assignment uses sigmoid (no
    competition across hyperedges).
    """

    def __init__(self, d_in: int, d_out: int, n_hyperedges: int, n_iters: int = 2):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.E_h = n_hyperedges
        self.K = n_iters

        Q = torch.empty(n_hyperedges, d_in)
        nn.init.orthogonal_(Q, gain=0.5)
        self.Q_seeds = nn.Parameter(Q)

        self.out_proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.res_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, H_seq: torch.Tensor):
        B, T, N, d = H_seq.shape

        Q_b = self.Q_seeds.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, E_h, d)
        M = None
        for _ in range(self.K):
            logits = torch.einsum("btnd,bed->btne", H_seq, Q_b) / (d ** 0.5)
            M = torch.sigmoid(logits)                                   # (B, T, N, E_h)
            w_sum = M.sum(dim=(1, 2)).clamp_min(1e-6).unsqueeze(-1)     # (B, E_h, 1)
            Q_b = torch.einsum("btne,btnd->bed", M, H_seq) / w_sum      # (B, E_h, d)

        H_upd = torch.einsum("btne,bed->btnd", M, Q_b)                  # (B, T, N, d)

        out = self.out_proj(F.gelu(H_upd))
        out = self.norm(out + self.res_proj(H_seq))
        return out, M


class HyperedgeBlock(nn.Module):
    """One layer of (dynamic) hypergraph convolution.

    Input:
        H_seq: (B, T, N, d) — node embeddings over time.
    Output:
        H_seq': (B, T, N, d_out) — residual-updated node embeddings.
        M_seq:  (B, T, N, E_h)  — soft membership (returned for regularization
                                  or inspection).
    """

    def __init__(self, d_in: int, d_out: int,
                 n_hyperedges: int, static_queries: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.E_h = n_hyperedges
        self.static_queries = static_queries

        if static_queries:
            # Single learned set of hyperedge queries (no time dependence).
            self.Q_static = nn.Parameter(torch.randn(n_hyperedges, d_in) * 0.02)
        else:
            # Time-conditioned queries: pool channel embeddings at time t,
            # produce E_h × d_in queries.
            self.query_mlp = nn.Sequential(
                nn.Linear(d_in, 2 * d_in),
                nn.GELU(),
                nn.Linear(2 * d_in, n_hyperedges * d_in),
            )

        # Output projection on the aggregated node update.
        self.out_proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        # Residual path: if dims mismatch, project; else identity.
        if d_in != d_out:
            self.res_proj = nn.Linear(d_in, d_out)
        else:
            self.res_proj = nn.Identity()

    def forward(self, H_seq: torch.Tensor):
        B, T, N, d = H_seq.shape
        E_h = self.E_h

        # Build per-timestep hyperedge queries Q_seq: (B, T, E_h, d).
        if self.static_queries:
            Q_seq = self.Q_static.unsqueeze(0).unsqueeze(0).expand(B, T, E_h, d)
        else:
            g_seq = H_seq.mean(dim=2)                                 # (B, T, d)
            Q_seq = self.query_mlp(g_seq).view(B, T, E_h, d)          # (B, T, E_h, d)

        # Soft membership M[b, t, n, e] = sigmoid(<H_t[n], Q_t[e]> / sqrt(d))
        logits = torch.einsum("btnd,bted->btne", H_seq, Q_seq) / (d ** 0.5)
        M = torch.sigmoid(logits)                                      # (B, T, N, E_h)

        # Hyperedge embedding: weighted mean of member nodes.
        w_sum = M.sum(dim=2).clamp_min(1e-6).unsqueeze(-1)             # (B, T, E_h, 1)
        h_edge = torch.einsum("btne,btnd->bted", M, H_seq) / w_sum     # (B, T, E_h, d)

        # Node update: pull from incident hyperedges.
        H_upd = torch.einsum("btne,bted->btnd", M, h_edge)             # (B, T, N, d)

        out = self.out_proj(H_upd)
        out = F.gelu(out)
        out = self.norm(out + self.res_proj(H_seq))                    # residual
        return out, M


class SpatioTemporalHyperedgeBlock(nn.Module):
    """Hyperedge as a (channel, time) entity.

    Differs from `HyperedgeBlock` in the pooling axis:
        spatial-only:    h_e[t] = Σ_n M[t,n,e] * H[t,n]    →  (B, T, E_h, d)
        spatio-temporal: h_e    = Σ_{n,t} M[t,n,e] * H[t,n] →  (B, E_h, d)

    Hyperedge `e` thus represents one spatio-temporal pattern (e.g. a seizure
    spread trajectory: ch3 at t=2 → ch7 at t=3 → ch14 at t=4) as a single
    atomic primitive. Node update broadcasts this time-collapsed hyperedge
    embedding back to every (t, n) position.

    Args:
        d_in, d_out: feature dims.
        n_hyperedges: number of hyperedges E_h.
        share_query_across_time: if True, use a single learnable Q ∈ (E_h, d).
            (Time-conditioned Q is not used here — the spatio-temporal pooling
            is what carries the temporal information into the hyperedge.)
    """

    def __init__(self, d_in: int, d_out: int, n_hyperedges: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.E_h = n_hyperedges

        # Single static query set; the spatio-temporal pooling does the time
        # mixing, so a time-conditioned query is redundant here.
        self.Q = nn.Parameter(torch.randn(n_hyperedges, d_in) * 0.02)

        self.out_proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        if d_in != d_out:
            self.res_proj = nn.Linear(d_in, d_out)
        else:
            self.res_proj = nn.Identity()

    def forward(self, H_seq: torch.Tensor):
        B, T, N, d = H_seq.shape
        E_h = self.E_h

        # Membership uses static query broadcast over time:
        #   M[b, t, n, e] = sigmoid(<H[b,t,n], Q[e]> / sqrt(d))
        logits = torch.einsum("btnd,ed->btne", H_seq, self.Q) / (d ** 0.5)
        M = torch.sigmoid(logits)                                      # (B, T, N, E_h)

        # Spatio-temporal hyperedge embedding: pool over (N, T).
        w_sum_st = M.sum(dim=(1, 2)).clamp_min(1e-6).unsqueeze(-1)     # (B, E_h, 1)
        h_edge_st = torch.einsum("btne,btnd->bed", M, H_seq) / w_sum_st  # (B, E_h, d)
        # Stash for optional aux heads:
        #   `last_h_edge`: (B, E_h, d) → per-edge BCE deep supervision
        #   `last_M`:      (B, T, N, E_h) → soft-assignment entropy reg
        self.last_h_edge = h_edge_st
        self.last_M = M

        # Broadcast time-collapsed hyperedge back to every (t, n).
        H_upd = torch.einsum("btne,bed->btnd", M, h_edge_st)           # (B, T, N, d)

        out = self.out_proj(H_upd)
        out = F.gelu(out)
        out = self.norm(out + self.res_proj(H_seq))
        return out, M


class LightDynHyper(nn.Module):
    """Per-channel Mamba → 2× HyperedgeBlock → PMA channel readout → FC."""

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
        static_queries: bool = False,
        n_pma_seeds: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model

        # --- Per-channel (Bi-)Mamba backbone ---
        self.backbone = BiMambaBackbone(
            d_input=d_input, d_model=d_model,
            n_layers=n_mamba_layers, bidirectional=bidirectional,
        )

        # --- Hypergraph layers ---
        layers = []
        for i in range(n_hyper_layers):
            d_in = d_model if i == 0 else d_hidden
            layers.append(HyperedgeBlock(d_in, d_hidden,
                                         n_hyperedges=n_hyperedges,
                                         static_queries=static_queries))
        self.hyper_layers = nn.ModuleList(layers)

        # --- PMA readout (Set Transformer style, learnable seed query) ---
        self.pma_seeds = nn.Parameter(torch.randn(n_pma_seeds, d_hidden) * 0.02)
        self.pma_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden * n_pma_seeds, n_classes)

    def node_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, d_input) → (B, T, N, d_model). Keep full time axis."""
        return self.backbone(x)

    def pma_readout(self, H_last: torch.Tensor) -> torch.Tensor:
        """H_last: (B, N, d) → (B, n_seeds * d) via learnable seed attention."""
        B, N, d = H_last.shape
        seeds = self.pma_seeds.unsqueeze(0).expand(B, -1, -1)          # (B, n_seeds, d)
        attn = torch.einsum("bsd,bnd->bsn", seeds, H_last) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bsn,bnd->bsd", attn, H_last)                # (B, n_seeds, d)
        out = self.pma_norm(out)
        return out.reshape(B, -1)

    def forward(self, x: torch.Tensor):
        # x: (B, N, T, d_input)
        H_seq = self.node_mamba(x)                                     # (B, T, N, d)
        for layer in self.hyper_layers:
            H_seq, _ = layer(H_seq)
        H_last = H_seq[:, -1, :, :]                                    # (B, N, d)
        z = self.pma_readout(H_last)                                   # (B, d * n_seeds)
        return self.classifier(self.dropout(z)), H_last


class LightDynHyper_classification(nn.Module):
    """main.py-compatible wrapper. Same call signature as LightEvoBrain_classification.

        logits, hidden = model(x, seq_lengths, adj)

    `x` from the EvoBrain CHB dataloader is (B, T, N, D); `adj` is ignored
    here because hyperedges are learned end-to-end from H, not from the
    correlation prior.
    """

    def __init__(self, args, num_classes,
                 device=None, static_queries: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightDynHyper(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_hyper_layers=getattr(args, "n_hyper_layers", 2),
            n_hyperedges=getattr(args, "n_hyperedges", 8),
            static_queries=static_queries,
            n_pma_seeds=getattr(args, "n_pma_seeds", 1),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        # (B, T, N, D) → (B, N, T, D)
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        logits, hidden = self.model(x)
        return logits, hidden


class MultiScaleSTHyperedgeBlock(nn.Module):
    """ST-hyperedge with per-edge learnable temporal receptive field.

    Each hyperedge `e` has its own temporal weighting `W[e, :] ∈ Δ(T)`
    (softmax over time) that biases which timesteps it pools from. Edges are
    initialized in K groups, each group seeded with a different temporal
    scope (short=onset, medium=propagation, long=generalization), letting
    the model express seizure dynamics at multiple time scales.
    """

    def __init__(self, d_in: int, d_out: int, n_hyperedges: int,
                 T_max: int = 12, scales=(2, 6, 12)):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.E_h = n_hyperedges
        self.T_max = T_max
        self.scales = scales

        self.Q = nn.Parameter(torch.randn(n_hyperedges, d_in) * 0.02)
        # Per-edge temporal mask logits (softmax along T).
        self.time_mask_logit = nn.Parameter(torch.zeros(n_hyperedges, T_max))

        # Seed: split edges into K groups, each centered at T/2 with σ=scale/2.
        K = len(scales)
        edges_per = n_hyperedges // K
        with torch.no_grad():
            for k, scale in enumerate(scales):
                start = k * edges_per
                end = n_hyperedges if k == K - 1 else (k + 1) * edges_per
                center = (T_max - 1) / 2.0
                sigma = max(scale / 2.0, 0.5)
                t_pos = torch.arange(T_max).float()
                init = -((t_pos - center) ** 2) / (2 * sigma ** 2)
                self.time_mask_logit[start:end] = init.unsqueeze(0)

        self.out_proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        if d_in != d_out:
            self.res_proj = nn.Linear(d_in, d_out)
        else:
            self.res_proj = nn.Identity()

    def forward(self, H_seq: torch.Tensor):
        B, T, N, d = H_seq.shape
        E_h = self.E_h

        logits = torch.einsum("btnd,ed->btne", H_seq, self.Q) / (d ** 0.5)
        M = torch.sigmoid(logits)                                      # (B, T, N, E_h)

        # Per-edge temporal weights (softmax over T) — pad/crop if T != T_max.
        W = torch.softmax(self.time_mask_logit[:, :T], dim=-1)         # (E_h, T)

        # Weighted spatio-temporal pool: each edge weights its own time slots.
        # M_weighted[b, t, n, e] = M[b, t, n, e] * W[e, t]
        M_w = M * W.t().unsqueeze(0).unsqueeze(2)                      # (B, T, N, E_h)

        w_sum_st = M_w.sum(dim=(1, 2)).clamp_min(1e-6).unsqueeze(-1)   # (B, E_h, 1)
        h_edge_st = torch.einsum("btne,btnd->bed", M_w, H_seq) / w_sum_st  # (B, E_h, d)

        H_upd = torch.einsum("btne,bed->btnd", M, h_edge_st)           # (B, T, N, d)

        out = self.out_proj(H_upd)
        out = F.gelu(out)
        out = self.norm(out + self.res_proj(H_seq))
        return out, M


class LightSTHyperMScale(nn.Module):
    """Per-channel backbone → 2× MultiScaleSTHyperedgeBlock → temporal-mean → PMA → FC."""

    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_hidden: int = 64,
        n_nodes: int = 22,
        n_classes: int = 1,
        n_mamba_layers: int = 2,
        n_hyper_layers: int = 2,
        n_hyperedges: int = 9,    # default divisible by len(scales)=3
        T_max: int = 12,
        scales=(2, 6, 12),
        n_pma_seeds: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model

        self.backbone = BiMambaBackbone(
            d_input=d_input, d_model=d_model,
            n_layers=n_mamba_layers, bidirectional=bidirectional,
        )

        layers = []
        for i in range(n_hyper_layers):
            d_in_i = d_model if i == 0 else d_hidden
            layers.append(MultiScaleSTHyperedgeBlock(
                d_in_i, d_hidden, n_hyperedges=n_hyperedges,
                T_max=T_max, scales=scales))
        self.hyper_layers = nn.ModuleList(layers)

        self.pma_seeds = nn.Parameter(torch.randn(n_pma_seeds, d_hidden) * 0.02)
        self.pma_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden * n_pma_seeds, n_classes)

    def pma_readout(self, H_pool: torch.Tensor) -> torch.Tensor:
        B, N, d = H_pool.shape
        seeds = self.pma_seeds.unsqueeze(0).expand(B, -1, -1)
        attn = torch.einsum("bsd,bnd->bsn", seeds, H_pool) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bsn,bnd->bsd", attn, H_pool)
        out = self.pma_norm(out)
        return out.reshape(B, -1)

    def forward(self, x: torch.Tensor):
        H_seq = self.backbone(x)                                       # (B, T, N, d)
        for layer in self.hyper_layers:
            H_seq, _ = layer(H_seq)
        H_pool = H_seq.mean(dim=1)                                     # (B, N, d)
        z = self.pma_readout(H_pool)
        return self.classifier(self.dropout(z)), H_pool


class LightSTHyperMScale_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightSTHyperMScale(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_hyper_layers=getattr(args, "n_hyper_layers", 2),
            n_hyperedges=getattr(args, "n_hyperedges", 9),
            T_max=getattr(args, "max_seq_len", 12),
            n_pma_seeds=getattr(args, "n_pma_seeds", 1),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        logits, hidden = self.model(x)
        return logits, hidden


class LightSTHyper(nn.Module):
    """Per-channel Mamba → 2× SpatioTemporalHyperedgeBlock → PMA readout → FC.

    Differs from `LightDynHyper` only in the hypergraph layer: hyperedges are
    spatio-temporal entities (pool over both channel and time) rather than
    per-timestep spatial groupings. The temporal axis is consumed inside the
    hypergraph module, so the readout uses the temporal-mean rather than the
    last frame.
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
        hyper_block_type: str = "static",
        use_input_norm: bool = False,
        n_iters: int = 2,
        use_node_emb: bool = False,
        timesnet_k: int = 2,
        aux_type: str = "none",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.backbone_type = backbone_type
        assert aux_type in ("none", "bce", "entropy")
        self.aux_type = aux_type

        self.input_norm = InputInstanceNorm() if use_input_norm else nn.Identity()

        # Backbone ablation: mamba (main) / linear / dwsep
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
            raise ValueError(f"Unknown backbone_type: {backbone_type} "
                             f"(supported: mamba, linear, dwsep)")

        self.use_node_emb = use_node_emb
        if use_node_emb:
            self.node_emb = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)

        def make_block(d_in, d_out):
            if hyper_block_type == "static":
                return SpatioTemporalHyperedgeBlock(d_in, d_out, n_hyperedges=n_hyperedges)
            if hyper_block_type == "adaptive":
                return AdaptiveSpatioTemporalHyperedgeBlock(d_in, d_out, n_hyperedges=n_hyperedges)
            if hyper_block_type == "iterative":
                return IterativeSpatioTemporalHyperedgeBlock(d_in, d_out, n_hyperedges=n_hyperedges, n_iters=n_iters)
            raise ValueError(f"Unknown hyper_block_type: {hyper_block_type}")

        layers = []
        for i in range(n_hyper_layers):
            d_in = d_model if i == 0 else d_hidden
            layers.append(make_block(d_in, d_hidden))
        self.hyper_layers = nn.ModuleList(layers)

        self.pma_seeds = nn.Parameter(torch.randn(n_pma_seeds, d_hidden) * 0.02)
        self.pma_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden * n_pma_seeds, n_classes)

        # Per-edge classifier used only for aux_type == "bce".
        self.aux_classifier = (
            nn.Linear(d_hidden, n_classes) if aux_type == "bce" else None
        )

    def compute_aux_loss(self, y: torch.Tensor) -> torch.Tensor:
        """Auxiliary regularizer on the LAST hypergraph layer.
            aux_type == "bce":     per-edge BCE deep-supervision (uses h_edge)
            aux_type == "entropy": minimize per-(t,n) entropy over edge
                                   memberships, encouraging hard assignment
                                   (uses M)
            aux_type == "none":    0-scalar (no-op)
        """
        if self.aux_type == "none":
            return y.new_zeros(())
        last = self.hyper_layers[-1]
        if self.aux_type == "bce":
            if self.aux_classifier is None or not hasattr(last, "last_h_edge"):
                return y.new_zeros(())
            h_edge = last.last_h_edge                   # (B, E_h, d)
            edge_logits = self.aux_classifier(h_edge)   # (B, E_h, n_classes)
            if edge_logits.shape[-1] == 1:
                edge_logits = edge_logits.squeeze(-1)
                y_b = y.float().unsqueeze(-1).expand_as(edge_logits)
                return F.binary_cross_entropy_with_logits(edge_logits, y_b)
            B, E_h, C = edge_logits.shape
            return F.cross_entropy(
                edge_logits.reshape(B * E_h, C),
                y.long().unsqueeze(-1).expand(B, E_h).reshape(B * E_h),
            )
        if self.aux_type == "entropy":
            if not hasattr(last, "last_M"):
                return y.new_zeros(())
            M = last.last_M                             # (B, T, N, E_h)
            # Normalize over edge axis → soft assignment per (t, n).
            p = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)   # (B, T, N)
            return ent.mean()
        return y.new_zeros(())

    def pma_readout(self, H_pool: torch.Tensor) -> torch.Tensor:
        # H_pool: (B, N, d) → (B, n_seeds * d)
        B, N, d = H_pool.shape
        seeds = self.pma_seeds.unsqueeze(0).expand(B, -1, -1)
        attn = torch.einsum("bsd,bnd->bsn", seeds, H_pool) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bsn,bnd->bsd", attn, H_pool)
        out = self.pma_norm(out)
        return out.reshape(B, -1)

    def forward(self, x: torch.Tensor):
        # x: (B, N, T, d_input) → H: (B, T, N, d)
        x = self.input_norm(x)
        H_seq = self.backbone(x)
        if self.use_node_emb:
            # H_seq: (B, T, N, d); node_emb: (N, d) → broadcast on B, T.
            H_seq = H_seq + self.node_emb.unsqueeze(0).unsqueeze(0)
        for layer in self.hyper_layers:
            H_seq, _ = layer(H_seq)
        # Time has been mixed inside the hyperedge — use temporal mean
        # rather than last-frame to keep the full-clip integration.
        H_pool = H_seq.mean(dim=1)                                     # (B, N, d)
        z = self.pma_readout(H_pool)
        return self.classifier(self.dropout(z)), H_pool


class LightSTHyper_classification(nn.Module):
    """main.py-compatible wrapper, same signature as LightDynHyper_classification."""

    def __init__(self, args, num_classes, device=None,
                 backbone_type: str = "mamba",
                 hyper_block_type: str = "static",
                 use_input_norm: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightSTHyper(
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
            hyper_block_type=hyper_block_type,
            use_input_norm=use_input_norm,
            n_iters=getattr(args, "n_iters", 2),
            use_node_emb=getattr(args, "use_node_emb", False),
            timesnet_k=getattr(args, "timesnet_k", 2),
            aux_type=getattr(args, "aux_type", "none"),
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        logits, hidden = self.model(x)
        return logits, hidden

    def compute_aux_loss(self, y):
        return self.model.compute_aux_loss(y)
