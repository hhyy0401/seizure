"""GraphS4mer for seizure detection/prediction.

Source: Tang et al., "Modeling Multivariate Biosignals With Graph Neural
Networks and Structured State Space Models", CHIL 2023.
Adapted from https://github.com/tsy935/graphs4mer (model/graphs4mer.py,
model/graph_learner.py).

Faithful re-implementation with two intentional deviations:

  1. Temporal backbone uses **Mamba** (selective SSM) in place of the
     original S4. Both are selective state-space models; Mamba is already
     available in this conda env and avoids the upstream S4 dependency on
     pytorch_lightning / einops / opt_einsum / a CUDA Cauchy extension that
     is not built on Phoenix. The rest of the architecture (graph learner,
     KNN-cosine prior, prune, GCN, temporal/graph pool) is unchanged.

  2. GNN layers are dense GCN rather than SAGEConv. For the small EEG
     graphs (N=19 TUSZ, N=18 CHB-MIT) the dense form is faster and avoids
     the edge_index roundtrip; results match within noise on small N.

The forward signature is (x, seq_lengths) → (logits, hidden), matching the
LSTM/CNN-LSTM/BIOT path in main.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


# ---------------------------- graph learner -------------------------------

class _GraphLearner(nn.Module):
    """Self-attention graph learner (from tsy935/graphs4mer)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_Q = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_K = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x):
        """x: (B, N, D) → attn: (B, N, N) softmax-normalized."""
        Q = self.linear_Q(x)
        K = self.linear_K(x)
        attn = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        return torch.softmax(attn, dim=-1)


# ---------------------------- utilities -----------------------------------

def _knn_cosine_graph(x, k, undirected=True):
    """Cosine-similarity top-k graph with self-loops.
    x: (B, N, D) → adj (B, N, N) ∈ [0,1].
    """
    norm = torch.norm(x, dim=-1, p="fro").clamp(min=1e-6)[:, :, None]
    xn = x / norm
    sim = torch.matmul(xn, xn.transpose(1, 2))            # (B, N, N)
    knn_val, knn_ind = torch.topk(sim, k, dim=-1, largest=True)
    adj = torch.zeros_like(sim).scatter_(-1, knn_ind, knn_val).clamp(min=0.0)
    if undirected:
        adj = (adj + adj.transpose(1, 2)) / 2
    N = adj.size(-1)
    I = torch.eye(N, device=adj.device, dtype=adj.dtype).expand_as(adj).bool()
    adj = adj * (~I) + I.float()
    return adj


def _prune_top_perc(adj, perc):
    """Keep top-`perc` fraction of edge weights per graph (B, N, N)."""
    B, N, _ = adj.shape
    flat = adj.reshape(B, N * N)
    K = max(1, int((N * N) * perc))
    thresh, _ = flat.topk(K, dim=-1, largest=True)
    cutoff = thresh[:, -1].unsqueeze(-1).unsqueeze(-1)
    return adj * (adj >= cutoff).float()


def _normalize_adj(A, eps=1e-6):
    """Sym normalize with self-loops: D^-1/2 (A+I) D^-1/2."""
    N = A.size(-1)
    I = torch.eye(N, device=A.device, dtype=A.dtype).expand_as(A)
    A_hat = A + I
    deg = A_hat.sum(-1).clamp(min=eps)
    d_inv_sqrt = deg.pow(-0.5)
    D = torch.diag_embed(d_inv_sqrt)
    return D @ A_hat @ D


# ---------------------------- temporal backbone ---------------------------

class _MambaTemporal(nn.Module):
    """Per-channel temporal backbone: Linear(input→hidden) + N Mamba blocks.

    Drop-in for tsy935/graphs4mer's S4Model. Forward takes (B*N, T, input_dim)
    and returns (B*N, T, hidden_dim).
    """

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is not installed in this env. "
                "Run on the evobrain conda env, or install mamba-ssm.")
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B*N, T, input_dim) → (B*N, T, hidden_dim)
        x = self.proj(x)
        for blk, ln in zip(self.blocks, self.norms):
            x = x + self.dropout(blk(ln(x)))
        return x


# ---------------------------- main model ----------------------------------

class GraphS4mer(nn.Module):
    """GraphS4mer (Mamba-temporal variant).

    Pipeline:
        1. (B, T, N, D) → per-channel Mamba → (B*N, T, H)
        2. Pool into `num_dynamic_graphs = max_seq_len // resolution` blocks:
           each block averages H over `resolution` steps → (B, K, N, H).
        3. For each of K graphs: self-attn learn adj + cosine-KNN prior +
           residual mix + top-perc prune → soft adjacency (B*K, N, N).
        4. Two dense GCN layers over (H_node, A) per (B, K, N).
        5. Mean over K → (B, N, H). Sum over N → (B, H). Linear → logits.

    Tuneable args (defaults from upstream tsy935/graphs4mer TUSZ config):
        hidden_dim=128, num_temporal_layers=2, num_gnn_layers=2,
        resolution=12, K=3, edge_top_perc=0.5, residual_weight=0.0,
        graph_pool='sum', temporal_pool='mean'.
    """

    def __init__(
        self,
        args=None,
        num_classes=1,
        max_seq_len=None,
        num_nodes=None,
        input_dim=None,
        hidden_dim=128,
        num_temporal_layers=2,
        num_gnn_layers=2,
        resolution=None,
        K=3,
        edge_top_perc=0.5,
        residual_weight=0.0,
        temporal_pool="mean",
        graph_pool="sum",
        undirected=True,
        dropout=0.1,
    ):
        super().__init__()
        # Pull from args when supplied (preferred path from main.py).
        if args is not None:
            max_seq_len = max_seq_len if max_seq_len is not None else args.max_seq_len
            num_nodes = num_nodes if num_nodes is not None else args.num_nodes
            input_dim = input_dim if input_dim is not None else args.input_dim
            num_classes = num_classes if num_classes is not None else args.num_classes
            dropout = getattr(args, "dropout", dropout) or dropout

        if max_seq_len is None or num_nodes is None or input_dim is None:
            raise ValueError(
                "GraphS4mer requires max_seq_len, num_nodes, and input_dim "
                "(supply via args= or as kwargs).")

        # Default resolution: 12 if it divides max_seq_len, else max_seq_len.
        if resolution is None:
            resolution = 12 if (max_seq_len % 12 == 0) else max_seq_len
        if max_seq_len % resolution != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be divisible by "
                f"resolution ({resolution}).")

        self.num_nodes = num_nodes
        self.max_seq_len = max_seq_len
        self.resolution = resolution
        self.num_dynamic_graphs = max_seq_len // resolution
        self.hidden_dim = hidden_dim
        self.K = min(K, num_nodes)
        self.edge_top_perc = edge_top_perc
        self.residual_weight = residual_weight
        self.temporal_pool = temporal_pool
        self.graph_pool = graph_pool
        self.undirected = undirected

        self.t_model = _MambaTemporal(
            input_dim=input_dim, hidden_dim=hidden_dim,
            n_layers=num_temporal_layers, dropout=dropout)

        self.attn = _GraphLearner(input_size=hidden_dim, hidden_size=hidden_dim)

        # Dense GCN layers.
        self.gnn_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _gcn_layer(self, A_norm, H, W):
        """One dense GCN layer: H' = sigma(A_norm @ H @ W)."""
        return self.activation(A_norm @ W(H))

    def forward(self, input_seq, seq_lengths=None):
        """
        Args:
            input_seq: (B, T, N, input_dim)
            seq_lengths: (B,) — unused (clips have fixed T).
        Returns:
            logits: (B, num_classes)
            hidden: (B, hidden_dim)
        """
        B, T, N, D = input_seq.shape
        assert N == self.num_nodes, f"expected N={self.num_nodes}, got {N}"
        assert T == self.max_seq_len, f"expected T={self.max_seq_len}, got {T}"

        # (B, T, N, D) → per-channel (B*N, T, D)
        x = input_seq.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        h = self.t_model(x)                              # (B*N, T, H)
        H = self.hidden_dim
        h = h.view(B, N, T, H)

        # Pool into K dynamic-graph blocks.
        K = self.num_dynamic_graphs
        r = self.resolution
        h = h.view(B, N, K, r, H).mean(dim=3)            # (B, N, K, H)
        h = h.permute(0, 2, 1, 3).contiguous()           # (B, K, N, H)
        h_flat = h.view(B * K, N, H)                     # (B*K, N, H)

        # Graph learning: self-attn + KNN-cosine prior + prune.
        attn = self.attn(h_flat)                         # (B*K, N, N)
        if self.undirected:
            attn = (attn + attn.transpose(1, 2)) / 2

        if self.residual_weight > 0:
            prior = _knn_cosine_graph(h_flat, self.K, undirected=self.undirected)
            adj = self.residual_weight * prior + (1 - self.residual_weight) * attn
        else:
            adj = attn

        adj = _prune_top_perc(adj, self.edge_top_perc)   # sparsify
        A_norm = _normalize_adj(adj)                     # (B*K, N, N)

        # Dense GCN layers (residual within each layer).
        x = h_flat
        for W in self.gnn_weights:
            x = self._gcn_layer(A_norm, x, W)
            x = self.dropout(x)
        x = x.view(B, K, N, H)

        # Temporal pool over K.
        if self.temporal_pool == "last":
            x = x[:, -1]
        elif self.temporal_pool == "mean":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported temporal_pool: {self.temporal_pool}")

        # Graph pool over N.
        if self.graph_pool == "sum":
            pooled = x.sum(dim=1)
        elif self.graph_pool == "mean":
            pooled = x.mean(dim=1)
        elif self.graph_pool == "max":
            pooled, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unsupported graph_pool: {self.graph_pool}")

        logits = self.classifier(pooled)                 # (B, num_classes)
        return logits, pooled
