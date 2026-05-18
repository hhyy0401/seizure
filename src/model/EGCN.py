"""EvolveGCN-O for seizure detection/prediction.

Source: Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for
Dynamic Graphs", AAAI 2020. Code adapted verbatim from
https://github.com/IBM/EvolveGCN (egcn_o.py).

The original EvolveGCN forward processes a list-of-snapshots
(A_list, Nodes_list) where each snapshot is a single graph (N, D). Our
dataloader produces dense batched inputs:

    x   : (B, T, N, input_dim)        node features per (batch, time, channel)
    adj : (B, T, N, N)                dynamic adjacency per (batch, time)

The original ``Ahat.matmul(node_embs.matmul(GCN_weights))`` already broadcasts
when ``Ahat`` is (B, N, N) and ``node_embs`` is (B, N, D), so the per-timestep
loop is preserved unchanged — we just slice by time.

EvolveGCN-O variant (W_t = GRU(W_{t-1})) is used; the -H variant requires a
TopK over node embeddings which is fragile for our 19-channel EEG setup.
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List


# ---- helpers inlined from IBM/EvolveGCN's utils.py ----------------------

class _Namespace:
    """Tiny attribute dict, replaces utils.Namespace in IBM/EvolveGCN."""
    def __init__(self, d=None):
        if d is not None:
            for k, v in d.items():
                setattr(self, k, v)


def _pad_with_last_val(t, k):
    # Used only inside TopK, which we never call from EGCN-O. Kept for parity.
    pad = t[-1].repeat(k - t.size(0))
    return torch.cat([t, pad], dim=0)


# ---- EvolveGCN-O core (verbatim from egcn_o.py, with utils renames) -----

class mat_GRU_gate(nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self._reset(self.W)
        self.U = Parameter(torch.Tensor(rows, rows))
        self._reset(self.U)
        self.bias = Parameter(torch.zeros(rows, cols))

    @staticmethod
    def _reset(t):
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        return self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)


class TopK(nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        stdv = 1.0 / math.sqrt(self.scorer.size(0))
        self.scorer.data.uniform_(-stdv, stdv)
        self.k = k

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask
        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]
        if topk_indices.size(0) < self.k:
            topk_indices = _pad_with_last_val(topk_indices, self.k)
        tanh = nn.Tanh()
        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))
        return out.t()


class mat_GRU_cell(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows, args.cols, nn.Sigmoid())
        self.reset = mat_GRU_gate(args.rows, args.cols, nn.Sigmoid())
        self.htilda = mat_GRU_gate(args.rows, args.cols, nn.Tanh())
        self.choose_topk = TopK(feats=args.rows, k=args.cols)

    def forward(self, prev_Q):
        z_topk = prev_Q
        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)
        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)
        new_Q = (1 - update) * prev_Q + update * h_cap
        return new_Q


class GRCU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = _Namespace({"rows": args.in_feats, "cols": args.out_feats})
        self.evolve_weights = mat_GRU_cell(cell_args)
        self.activation = args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(args.in_feats, args.out_feats))
        stdv = 1.0 / math.sqrt(self.GCN_init_weights.size(1))
        self.GCN_init_weights.data.uniform_(-stdv, stdv)

    def forward(self, A_list: List[torch.Tensor], node_embs_list: List[torch.Tensor]):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            GCN_weights = self.evolve_weights(GCN_weights)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            out_seq.append(node_embs)
        return out_seq


class EGCN(nn.Module):
    """EvolveGCN-O stack."""

    def __init__(self, feats, activation):
        super().__init__()
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            grcu_args = _Namespace({
                "in_feats": feats[i - 1],
                "out_feats": feats[i],
                "activation": activation,
            })
            self.GRCU_layers.append(GRCU(grcu_args))

    def forward(self, A_list: List[torch.Tensor], Nodes_list: List[torch.Tensor]):
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)
        return Nodes_list[-1]


# ---- Adapter used by main.py --------------------------------------------

def _normalize_adj(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric normalization with self-loops: A_hat = D^-1/2 (A+I) D^-1/2.

    Args:
        A: (..., N, N) — dense adjacency (may be the dataloader's xcorr
           top-k graph; entries may be negative, so we take |A|).
    Returns:
        same shape as A.
    """
    A = A.abs()
    N = A.size(-1)
    I = torch.eye(N, device=A.device, dtype=A.dtype).expand_as(A)
    A_hat = A + I
    deg = A_hat.sum(dim=-1).clamp(min=eps)
    d_inv_sqrt = deg.pow(-0.5)
    D = torch.diag_embed(d_inv_sqrt)
    return D @ A_hat @ D


class EvolveGCN_Model_classification(nn.Module):
    """EvolveGCN-O wrapper matching the (x, seq_lengths, adj) → (logits, hidden) API.

    Architecture:
      - Two GRCU layers (input_dim → rnn_units → rnn_units)
      - Max-pool over nodes (matches EvoBrain's default `agg=max`)
      - Linear → num_classes
    """

    def __init__(self, args, num_classes, device=None):
        super().__init__()
        self.num_nodes = args.num_nodes
        self.num_classes = num_classes
        self.device = device
        # Hidden width follows EvoBrain's default rnn_units (64).
        hidden = args.rnn_units
        self.egcn = EGCN(
            feats=[args.input_dim, hidden, hidden],
            activation=nn.RReLU(),
        )
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden, num_classes)
        self.agg = getattr(args, "agg", "max")

    def forward(self, input_seq, seq_lengths, adj):
        """
        Args:
            input_seq: (B, T, N, input_dim)
            seq_lengths: (B,)   — unused (all clips have fixed T)
            adj: (B, T, N, N)   — dynamic adjacency from dataloader (xcorr top-k)
        Returns:
            logits: (B, num_classes)
            hidden: (B, hidden) — node-pooled embedding (used for hidden.csv)
        """
        B, T, N, _ = input_seq.shape

        # Normalize per (batch, time) adjacency.
        Ahat = _normalize_adj(adj)                         # (B, T, N, N)

        A_list = [Ahat[:, t] for t in range(T)]            # each (B, N, N)
        Nodes_list = [input_seq[:, t] for t in range(T)]   # each (B, N, input_dim)

        final = self.egcn(A_list, Nodes_list)              # (B, N, hidden)

        if self.agg == "max":
            pooled, _ = final.max(dim=1)
        elif self.agg == "mean":
            pooled = final.mean(dim=1)
        elif self.agg == "sum":
            pooled = final.sum(dim=1)
        else:
            raise ValueError(f"Unsupported agg: {self.agg}")

        logits = self.fc(self.relu(self.dropout(pooled)))  # (B, num_classes)
        return logits, pooled
