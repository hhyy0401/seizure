"""
`gru_gcn_dense` — GRU-GCN with per-second (seq2seq) output.

Variant of `GRU_GCN_classification` for point-wise (per-second) seizure
detection. The GRU backbone already produces (T, B, N, D) hidden states.
Paper takes only the last-t hidden, GCN-processes it, then FC + agg over N
-> (B, 1).

Dense variant: per-t FC + agg over N at every timestep. To preserve the
spatial GNN contribution from the paper, we still run the GCN once on the
last-t hidden state and fuse it into the per-t stream as a residual at
t = T-1 (paper-compatible).

Forward output: (B, T, n_classes), hidden (B, T, N, D)
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from model.gru_gcn import GRU_GCN


class GRU_GCN_Dense(nn.Module):
    def __init__(self, args, num_classes: int, device=None, gnn: str = "gcn"):
        super().__init__()
        self.num_nodes = args.num_nodes
        self.num_classes = num_classes
        self.device = device
        self.agg = args.agg

        self.gru_gcn = GRU_GCN(
            feat_input_size_edge=args.input_dim,
            feat_input_size_node=args.input_dim,
            feat_target_size=args.rnn_units,
            embed_inside_size=args.input_dim,
            convolve=gnn,
            reduce_edge="gru",
            reduce_node="gru",
            skip=False,
            activate="tanh",
            concat=True,
            neo_gnn=True,
        )

        # Per-t FC operates on the GRU node hidden dim (embed_inside_size).
        d_backbone = args.input_dim
        if args.agg == "concat":
            self.fc_dense = nn.Linear(d_backbone * self.num_nodes, num_classes)
        else:
            self.fc_dense = nn.Linear(d_backbone, num_classes)

        # Lazy projection of GCN output (feat_target_size + extras) back to d_backbone
        # so we can fuse it into the per-t stream at t = T-1.
        self._d_backbone = d_backbone
        self.gnn_to_d = None  # lazy

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def _gru_then_gnn(self, inputs, supports):
        """Replicate GRU_GCN.forward but expose both (T,B,N,D) node hidden seq
        and final GCN output. Mirrors gru_gcn.py:481-543.
        """
        T_, B, N, F_ = inputs.shape
        device = inputs.device

        node_in = inputs.reshape(T_, B * N, F_)
        node_seq, _ = self.gru_gcn.snn_node(node_in)
        node_seq = node_seq.reshape(T_, B, N, self.gru_gcn.embed_inside_size)
        node_final = self.gru_gcn.activate(node_seq[-1])                    # (B, N, D)

        if supports.dim() == 5 and supports.shape[2] == 1:
            adj = supports.squeeze(2)
        elif supports.dim() == 4:
            adj = supports
        else:
            raise ValueError(f"unexpected supports shape: {supports.shape}")

        edge_in = adj.permute(1, 0, 2, 3).reshape(T_, B * N * N, 1)
        edge_seq, _ = self.gru_gcn.snn_edge(edge_in)
        edge_seq = edge_seq.reshape(T_, B, N * N, self.gru_gcn.embed_inside_size)
        edge_final = edge_seq[-1]

        edge_index = self.gru_gcn._create_full_edge_index(N, device=device)
        graphs = []
        for b in range(B):
            graphs.append(Data(
                x=node_final[b], edge_index=edge_index, edge_attr=edge_final[b]))
        batch_graph = Batch.from_data_list(graphs)
        out_nodes = self.gru_gcn.gnnx2(
            batch_graph.edge_index, batch_graph.edge_attr, batch_graph.x)
        out_nodes = out_nodes.reshape(B, N, -1)                             # (B, N, d_gcn)

        # Per-t node hidden seq (B, T, N, D)
        H_seq = node_seq.permute(1, 0, 2, 3).contiguous()
        return H_seq, out_nodes

    def forward(self, input_seq, seq_lengths, adj):
        """
        Args:
            input_seq: (B, T, N, D)
            adj: (B, T, N, N) or (B, T, 1, N, N)
        Returns:
            logits: (B, T, n_classes)
            hidden: (B, T, N, D)
        """
        B, T_, N, D = input_seq.shape

        # (T, B, N, D)
        x_tbnd = input_seq.transpose(0, 1)
        H_seq, gnn_out = self._gru_then_gnn(x_tbnd, adj)                    # (B,T,N,D), (B,N,d_gcn)

        # Lazy-init projection (d_gcn → d_backbone)
        if self.gnn_to_d is None:
            d_gcn = gnn_out.shape[-1]
            self.gnn_to_d = nn.Linear(d_gcn, self._d_backbone).to(gnn_out.device)
        gnn_proj = self.gnn_to_d(gnn_out)                                   # (B, N, D)

        # Fuse GCN info into the last timestep (residual).
        H_seq[:, -1, :, :] = H_seq[:, -1, :, :] + gnn_proj

        if self.agg == "concat":
            z = H_seq.reshape(B, T_, -1)
            logits = self.fc_dense(self.relu(self.dropout(z)))              # (B, T, n_cls)
        else:
            z = self.fc_dense(self.relu(self.dropout(H_seq)))               # (B, T, N, n_cls)
            if self.agg == "max":
                logits, _ = torch.max(z, dim=2)
            elif self.agg == "mean":
                logits = torch.mean(z, dim=2)
            elif self.agg == "sum":
                logits = torch.sum(z, dim=2)
            else:
                raise ValueError(f"Unsupported agg: {self.agg}")

        return logits, H_seq


# Alias matching the convention used elsewhere.
GRU_GCN_Dense_classification = GRU_GCN_Dense
