"""
`EvoBrain_dense` — Paper EvoBrain backbone with per-second (seq2seq) output.

Variant of `EvoBrain_classification` for point-wise (per-second) seizure
detection. Reuses the paper's two-stream Mamba backbone and last-timestep
GNN unchanged, but replaces the single-clip readout with a per-timestep
readout: every t in (T,) gets its own logit via FC + node pooling.

Differences vs paper EvoBrain:
  - Paper:  node_last (B, N, d) -> GNN -> (B, N, d') -> FC -> max-pool over N
            -> (B, 1) clip logit
  - Ours:   node_embeds (T, B, N, d) -> per-t FC -> per-t max-pool over N
            -> (B, T, 1) per-second logits.  GNN still runs at last-t only
            (paper-original); its output is fused into the per-t stream as
            a residual at t = T-1 via a lazy-init projection.

This is the cheapest, most faithful "EvoBrain made dense": every learnable
component of the paper is preserved, only the output head changes.

Forward output:  (B, T, n_classes), H_seq (B, T, N, d)
"""
from typing import Tuple

import torch
import torch.nn as nn

from model.EvoBrain import EvoBrain


class EvoBrainDense(nn.Module):
    def __init__(self, args, num_classes, device=None,
                 gnn="ssg", num_eigenvectors=16):
        super().__init__()
        self.num_nodes = args.num_nodes
        self.num_classes = num_classes
        self.device = device
        self.num_eigenvectors = num_eigenvectors
        self.agg = args.agg

        self.evobrain = EvoBrain(
            feat_input_size_edge=args.input_dim,
            feat_input_size_node=args.input_dim,
            feat_target_size=args.rnn_units,
            embed_inside_size=args.input_dim,
            convolve=gnn,
            reduce_edge="mamba",
            reduce_node="mamba",
            skip=False,
            activate="tanh",
            concat=True,
            neo_gnn=True,
            num_eigenvectors=num_eigenvectors,
        )

        # snn_node (mamba) keeps input dim. GNN at last-t has its own (possibly
        # different) output dim; project it back via a lazy Linear.
        d_backbone = args.input_dim

        if args.agg == "concat":
            self.fc_dense = nn.Linear(d_backbone * self.num_nodes, num_classes)
        else:
            self.fc_dense = nn.Linear(d_backbone, num_classes)

        self._d_backbone = d_backbone
        self.gnn_to_d = None  # lazy-init on first forward

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def _evobrain_backbone(self, inputs: torch.Tensor,
                           supports: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replicate paper EvoBrain's forward up to and including the
        last-t GNN, returning:
            node_embeds: (T, B, N, d)  — full-time backbone output
            gnn_out:     (B, N, d')    — last-t GNN output
        """
        T_, b, node, dim = inputs.shape
        x = inputs.reshape(T_, b * node, dim)

        # BiMamba over time per (b*node) flat batch.
        x = x.permute(1, 0, 2)
        x = self.evobrain.snn_node.forward(x)
        x = x.permute(1, 0, 2)
        node_embeds = x.reshape(T_, b, node, dim)                          # (T, B, N, d)

        # Edge stream + last-t GNN — same as paper.
        if supports.shape[2] == 1:
            supports = torch.squeeze(supports, dim=2)
        edge_tuples, edge_features = self.evobrain.create_edge_tuples_and_features(supports)
        edge_features = edge_features.reshape(T_, -1, 1)

        e = edge_features.permute(1, 0, 2)
        e = self.evobrain.snn_edge.forward(e)
        e = e.permute(1, 0, 2)
        edge_embeds = e.reshape(T_, b, -1, dim)

        device = next(self.parameters()).device
        edge_tuples = edge_tuples.to(device)

        node_last = node_embeds[-1].to(device)                             # (B, N, d)
        edge_last = edge_embeds[-1].to(device)                             # (B, E, d)
        edge_weights = self.evobrain.edge_activate(
            self.evobrain.edge_transform(edge_last))                       # (B, E, 1)

        # Per-sample last-t GNN assembly (same as paper).
        from torch_geometric.data import Data
        node_with_pe_list = []
        edge_index_list = []
        edge_weight_list = []
        for i in range(b):
            ew = edge_weights[i]
            ew = ew / (ew.max() + 1e-6)
            data = Data(x=node_last[i], edge_index=edge_tuples,
                        edge_weight=ew.squeeze(-1))
            if self.num_eigenvectors > 0:
                data = self.evobrain.laplacian_pe(data.detach())
                pe = data.laplacian_eigenvector_pe.to(device)
                x_i = torch.cat([node_last[i], pe], dim=-1)
            else:
                x_i = node_last[i]
            node_with_pe_list.append(x_i)
            edge_index_list.append(edge_tuples + i * node)
            edge_weight_list.append(edge_weights[i].squeeze(-1))

        x = torch.stack(node_with_pe_list, dim=0)
        x = self.evobrain.activate(x)
        x = x.view(b * node_last.size(1), -1)

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_weight = torch.cat(edge_weight_list, dim=0)

        gnn_out = self.evobrain.gnnx2.forward(edge_index, edge_weight, x)
        gnn_out = gnn_out.view(b, node_last.size(1), -1)                   # (B, N, d_gnn)
        return node_embeds, gnn_out

    def forward(self, input_seq, seq_lengths, adj):
        """
        Args:
            input_seq: (B, T, N, D)
            adj: (B, T, ..., N, N) per paper code
        Returns:
            logits: (B, T, n_classes)
            hidden: (B, T, N, d)
        """
        B, T_, N, D = input_seq.shape
        inputs_tbnd = torch.transpose(input_seq, 0, 1)

        node_embeds, gnn_out = self._evobrain_backbone(inputs_tbnd, adj)
        # Lazy-init projection on first forward (d_gnn determined here).
        if self.gnn_to_d is None:
            d_gnn_out = gnn_out.shape[-1]
            self.gnn_to_d = nn.Linear(d_gnn_out, self._d_backbone).to(gnn_out.device)
        gnn_proj = self.gnn_to_d(gnn_out)                                  # (B, N, d_backbone)

        # H_seq: (B, T, N, d) — per-t backbone, with GNN info added at t=T-1.
        H_seq = node_embeds.permute(1, 0, 2, 3).contiguous()
        H_seq[:, -1, :, :] = H_seq[:, -1, :, :] + gnn_proj

        # Per-t readout
        if self.agg == "concat":
            z = H_seq.reshape(B, T_, -1)
            logits = self.fc_dense(self.relu(self.dropout(z)))
        else:
            z = self.fc_dense(self.relu(self.dropout(H_seq)))              # (B, T, N, n_cls)
            if self.agg == "max":
                logits, _ = torch.max(z, dim=2)
            elif self.agg == "mean":
                logits = torch.mean(z, dim=2)
            elif self.agg == "sum":
                logits = torch.sum(z, dim=2)
            else:
                raise ValueError(f"Unsupported agg: {self.agg}")

        return logits, H_seq


# Alias matching naming convention used elsewhere in main.py.
EvoBrainDense_classification = EvoBrainDense
