"""
EEG wrapper around the OFFICIAL MSHyper Model.

Algorithm (fixed-hypergraph get_mask, HypergraphConv with edge-edge
attention, Bottleneck_Construct, 2-path summation) lives in:
    model/baselines_official/mshyper/MSHyper.py (downloaded byte-faithful
    from https://github.com/shangzongjiang/MSHyper @ main)

This file only:
  1. Projects EEG STFT input (B, N, T, F) → (B, T, N*d_inner).
  2. Replaces forecasting Linear with FF classification head.
  3. Returns just logits (MSHyper has no constraint loss).
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

from model.baselines_official.mshyper.MSHyper import Model as _OfficialModel


class MSHyperEEG(nn.Module):
    def __init__(self, num_nodes=19, input_dim=100, max_seq_len=12,
                 d_inner=16, window_size=(2, 2, 3),
                 inner_size=4, khop=2,
                 d_model=128, d_bottleneck=128, pred_len=16,
                 dropout=0.0, num_classes=1):
        # NOTE: official MSHyper get_mask hardcodes window_size[2] access
        # in the mix-block (Layers.py:182). Must supply 3-element window.
        # window=[2,2,3] @ seq_len=12 gives all_size=[12,6,3,1] — last
        # scale is degenerate (1 node) but only the intra-scale block of
        # get_mask is returned, so the empty-ish last scale is harmless.
        # khop=2 (not paper's 3) because the official intra-skip block in
        # get_mask has an off-by-one on small seq_len: with khop=3 it
        # generates a node index = seq_length (Ms_length) which is OOB
        # for x.transpose(0,1).index_select. khop=2 stays in bounds.
        super().__init__()
        self.N = num_nodes
        self.T = max_seq_len
        self.F_in = input_dim
        self.d_inner = d_inner

        self.feat_proj = nn.Linear(input_dim, d_inner)
        self.channels = num_nodes * d_inner

        configs = SimpleNamespace(
            seq_len=max_seq_len,
            pred_len=pred_len,
            enc_in=self.channels,
            dec_in=self.channels,
            individual=False,
            window_size=list(window_size),
            inner_size=inner_size,
            khop=khop,
            d_model=d_model,
            d_bottleneck=d_bottleneck,
            dropout=dropout,
            CSCM='Bottleneck_Construct',
        )
        self._official = _OfficialModel(configs)

        self.head_norm = nn.LayerNorm(self.channels)
        self.classifier = nn.Linear(self.channels, num_classes)

        self._hyper_loss_buf = torch.tensor(0.0)

    def hypergraph_aux_loss(self):
        # MSHyper has NO aux loss (fixed hypergraph). Return zero scalar.
        return self._hyper_loss_buf

    def forward(self, x_in, *_args, **_kwargs):
        """x_in: (B, N, T, F). Returns (B, num_classes)."""
        B = x_in.size(0)
        h = self.feat_proj(x_in)                       # (B, N, T, d_inner)
        h = h.permute(0, 2, 1, 3).contiguous()         # (B, T, N, d_inner)
        x = h.view(B, self.T, self.channels)           # (B, seq_len, enc_in)

        out = self._official(x, None)                  # (B, pred_len, enc_in)

        pooled = out.mean(dim=1)                       # (B, enc_in)
        logits = self.classifier(self.head_norm(pooled))  # (B, num_classes)
        return logits


class MSHyperEEG_classification(MSHyperEEG):
    pass
