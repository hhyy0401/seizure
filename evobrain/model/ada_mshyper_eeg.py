"""
EEG wrapper around the OFFICIAL Ada-MSHyper Model.

The actual algorithm (AHL, HypergraphConv, Bottleneck_Construct CSCM, 3-path
summation, constraint loss) lives in:
    model/baselines_official/ada_mshyper/ASHyper.py (downloaded byte-faithful
    from https://github.com/shangzongjiang/Ada-MSHyper @ main)

This file only:
  1. Projects EEG STFT input (B, N, T, F) → (B, T, N*d_inner) so it matches
     the official (B, seq_len, enc_in) contract.
  2. Replaces the official Linear(pred_len) forecasting head with an FF
     classification head (LayerNorm + Linear(channels, num_classes)).
  3. Re-exposes the constraint loss via .hypergraph_aux_loss() so main.py
     can add it to the total loss (matching official's dual-loss training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from model.baselines_official.ada_mshyper.ASHyper import Model as _OfficialModel


class AdaMSHyperEEG(nn.Module):
    """EEG adapter around official Ada-MSHyper.

    Args:
        num_nodes:   N channels (19 TUSZ / 22 CHB-MIT).
        input_dim:   F STFT bins.
        max_seq_len: T timesteps.
        d_inner:     per-(B,N,T) embedding dim before flatten. The official
                     `enc_in` will be N*d_inner.
        window_size: CSCM pyramid pooling factors.
        hyper_num:   #hyperedges per scale (paper default [50,20,10] for
                     window=[4,4] @ seq_len=96).
        topk:        TopK assignments per node in AHL (paper default 3).
        d_model:     internal embedding dim for AHL embeddings (paper 512;
                     we keep modest because our seq_len is short).
        inner_size:  AHL inner_size (used for hypergraph mask construction).
        pred_len:    internal forecasting horizon — re-used as embedding
                     length before FF head.
    """
    def __init__(self, num_nodes=19, input_dim=100, max_seq_len=12,
                 d_inner=32, window_size=(2, 2),
                 hyper_num=(16, 8, 4), topk=3,
                 d_model=16, inner_size=4, pred_len=16,
                 num_classes=1):
        super().__init__()
        self.N = num_nodes
        self.T = max_seq_len
        self.F_in = input_dim
        self.d_inner = d_inner

        # EEG-side: F-dim STFT → d_inner per (B, N, T)
        self.feat_proj = nn.Linear(input_dim, d_inner)
        self.channels = num_nodes * d_inner   # official `enc_in`

        # Build configs object for the official Model
        configs = SimpleNamespace(
            seq_len=max_seq_len,
            pred_len=pred_len,
            enc_in=self.channels,
            individual=False,
            window_size=list(window_size),
            hyper_num=list(hyper_num),
            CSCM='Bottleneck_Construct',
            d_model=d_model,
            inner_size=inner_size,
            k=topk,
        )
        self._official = _OfficialModel(configs)

        # Replace forecasting head: official returns (B, pred_len, enc_in).
        # We pool over pred_len then classify enc_in → num_classes.
        self.head_norm = nn.LayerNorm(self.channels)
        self.classifier = nn.Linear(self.channels, num_classes)

        self._hyper_loss_buf = torch.tensor(0.0)

    def hypergraph_aux_loss(self):
        return self._hyper_loss_buf

    def forward(self, x_in, *_args, **_kwargs):
        """x_in: (B, N, T, F). Returns (B, num_classes)."""
        B = x_in.size(0)
        h = self.feat_proj(x_in)                      # (B, N, T, d_inner)
        h = h.permute(0, 2, 1, 3).contiguous()        # (B, T, N, d_inner)
        x = h.view(B, self.T, self.channels)          # (B, seq_len, enc_in)

        # Call official Model; signature is (x, x_mark_enc) but body
        # never references x_mark_enc, so we pass None.
        out, conloss = self._official(x, None)        # out: (B, pred_len, enc_in)

        self._hyper_loss_buf = conloss if isinstance(conloss, torch.Tensor) \
            else torch.tensor(0.0, device=x_in.device)

        # FF classification head: pool over pred_len axis + Linear.
        pooled = out.mean(dim=1)                      # (B, enc_in)
        logits = self.classifier(self.head_norm(pooled))  # (B, num_classes)
        return logits


class AdaMSHyperEEG_classification(AdaMSHyperEEG):
    """Convention-matching alias for main.py dispatch."""
    pass
