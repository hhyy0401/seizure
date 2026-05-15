"""
ST-Hyper (CIKM 2025) EEG baseline — BEST-EFFORT from paper only.

ST-Hyper's source code is NOT publicly released (the authors' GitHub —
shangzongjiang — hosts Ada-MSHyper and MSHyper but no ST-Hyper repo as of
2026-05). The paper is paywalled at the CIKM venue + an arXiv version
(2509.02217) exists.

The defining design choice from §3 of the paper:
    "we treat each feature of 𝓧 as a node in a hypergraph. Thus, the total
     number of nodes in a hypergraph α is (N₁+…+Nⱼ)×K."

i.e. ST-Hyper bundles (spatial scale, temporal scale, position) tuples into
hyperedges, in contrast to Ada-MSHyper (temporal scales only) and STHAT-style
methods (spatial only).

The paper also adds (compared to Ada-MSHyper): STPM module (Spatial Pyramidal
Graph via DTW adjacency + memory net + GCRU encoder), Fusion module, and
edge-edge GAT phase. We do NOT reproduce these (~5-7 days of work and no
ground-truth code to validate against).

This wrapper therefore implements ST-Hyper as:
    "official Ada-MSHyper Model with joint (N*T) input axis"
which captures the core ST-Hyper claim (joint N×T nodes) on top of the
official Ada-MSHyper machinery. We document this as an ablation, not a
full reproduction.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

from model.baselines_official.ada_mshyper.ASHyper import Model as _OfficialAdaModel


class STHyperEEG(nn.Module):
    def __init__(self, num_nodes=19, input_dim=100, max_seq_len=12,
                 d_inner=32, window_size=(4, 4),
                 hyper_num=(40, 20, 10), topk=3,
                 d_model=16, inner_size=4, pred_len=16,
                 num_classes=1):
        super().__init__()
        self.N = num_nodes
        self.T = max_seq_len
        self.F_in = input_dim
        self.d_inner = d_inner

        self.feat_proj = nn.Linear(input_dim, d_inner)
        # Joint (N×T) sequence — ST-Hyper §3 contract
        self.seq_len = num_nodes * max_seq_len
        self.channels = d_inner   # enc_in is per-(N,T) feature dim

        configs = SimpleNamespace(
            seq_len=self.seq_len,
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
        self._official = _OfficialAdaModel(configs)

        self.head_norm = nn.LayerNorm(self.channels)
        self.classifier = nn.Linear(self.channels, num_classes)

        self._hyper_loss_buf = torch.tensor(0.0)

    def hypergraph_aux_loss(self):
        return self._hyper_loss_buf

    def forward(self, x_in, *_args, **_kwargs):
        """x_in: (B, N, T, F). Returns (B, num_classes)."""
        B = x_in.size(0)
        h = self.feat_proj(x_in)                       # (B, N, T, d_inner)
        x = h.view(B, self.seq_len, self.channels)     # (B, N*T, d_inner)

        out, conloss = self._official(x, None)         # (B, pred_len, enc_in)
        self._hyper_loss_buf = conloss if isinstance(conloss, torch.Tensor) \
            else torch.tensor(0.0, device=x_in.device)

        pooled = out.mean(dim=1)                       # (B, enc_in=d_inner)
        logits = self.classifier(self.head_norm(pooled))
        return logits


class STHyperEEG_classification(STHyperEEG):
    pass
