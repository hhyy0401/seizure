"""
`light_st_hyper_band` — Band-specific Spatio-Temporal Hyperedge.

Builds on `light_st_hyper` (spatio-temporal soft hypergraph) by splitting the
STFT input into 5 clinically-meaningful frequency bands and running an
independent spatio-temporal hyperedge per band. Each hyperedge in band k
captures (channel, time) tuples that synchronize within band k.

Motivation (EEG seizure dynamics):
  - Pre-ictal / onset:  high-frequency (β, γ) burst, focal
  - Propagation:        θ, α rhythm spread to adjacent channels
  - Generalization:     β, δ bilateral synchronization
A band-grounded hyperedge primitive lets the model express each stage
atomically rather than collapsing all frequencies into one embedding.

Frequency mapping (STFT @ fs=200Hz, n=200 → bins are 1 Hz wide):
    δ : bins  1- 4   ( 4 dims)
    θ : bins  4- 8   ( 4 dims)
    α : bins  8-13   ( 5 dims)
    β : bins 13-30   (17 dims)
    γ : bins 30-50   (20 dims)
DC (bin 0) and bins 50+ (above γ) are dropped.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.light_dyn_hyper import SpatioTemporalHyperedgeBlock, LinearBackbone
from model.mamba_backbone import BiMambaBackbone


# (name, lo_bin_inclusive, hi_bin_exclusive)
BAND_BINS: Tuple[Tuple[str, int, int], ...] = (
    ("delta",  1,  4),
    ("theta",  4,  8),
    ("alpha",  8, 13),
    ("beta",  13, 30),
    ("gamma", 30, 50),
)


class BandSplit(nn.Module):
    """Split STFT input along freq dim into 5 band tensors.

    Input  x: (B, N, T, F_total)  — F_total = 100 by default
    Output:    list of 5 tensors, each (B, N, T, F_band_k)
    """

    def __init__(self, band_bins: Tuple[Tuple[str, int, int], ...] = BAND_BINS):
        super().__init__()
        self.band_bins = band_bins
        self.band_widths = [hi - lo for _, lo, hi in band_bins]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [x[..., lo:hi] for _, lo, hi in self.band_bins]


class LightSTHyperBand(nn.Module):
    """5 parallel ST-hypergraph streams (one per EEG band), then gated fusion.

    Args:
        backbone_type: "linear" or "mamba". Backbones operate per-band.
        gate_mode: "learned" (MLP gate softmax over bands) or "uniform".
    """

    def __init__(
        self,
        d_input_total: int = 100,
        d_band: int = 32,
        d_hidden: int = 64,
        n_nodes: int = 22,
        n_classes: int = 1,
        n_mamba_layers: int = 2,
        n_hyper_layers: int = 2,
        n_hyperedges_per_band: int = 4,
        n_pma_seeds: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        backbone_type: str = "linear",
        gate_mode: str = "learned",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_band = d_band
        self.d_hidden = d_hidden
        self.gate_mode = gate_mode
        self.backbone_type = backbone_type

        self.split = BandSplit()
        n_bands = len(BAND_BINS)
        self.n_bands = n_bands

        # Per-band backbones (each takes a band slice → d_band embedding).
        backbones = []
        for _, lo, hi in BAND_BINS:
            d_in = hi - lo
            if backbone_type == "linear":
                backbones.append(LinearBackbone(d_input=d_in, d_model=d_band))
            elif backbone_type == "mamba":
                backbones.append(BiMambaBackbone(
                    d_input=d_in, d_model=d_band,
                    n_layers=n_mamba_layers, bidirectional=bidirectional,
                ))
            else:
                raise ValueError(f"Unknown backbone_type: {backbone_type}")
        self.backbones = nn.ModuleList(backbones)

        # Per-band stacks of ST-hyperedge blocks.
        # Each band keeps its own hyperedge query set (so hyperedges are
        # band-specialized rather than generic).
        per_band_layers = []
        for _ in range(n_bands):
            layers = []
            for i in range(n_hyper_layers):
                d_in_i = d_band if i == 0 else d_hidden
                layers.append(SpatioTemporalHyperedgeBlock(
                    d_in_i, d_hidden, n_hyperedges=n_hyperedges_per_band))
            per_band_layers.append(nn.ModuleList(layers))
        self.per_band_layers = nn.ModuleList(per_band_layers)

        # Band gate (softmax over 5 bands), conditioned on a global summary
        # of all band readouts. Mirrors the gating idea in E+ but operating
        # over hyperedge readouts instead of GCN readouts.
        if gate_mode == "learned":
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_hidden * n_bands, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, n_bands),
            )

        # PMA readout shared across bands.
        self.pma_seeds = nn.Parameter(torch.randn(n_pma_seeds, d_hidden) * 0.02)
        self.pma_norm = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden * n_pma_seeds, n_classes)

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
        # x: (B, N, T, F_total=100)
        band_xs = self.split(x)                                       # 5 tensors

        band_readouts = []          # each (B, d_hidden) after mean over channels
        band_node_pools = []        # each (B, N, d_hidden) for final per-band readout
        for k, (xk, backbone, layer_stack) in enumerate(
                zip(band_xs, self.backbones, self.per_band_layers)):
            H = backbone(xk)                                          # (B, T, N, d_band)
            for layer in layer_stack:
                H, _ = layer(H)                                       # (B, T, N, d_hidden)
            # Temporal mean: time mixing already happens inside ST-hyperedge.
            H_pool = H.mean(dim=1)                                    # (B, N, d_hidden)
            band_node_pools.append(H_pool)
            band_readouts.append(H_pool.mean(dim=1))                  # (B, d_hidden) crude summary for gate

        # Band gate: softmax weights over 5 bands.
        if self.gate_mode == "learned":
            summary = torch.cat(band_readouts, dim=-1)                # (B, 5*d_hidden)
            gate = torch.softmax(self.gate_mlp(summary), dim=-1)      # (B, 5)
        else:
            gate = x.new_full((x.size(0), self.n_bands), 1.0 / self.n_bands)

        # Fuse band-level node pools: weighted sum over bands → (B, N, d_hidden).
        stacked = torch.stack(band_node_pools, dim=1)                 # (B, 5, N, d_hidden)
        H_fused = (gate.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)

        z = self.pma_readout(H_fused)                                 # (B, n_seeds*d)
        return self.classifier(self.dropout(z)), H_fused


class LightSTHyperBand_classification(nn.Module):
    """main.py-compatible wrapper. Same signature as other light variants."""

    def __init__(self, args, num_classes, device=None,
                 backbone_type: str = "linear", gate_mode: str = "learned"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = LightSTHyperBand(
            d_input_total=args.input_dim,
            d_band=getattr(args, "d_band", 32),
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_hyper_layers=getattr(args, "n_hyper_layers", 2),
            n_hyperedges_per_band=getattr(args, "n_hyperedges_per_band", 4),
            n_pma_seeds=getattr(args, "n_pma_seeds", 1),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
            backbone_type=backbone_type,
            gate_mode=gate_mode,
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        # (B, T, N, D) → (B, N, T, D)
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        logits, hidden = self.model(x)
        return logits, hidden
