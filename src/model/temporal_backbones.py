"""
Drop-in temporal-encoder backbones for `LightSTHyper`.

Same input/output convention as `BiMambaBackbone`:
    Input  x: (B, N, T, d_input)
    Output H: (B, T, N, d_model)

`DepthwiseSeparable1DBackbone` is the lightweight Conv1d backbone kept for
ablation against BiMamba. TCN, TimesNet, and NCDE variants were explored
in development and moved to legacy/.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparable1DBackbone(nn.Module):
    """Per-channel depthwise-separable 1D conv stack on the time axis.

    EEGNet-style block: depthwise conv (groups=d_model) → pointwise 1×1 →
    GELU → LayerNorm with residual.

    Each channel n ∈ [N] gets the same conv weights (channels are batched
    into BN), so the temporal pattern detector is shared across electrodes —
    node-specific information must come from the learnable node embedding or
    from the downstream hyperedge.

    Param count (per layer): kernel * d_model + d_model^2 + 2 * d_model.
    For d_model=64, kernel=5 → ~4.5K per layer (vs ~26K per Mamba layer).
    """

    def __init__(self, d_input: int, d_model: int,
                 n_layers: int = 2, kernel_size: int = 5):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers

        self.input_proj = nn.Linear(d_input, d_model)
        pad = kernel_size // 2
        self.depthwise = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                      padding=pad, groups=d_model)
            for _ in range(n_layers)
        ])
        self.pointwise = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=1)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, D = x.shape
        h = self.input_proj(x).reshape(B * N, T, self.d_model)   # (BN, T, d)
        for dw, pw, norm in zip(self.depthwise, self.pointwise, self.norms):
            h_c = h.transpose(1, 2)                              # (BN, d, T)
            h_c = dw(h_c)
            h_c = pw(h_c)
            h_c = F.gelu(h_c).transpose(1, 2)                    # (BN, T, d)
            h = norm(h + h_c)
        return h.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()
