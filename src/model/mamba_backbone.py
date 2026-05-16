"""
Shared Mamba backbone for the LightEvoBrain-derived variants.

`BiMambaBackbone` runs each Mamba layer forward AND backward on the per-channel
time axis, then averages the two — i.e. each channel embedding at time t sees
both past and future context within the clip. This is the standard
"Bi-Mamba" trick used by Brain-Go-Brr v4 and other recent EEG SSM papers.

When `bidirectional=False` the module reduces to the original LightEvoBrain
per-channel Mamba (no behavior change on a model that opts out).

Output convention:
    Input  x:  (B, N, T, d_input)        STFT features per channel/timestep
    Output H:  (B, T, N, d_model)        the time axis is kept (NOT collapsed
                                          to last step) so downstream graph
                                          layers can use full-clip dynamics.
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMambaBackbone(nn.Module):
    def __init__(self, d_input: int, d_model: int,
                 n_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(d_input, d_model)
        self.fwd = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])
        if bidirectional:
            # Separate Mamba weights for the reverse direction (standard practice).
            self.bwd = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])

        # Per-layer LayerNorm helps stability when summing fwd+bwd.
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, d_input)
        returns H: (B, T, N, d_model)
        """
        B, N, T, D = x.shape
        h = self.input_proj(x.reshape(B * N, T, D))          # (B*N, T, d_model)

        for i in range(self.n_layers):
            h_f = self.fwd[i](h)
            if self.bidirectional:
                # Reverse along time, run, reverse back.
                h_b = self.bwd[i](h.flip(dims=[1])).flip(dims=[1])
                h_new = 0.5 * (h_f + h_b)
            else:
                h_new = h_f
            # Residual + norm. Mamba already has internal residuals but this
            # makes the bi-directional average stable.
            h = self.layer_norms[i](h + h_new)

        # Back to (B, T, N, d_model)
        return h.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()
