"""
Drop-in temporal-encoder backbones for `LightSTHyper`.

Same input/output convention as `BiMambaBackbone`:
    Input  x: (B, N, T, d_input)
    Output H: (B, T, N, d_model)

Two backbones:
- `DepthwiseSeparable1DBackbone`: per-channel depthwise-sep 1D conv on time.
  Capacity REDUCED relative to BiMamba; aligned with the "less capacity is
  more" finding from the λ-winner ablation on TUSZ.
- `TimesNetBackbone`: FFT period-selection → 2D conv inception → amplitude-
  weighted sum. Adds *new* prior (explicit periodicity) at the cost of more
  params; treated as a graph-as-feature-style hypothesis, not as raw capacity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparable1DBackbone(nn.Module):
    """Per-channel depthwise-separable 1D conv stack, applied to the time axis.

    Each channel n ∈ [N] gets the same conv weights (channels are batched into
    BN), so the temporal pattern detector is shared across electrodes — node-
    specific information must come from the learnable node embedding or from
    the downstream hyperedge.

    EEGNet-style block: depthwise conv (groups=d_model) → pointwise 1×1 →
    GELU → LayerNorm with residual.

    Param count (per layer): kernel * d_model + d_model^2 + 2*d_model.
    For d_model=64, kernel=5 → ~4.5k per layer (vs ~26k per Mamba layer).
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


def _fft_top_periods(x: torch.Tensor, k: int):
    """Return top-k periods (along T) and per-sample amplitudes.

    x: (B, T, d). Excludes DC. Periods clamped to [2, T].
    """
    B, T, d = x.shape
    xf = torch.fft.rfft(x, dim=1)                                # (B, F, d)
    amp = xf.abs().mean(dim=-1)                                  # (B, F)
    amp_mean = amp.mean(dim=0)                                   # (F,)
    amp_mean[0] = 0                                              # drop DC
    k_eff = min(k, amp_mean.numel() - 1)
    _, top_idx = torch.topk(amp_mean, k_eff)                     # (k_eff,)
    top_idx_cpu = top_idx.detach().cpu().tolist()
    periods = []
    for f in top_idx_cpu:
        f = max(int(f), 1)
        p = max(min(T // f, T), 2)
        periods.append(p)
    weights = amp[:, top_idx]                                    # (B, k_eff)
    return periods, weights


class TimesBlock(nn.Module):
    """One TimesNet block: FFT period selection → 2D inception → weighted sum.

    Simplified from the original Times2D / TimesNet block — single 3×3 conv
    branch with GELU; original uses a multi-kernel inception (1, 3, 5).
    Keeps params low so we can stack on top of the hyperedge layer without
    blowing up.
    """

    def __init__(self, d_model: int, k: int = 2):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, d = x.shape
        periods, weights = _fft_top_periods(x, self.k)           # weights: (B, k_eff)
        k_eff = len(periods)
        outs = []
        for i, period in enumerate(periods):
            # Pad T so it's a multiple of `period`, then fold into (T/p, p).
            if T % period != 0:
                pad_len = period - (T % period)
                pad = x.new_zeros(B, pad_len, d)
                xp = torch.cat([x, pad], dim=1)
                L = T + pad_len
            else:
                xp = x
                L = T
            n_rows = L // period
            xp = xp.reshape(B, n_rows, period, d).permute(0, 3, 1, 2)  # (B, d, n_rows, period)
            xp = self.conv(xp)
            xp = xp.permute(0, 2, 3, 1).reshape(B, L, d)         # (B, L, d)
            outs.append(xp[:, :T, :])
        stacked = torch.stack(outs, dim=-1)                      # (B, T, d, k_eff)
        w = F.softmax(weights, dim=-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, k_eff)
        return (stacked * w).sum(dim=-1)                         # (B, T, d)


class TimesNetBackbone(nn.Module):
    """Per-channel TimesNet temporal encoder.

    Channels batched (B*N) so the FFT periodicity is computed per (sample,
    channel) pair — different electrodes can pick up different rhythms.
    """

    def __init__(self, d_input: int, d_model: int,
                 n_layers: int = 2, k: int = 2):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers

        self.input_proj = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([TimesBlock(d_model, k=k) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, D = x.shape
        h = self.input_proj(x).reshape(B * N, T, self.d_model)
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(h))
        return h.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()
