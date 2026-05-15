"""
Frequency-band functional connectivity (FC) on GPU.

Used by `light_mamba_band_plv` (E+). All ops are torch and differentiable
(no scipy / no detach), so this can live in the model forward path.

The clip-length PLV matrices for 22 channels @ 200Hz are small (~5 * 22 * 22
floats) and the bandpass+Hilbert is a couple of FFTs per band — total cost
is dominated by the FFT of length L (= 2400 for 12s clips).
"""
from typing import Sequence, Tuple

import torch


# Standard EEG bands (Hz). γ upper bound stays below Nyquist (100Hz @ 200Hz fs).
BANDS_DEFAULT: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 1.0,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
)


def _bandpass_fft(signal: torch.Tensor, fs: float, low: float, high: float) -> torch.Tensor:
    """Brick-wall bandpass via FFT mask. signal: (..., L). Returns same shape."""
    L = signal.shape[-1]
    X = torch.fft.rfft(signal, dim=-1)
    freqs = torch.fft.rfftfreq(L, 1.0 / fs).to(signal.device)
    mask = ((freqs >= low) & (freqs <= high)).to(X.dtype)
    return torch.fft.irfft(X * mask, n=L, dim=-1)


def _analytic_phase(signal: torch.Tensor) -> torch.Tensor:
    """Hilbert transform → instantaneous phase. signal: (..., L). Returns (..., L)."""
    L = signal.shape[-1]
    X = torch.fft.fft(signal, dim=-1)

    # Hilbert multiplier h[k]: 1 at DC and Nyquist, 2 for 1..L/2-1, 0 elsewhere.
    h = torch.zeros(L, device=signal.device, dtype=X.dtype)
    if L % 2 == 0:
        h[0] = 1.0
        h[L // 2] = 1.0
        h[1:L // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(L + 1) // 2] = 2.0
    # broadcast h over leading dims
    analytic = torch.fft.ifft(X * h, dim=-1)
    return torch.angle(analytic)


def plv_matrix(phase: torch.Tensor) -> torch.Tensor:
    """PLV from instantaneous phase. phase: (..., N, L) → (..., N, N)."""
    # complex exponential of phase, then pairwise inner product over time
    e = torch.exp(1j * phase.to(torch.float32))           # (..., N, L)
    # pairwise: E_t[e_i * conj(e_j)] over last dim
    inner = torch.einsum("...nl,...ml->...nm", e, e.conj()) / phase.shape[-1]
    return inner.abs().to(phase.dtype)


def band_plv(signal: torch.Tensor,
             fs: float,
             bands: Sequence[Tuple[str, float, float]] = BANDS_DEFAULT
             ) -> torch.Tensor:
    """Per-band PLV adjacency.

    Args:
        signal: (B, N, L) raw time-domain EEG.
        fs: sampling rate (200 Hz for CHB-MIT here).
        bands: list of (name, low_Hz, high_Hz).
    Returns:
        A_plv: (B, n_bands, N, N) — symmetric, in [0, 1], diagonal=1.
    """
    if signal.dim() != 3:
        raise ValueError(f"expected (B, N, L), got {tuple(signal.shape)}")
    out = []
    for _, lo, hi in bands:
        s_band = _bandpass_fft(signal, fs, lo, hi)        # (B, N, L)
        phi = _analytic_phase(s_band)                     # (B, N, L)
        out.append(plv_matrix(phi))                       # (B, N, N)
    return torch.stack(out, dim=1)                        # (B, K, N, N)


def correlation_matrix(signal: torch.Tensor) -> torch.Tensor:
    """Pearson correlation per pair. signal: (B, N, L) → (B, N, N)."""
    s = signal - signal.mean(dim=-1, keepdim=True)
    norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    s = s / norm
    return torch.einsum("bnl,bml->bnm", s, s)


def topk_sparsify(A: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k entries per row, zero out the rest. A: (..., N, N)."""
    if k >= A.shape[-1]:
        return A
    vals, idx = A.topk(k, dim=-1)
    out = torch.zeros_like(A)
    out.scatter_(-1, idx, vals)
    return out


def row_normalize(A: torch.Tensor) -> torch.Tensor:
    """Row-normalize so each row sums to 1 (after adding self-loops)."""
    A = A + torch.eye(A.shape[-1], device=A.device).expand_as(A)
    deg = A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return A / deg
