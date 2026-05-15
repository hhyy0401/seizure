"""
NCDE (Neural Controlled Differential Equation) per-channel backbone.

Drop-in replacement for `BiMambaBackbone` / `LinearBackbone`. Same I/O shape:
    Input  x:  (B, N, T, d_input)
    Output H:  (B, T, N, d_model)

Each channel's time series is treated as a path X_n(t) over t=0..T-1, lifted
to a continuous cubic spline. The hidden state h_n(t) follows
    dh/dt = f(h) · dX_n/dt
where f is a small MLP. The hidden trajectory is evaluated at the original T
timestamps to recover a regular discrete tensor that the downstream
hypergraph layer can consume.

Why this is interesting for seizure detection:
  - NCDEs model continuous dynamics, which matches the dynamical-system
    view of seizure transitions (bifurcations between brain states).
  - Robust to irregular sampling (not exploited here — EEG is uniform).
  - The ODE solver inherently produces smooth trajectories, potentially a
    useful inductive bias for non-stationary EEG.

Computational caveat:
  - Each forward pass invokes an ODE solver per sample, which can be 10–100×
    slower than Mamba on the same sequence. We use a low-order `rk4` solver
    with `step_size=1.0` to keep cost tractable for T=12 EEG clips.
"""
import torch
import torch.nn as nn

import torchcde


class _CDEFunc(nn.Module):
    """The vector field f(h) producing the matrix multiplied by dX/dt.

    Output shape must be (..., d_model, d_input) so that f(h) @ dX/dt gives
    (..., d_model). We project (d_model) → (d_model * d_input) and reshape.
    """

    def __init__(self, d_input: int, d_model: int, d_hidden: int = 64):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_model * d_input),
            nn.Tanh(),
        )

    def forward(self, t, h):
        # h: (..., d_model)
        out = self.net(h)
        out = out.view(*h.shape[:-1], self.d_model, self.d_input)
        return out


class NCDEBackbone(nn.Module):
    """Per-channel NCDE backbone.

    Treats each of N channels independently. We collapse (B, N) → batch, run
    the NCDE per channel, and reshape back. Trajectory is evaluated at the
    original integer timestamps t=0..T-1.
    """

    def __init__(self, d_input: int, d_model: int, d_hidden_field: int = 64,
                 solver: str = "rk4"):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.solver = solver

        # Initial-hidden projection: maps the first observation of the path
        # (and time) to the initial hidden state h(0).
        # torchcde augments the input with time → d_input + 1 channels.
        self.h0_proj = nn.Linear(d_input + 1, d_model)

        self.cde_func = _CDEFunc(d_input + 1, d_model, d_hidden=d_hidden_field)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, d_input) → (B, T, N, d_model)."""
        B, N, T, D = x.shape

        # Build a path tensor (B*N, T, D+1) where the +1 channel is time.
        t = torch.linspace(0.0, float(T - 1), T, device=x.device, dtype=x.dtype)
        x_flat = x.reshape(B * N, T, D)
        # Append explicit time channel (required by torchcde for time-aware paths).
        t_col = t.view(1, T, 1).expand(B * N, T, 1)
        path = torch.cat([t_col, x_flat], dim=-1)                     # (B*N, T, D+1)

        # Hermite cubic spline coefficients (works for regular sampling too).
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            path, t=t)
        X = torchcde.CubicSpline(coeffs, t=t)

        # Initial hidden state from the path's value at t=0.
        h0 = self.h0_proj(X.evaluate(t[0]))                           # (B*N, d_model)

        # Evaluate trajectory at the original T timestamps.
        h_seq = torchcde.cdeint(X=X, func=self.cde_func, z0=h0, t=t,
                                method=self.solver,
                                options={"step_size": 1.0})           # (B*N, T, d_model)

        h_seq = h_seq.reshape(B, N, T, self.d_model)
        # Convert to (B, T, N, d_model) convention used downstream.
        return h_seq.permute(0, 2, 1, 3).contiguous()


class _GraphCDEFunc(nn.Module):
    """Graph-aware vector field — cross-node attention before per-node field.

    Hidden state is treated as (B, N, d_model). Before producing the linear
    map applied to dX/dt, we let each node attend to all other nodes' hidden
    states (a single learned self-attention). This is the spatial coupling
    that distinguishes Graph-NCDE from plain per-channel NCDE.
    """

    def __init__(self, d_input: int, d_model: int, n_nodes: int,
                 d_hidden: int = 64):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_nodes = n_nodes

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)

        self.field = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_model * d_input),
            nn.Tanh(),
        )

    def forward(self, t, h):
        # h: (B, N * d_model)  — packed
        B = h.shape[0]
        N, d = self.n_nodes, self.d_model
        h_nd = h.view(B, N, d)

        q = self.q_proj(h_nd)                                          # (B, N, d)
        k = self.k_proj(h_nd)
        v = self.v_proj(h_nd)
        attn = torch.softmax(torch.einsum("bnd,bmd->bnm", q, k) / (d ** 0.5), dim=-1)
        h_attn = torch.einsum("bnm,bmd->bnd", attn, v)                 # (B, N, d)
        h_mix = self.attn_norm(h_nd + h_attn)                          # residual

        out = self.field(h_mix).view(B, N, d, self.d_input)            # (B, N, d, D_in)
        # Pack to (B, N*d, N*D_in) block-diagonal across N for cdeint contract.
        # Since dX/dt comes packed as (B, N*D_in), each node's contribution only
        # depends on its own dX/dt: f[b, n*d+i, n'*D+j] = δ_{n,n'} · out[b,n,i,j].
        D_in = self.d_input
        f_full = h.new_zeros(B, N * d, N * D_in)
        # Fill block-diagonal entries (vectorized along node axis).
        for n in range(N):
            f_full[:, n * d:(n + 1) * d, n * D_in:(n + 1) * D_in] = out[:, n]
        return f_full


class GraphNCDEBackbone(nn.Module):
    """Graph-coupled NCDE — cross-node attention in the vector field.

    Treats all N channels as a single packed hidden state of dim N·d_model so
    the vector field can mix information across channels at every ODE step
    (STG-NCDE-style coupling, simplified for hypergraph downstream).
    """

    def __init__(self, d_input: int, d_model: int, n_nodes: int,
                 d_hidden_field: int = 48, solver: str = "rk4"):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.solver = solver

        self.h0_proj = nn.Linear(d_input + 1, d_model)
        self.cde_func = _GraphCDEFunc(
            d_input + 1, d_model, n_nodes, d_hidden=d_hidden_field)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, d_input) → (B, T, N, d_model)
        B, N, T, D = x.shape

        t = torch.linspace(0.0, float(T - 1), T, device=x.device, dtype=x.dtype)
        # Build per-node path then pack (B, T, N*(D+1)).
        t_col = t.view(1, 1, T, 1).expand(B, N, T, 1)
        path = torch.cat([t_col, x], dim=-1)                           # (B, N, T, D+1)
        path = path.permute(0, 2, 1, 3).contiguous()                   # (B, T, N, D+1)
        path_flat = path.reshape(B, T, N * (D + 1))                    # (B, T, N*(D+1))

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            path_flat, t=t)
        X = torchcde.CubicSpline(coeffs, t=t)

        # Initial hidden: per-node projection of t=0 value, then pack.
        x0 = X.evaluate(t[0])                                          # (B, N*(D+1))
        x0 = x0.view(B, N, D + 1)
        h0 = self.h0_proj(x0).view(B, N * self.d_model)

        h_seq = torchcde.cdeint(X=X, func=self.cde_func, z0=h0, t=t,
                                method=self.solver,
                                options={"step_size": 1.0})            # (B, T, N*d)

        h_seq = h_seq.view(B, T, N, self.d_model)
        return h_seq.contiguous()
