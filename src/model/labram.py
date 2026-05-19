"""LaBraM (Jiang et al., ICLR 2024) wrapper for our detection pipeline.

We use the `braindecode.models.Labram` implementation because the publicly
distributed pretrained weight on HuggingFace
(braindecode/Labram-Braindecode/braindecode_labram_base.pt, ~23 MB) is
key-mapped for that class. It contains the same parameters as the original
labram-base checkpoint, just renamed under the braindecode naming scheme.

Input contract: our TUSZ / CHB-MIT dataloader yields x of shape
(B, T_sec, N, 200) with use_fft=False at 200 Hz. LaBraM expects
(B, n_chans, n_times) where n_times = sfreq * input_window_seconds.
We reshape to (B, N, T_sec * 200) and feed it in directly.

Pretrained weights are loaded with strict=False; classifier head is
randomly initialised and trained from scratch.
"""

import os
import torch
import torch.nn as nn

from braindecode.models import Labram


def _interp_1d(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate a positional / temporal embedding along its sequence axis.

    t: (1, L_old, D)  ->  (1, L_new, D)  via cubic interpolation.
    Used to stretch the pretrained 8-second temporal embedding to our
    12/60-s clip length.
    """
    if t.shape[1] == target_len:
        return t
    # (1, L, D) -> (1, D, L) -> interpolate L -> (1, D, target) -> (1, target, D)
    x = t.permute(0, 2, 1).contiguous()
    x = nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.permute(0, 2, 1).contiguous()


def _load_labram_pretrained(model: nn.Module, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    # Adjust shapes so strict=False actually absorbs the pretrained signal
    # instead of silently dropping it on shape mismatch.
    #   * position_embedding is the per-channel embedding in braindecode 1.3.x
    #     (sized for n_chans+1, not the original 128+1 canonical order).
    #     We SLICE the first n_chans+1 pretrained slots — interpolation
    #     would mix unrelated electrodes.
    #   * temporal_embedding spans time patches (sized for n_patches+1) and the
    #     pretrained window is ~8 s. We linearly INTERPOLATE so 12/60-s
    #     windows still benefit from the pretrained temporal structure.
    model_sd = model.state_dict()
    for k, mode in (("position_embedding", "slice"), ("temporal_embedding", "interp")):
        if k in sd and k in model_sd and sd[k].shape != model_sd[k].shape:
            if sd[k].shape[0] != model_sd[k].shape[0] or sd[k].shape[2] != model_sd[k].shape[2]:
                continue
            target_len = model_sd[k].shape[1]
            if mode == "slice":
                src_len = sd[k].shape[1]
                if src_len >= target_len:
                    print(f"[LaBraM] slicing {k}: {tuple(sd[k].shape)} -> first {target_len}")
                    sd[k] = sd[k][:, :target_len, :].clone()
                else:
                    print(f"[LaBraM] zero-padding {k}: {tuple(sd[k].shape)} -> {target_len}")
                    pad = torch.zeros(1, target_len - src_len, sd[k].shape[2], dtype=sd[k].dtype)
                    sd[k] = torch.cat([sd[k], pad], dim=1)
            else:  # interp
                print(f"[LaBraM] interpolating {k}: {tuple(sd[k].shape)} -> (1, {target_len}, {sd[k].shape[2]})")
                sd[k] = _interp_1d(sd[k], target_len)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[LaBraM] loaded {ckpt_path}")
    print(f"[LaBraM]   missing {len(miss)} keys (first 5): {miss[:5]}")
    print(f"[LaBraM]   unexpected {len(unexp)} keys (first 5): {unexp[:5]}")


class LaBraM_classification(nn.Module):
    """Thin wrapper that matches main.py's `model(x)` BIOT-style call."""

    def __init__(self, args, num_classes: int = 1, device=None):
        super().__init__()
        self.args = args
        n_chans = args.num_nodes
        n_times = args.max_seq_len * 200  # 200 Hz fixed by dataloader
        self.model = Labram(
            n_chans=n_chans,
            n_outputs=num_classes,
            sfreq=200,
            n_times=n_times,
            neural_tokenizer=True,
        )

        ckpt = getattr(args, "pretrained_path", None) or os.environ.get(
            "LABRAM_CKPT",
            "/storage/scratch1/3/hkim3239/eeg/pretrained/labram/"
            "braindecode_labram_base.pt",
        )
        if ckpt and os.path.isfile(ckpt):
            _load_labram_pretrained(self.model, ckpt)
        else:
            print(f"[LaBraM] WARNING: no pretrained weight at {ckpt}; training from scratch.")

    def forward(self, x):
        # x: (B, T_sec, N, 200) raw signal at 200 Hz
        # -> (B, N, T_sec * 200) which braindecode's Labram consumes directly
        if x.dim() == 4:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.flatten(2)
        logits = self.model(x)
        return logits, None
