"""EEGPT (Wang et al., NeurIPS 2024) wrapper for our detection pipeline.

The pretrained EEGPT encoder (eegpt_mcae_58chs_4s_large4E.ckpt) was trained
with 58-channel scalp EEG sampled at 256 Hz over 4-second windows
(img_size=[58, 1024], patch_stride=64). To plug into our existing TUSZ /
CHB-MIT dataloader (which yields raw signals at 200 Hz, shape
(B, T_sec, N, 200) when use_fft=False), the wrapper:

  1. reshapes to (B, N, T_sec * 200)
  2. lets the model's built-in temporal_interpolation resample to
     desired_time_len = 256 * 4 = 1024 (the pretrained 4-s window)
  3. relies on use_chan_conv=True (a learned Conv1d) to project N -> 58
     so we don't have to remap channel names to EEGPT's 58-electrode dict.
     This is the same path the EEGPT paper uses for downstream datasets
     whose montage differs from pretraining.

Pretrained weights are loaded with strict=False; classifier head and
chan_conv layer are randomly initialised and trained from scratch.
"""

import os
import torch
import torch.nn as nn

from model.eegpt_pretrained.EEGPT_mcae_finetune import EEGPTClassifier, CHANNEL_DICT

# CHANNEL_DICT in the vendored EEGPT code actually has 62 entries, but the
# pretrained encoder name ("eegpt_mcae_58chs_4s_large4E") + img_size[0]=58
# tells us only the first 58 chan_embed slots were ever exercised during
# pretraining. Pass exactly 58 names so chan_ids has the same length as the
# img_size[0]=58 spatial axis the patch_embed produces.
_EEGPT_58 = sorted(CHANNEL_DICT.keys(), key=lambda k: CHANNEL_DICT[k])[:58]


def _load_eegpt_pretrained(model: nn.Module, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    # The slim ckpt is stored fp16 to fit under 100 MB; cast every tensor
    # back to fp32 so it matches the randomly-initialised chan_conv / head
    # layers and doesn't trip "Input type (Half) and bias type (float)".
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pfx in ("model.", "module."):
            if nk.startswith(pfx):
                nk = nk[len(pfx):]
        if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
            cleaned[nk] = v.to(torch.float32)
        else:
            cleaned[nk] = v
    miss, unexp = model.load_state_dict(cleaned, strict=False)
    print(f"[EEGPT] loaded {ckpt_path}")
    print(f"[EEGPT]   missing {len(miss)} keys (first 5): {miss[:5]}")
    print(f"[EEGPT]   unexpected {len(unexp)} keys (first 5): {unexp[:5]}")


class EEGPT_classification(nn.Module):
    """Thin wrapper that matches main.py's `model(x)` BIOT-style call."""

    def __init__(self, args, num_classes: int = 1, device=None):
        super().__init__()
        self.args = args
        n_chans = args.num_nodes
        # The EEGPT encoder was pretrained with a 4-second window at 256 Hz.
        # Keep the model's pos/time embeddings at that size and let the
        # internal temporal_interpolation downsample our 12/60-s clips.
        self.classifier = EEGPTClassifier(
            num_classes=num_classes,
            in_channels=n_chans,
            img_size=[58, 256 * 4],
            patch_stride=64,
            use_channels_names=_EEGPT_58,
            use_chan_conv=True,
            use_predictor=True,
            use_mean_pooling=True,
            desired_time_len=256 * 4,
            interpolate_factor=2.0,
            qkv_bias=True,
        )

        ckpt = getattr(args, "pretrained_path", None) or os.environ.get(
            "EEGPT_CKPT",
            "/storage/scratch1/3/hkim3239/eeg/pretrained/eegpt/"
            "eegpt_mcae_58chs_4s_large4E.ckpt",
        )
        if ckpt and os.path.isfile(ckpt):
            _load_eegpt_pretrained(self.classifier, ckpt)
        else:
            print(f"[EEGPT] WARNING: no pretrained weight at {ckpt}; training from scratch.")

    def forward(self, x):
        # x: (B, T_sec, N, 200) raw signal at 200 Hz
        # -> (B, N, T_sec * 200) which the classifier interpolates to 4s@256Hz
        if x.dim() == 4:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.flatten(2)
        logits = self.classifier(x)
        return logits, None
