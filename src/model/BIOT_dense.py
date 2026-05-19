"""
`BIOT_dense` — BIOT with per-second (seq2seq) output.

Paper BIOT: STFT per channel -> patch embed -> per-channel positional encoding
+ channel token -> concat all channels -> Linear Attention Transformer
-> mean-pool over (16*ts) tokens -> classifier -> (B, 1).

Dense variant: keep the time axis. We split the (16*ts) tokens back into
(channels, ts), then for each of T=max_seq_len output seconds we average
the tokens belonging to that 1-second bin (across channels and within-bin
time tokens) -> (B, T, emb). Classifier applied per-t.

This matches the spirit of paper BIOT (mean-pooling over time within a bin)
while exposing per-second predictions. The encoder is unchanged.
"""
import numpy as np
import torch
import torch.nn as nn

from model.BIOT import BIOTEncoder, ClassificationHead


class BIOTDenseClassifier(nn.Module):
    """BIOT encoder + per-second classifier head.

    Args:
        n_classes:   number of output classes per t
        T_out:       number of output timesteps (e.g. clip_len = 12)
        emb_size:    encoder embedding dim
        heads, depth: transformer config
        n_channels:  channel count for channel tokens
        n_fft, hop_length: STFT config for the BIOT encoder
    """

    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=1,
                 T_out=12, **kwargs):
        super().__init__()
        self.T_out = T_out
        self.n_channels = kwargs.get('n_channels', 16)
        # BIOT-internal STFT: hop_length determines ts per channel
        self.n_fft = kwargs.get('n_fft', 200)
        self.hop_length = kwargs.get('hop_length', 100)
        self.biot_encoder = BIOTEncoder(
            emb_size=emb_size, heads=heads, depth=depth, **kwargs,
        )
        # Per-t classifier (shared params across t)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def encode_full_sequence(self, x_chan_first):
        """Reproduce BIOTEncoder.forward up to the transformer output, without
        the `.mean(dim=1)` collapse. Returns:
            emb_seq: (B, n_channels * ts_per_channel, emb_size)
            ts:      number of time tokens per channel
        """
        biot = self.biot_encoder
        emb_seq = []
        ts_per_channel = None
        for i in range(x_chan_first.shape[1]):
            channel_spec_emb = biot.stft(x_chan_first[:, i:i+1, :])
            channel_spec_emb = biot.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            ts_per_channel = ts
            channel_token_emb = (
                biot.channel_tokens(biot.index[i])
                .unsqueeze(0).unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            channel_emb = biot.positional_encoding(channel_spec_emb + channel_token_emb)
            emb_seq.append(channel_emb)
        emb_concat = torch.cat(emb_seq, dim=1)                # (B, N*ts, emb)
        emb_trans = biot.transformer(emb_concat)              # (B, N*ts, emb)
        return emb_trans, ts_per_channel

    def forward(self, x):
        """
        Args:
            x: (B, T_clip, N, D) — same input convention as the rest of the codebase.
               Will be permuted+reshaped into (B, N, T_clip * D) for BIOT.
        Returns:
            logits: (B, T_out, n_classes)
            hidden: (B, T_out, emb)
        """
        B, T_clip, N, D = x.shape
        # BIOT's interface expects (B, N, full_signal_length).
        x_perm = x.permute(0, 2, 1, 3)                          # (B, N, T_clip, D)
        x_chan_first = x_perm.reshape(B, N, T_clip * D)         # (B, N, T_clip*D)

        emb_trans, ts = self.encode_full_sequence(x_chan_first) # (B, N*ts, emb)
        # Reshape to (B, N, ts, emb), then average across channels to get
        # (B, ts, emb) — the time-resolved sequence.
        emb_trans = emb_trans.reshape(B, N, ts, -1)             # (B, N, ts, emb)
        emb_time = emb_trans.mean(dim=1)                        # (B, ts, emb)

        # Bin into T_out output timesteps. If ts == T_out (likely with our STFT
        # config), use directly. Otherwise interpolate.
        if ts == self.T_out:
            emb_T = emb_time                                    # (B, T_out, emb)
        else:
            # Linear interpolation along time axis to T_out.
            emb_T = torch.nn.functional.interpolate(
                emb_time.transpose(1, 2),                       # (B, emb, ts)
                size=self.T_out, mode='linear', align_corners=False
            ).transpose(1, 2)                                   # (B, T_out, emb)

        logits = self.classifier(emb_T)                         # (B, T_out, n_classes)
        return logits, emb_T


# Alias matching naming convention.
BIOTDenseClassifier_classification = BIOTDenseClassifier
