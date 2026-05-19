"""
`DCRNN_dense` — DCRNN with per-second (seq2seq) output.

Variant of `DCRNNModel_classification` for point-wise (per-second) seizure
detection. The encoder's hidden state output already has shape
(seq_len, B, num_nodes * rnn_units) — i.e. time axis is alive. The paper
takes only the last relevant timestep, then FC + max-pool over nodes -> (B, 1).

Dense variant: keep ALL timesteps, apply the same FC + max-pool at each t
-> (B, T, 1). The FC parameters are time-shared.

Forward output: (B, T, n_classes), hidden (B, T, N, rnn_units)
"""
import torch
import torch.nn as nn

from model.DCRNN import DCRNNEncoder


class DCRNNDense(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super().__init__()
        self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(
            input_dim=args.input_dim,
            max_diffusion_step=args.max_diffusion_step,
            hid_dim=args.rnn_units,
            num_nodes=args.num_nodes,
            num_rnn_layers=args.num_rnn_layers,
            dcgru_activation=args.dcgru_activation,
            filter_type=args.filter_type,
        )

        # Per-timestep FC (parameters shared across t).
        self.fc = nn.Linear(args.rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        """
        Args:
            input_seq: (B, T, N, D)
            seq_lengths: (B,)
            supports: (B, T, k, N, N) or similar — same as paper DCRNN
        Returns:
            logits: (B, T, n_classes)
            hidden: (B, T, N, rnn_units)
        """
        B, T_, N, _ = input_seq.shape

        # (T, B, N, D)
        x_tbnd = torch.transpose(input_seq, 0, 1)

        init_hidden = self.encoder.init_hidden(B).to(self._device)

        # encoder returns: output_hidden (num_layers, B, N*rnn_units),
        # current_inputs (T, B, N*rnn_units)  — the per-t hidden states of last layer.
        _, current_inputs = self.encoder(x_tbnd, init_hidden, supports)

        # current_inputs: (T, B, N * rnn_units) -> (B, T, N, rnn_units)
        H_seq = current_inputs.transpose(0, 1).view(B, T_, N, self.rnn_units)
        H_seq = H_seq.to(self._device)

        # Per-t FC + max-pool over N nodes
        z = self.fc(self.relu(self.dropout(H_seq)))                        # (B, T, N, n_cls)
        logits, _ = torch.max(z, dim=2)                                    # (B, T, n_cls)
        return logits, H_seq


# Alias matching the naming convention used elsewhere.
DCRNNDense_classification = DCRNNDense
