"""
`lstm_dense` — LSTM with per-second (seq2seq) output.

The paper LSTM takes the last-relevant timestep -> FC -> (B, 1).
Dense variant: every timestep -> same FC -> (B, T, 1). FC params shared
across t.
"""
import torch
import torch.nn as nn


class LSTMDense(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super().__init__()
        self._input_dim = args.input_dim
        self._num_nodes = args.num_nodes
        self._num_rnn_layers = args.num_rnn_layers
        self._rnn_units = args.rnn_units
        self._num_classes = num_classes
        self._device = device

        self.lstm = nn.LSTM(
            args.input_dim * args.num_nodes,
            args.rnn_units,
            args.num_rnn_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(args.rnn_units, num_classes)
        self.relu = nn.ReLU()

    def forward(self, inputs, seq_lengths):
        """
        Args:
            inputs: (B, T, N, D)
            seq_lengths: (B,)
        Returns:
            logits: (B, T, n_classes)
            hidden: (B, T, rnn_units)
        """
        B, T_, N, D = inputs.shape
        x = inputs.reshape(B, T_, -1)                                       # (B, T, N*D)

        h0, c0 = self.init_hidden(B)
        out, _ = self.lstm(x, (h0, c0))                                     # (B, T, rnn_units)

        # Per-t FC (shared)
        logits = self.fc(self.relu(self.dropout(out)))                      # (B, T, n_cls)
        return logits, out

    def init_hidden(self, batch_size):
        w = next(self.parameters()).data
        h = w.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        c = w.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        return h, c


# Alias
LSTMDense_classification = LSTMDense
