"""
`cnnlstm_dense` — CNN-LSTM with per-second (seq2seq) output.

Paper CNN-LSTM: Conv -> per-t FC -> LSTM -> last-t FC -> (B, 1).
Dense variant: same backbone, but final FC at every t -> (B, T, 1).
"""
import torch
import torch.nn as nn


class CNN_LSTM_Dense(nn.Module):
    def __init__(self, num_classes=1, dataset='TUSZ'):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Lazy fc1: dim depends on N (CHB-MIT has 22 channels, TUSZ has 19).
        # Hardcoded paper values broke whenever input shape differed. Lazy
        # init reads the actual flattened size on the first forward.
        self.fc1 = None
        self._fc1_out = 512

        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, seq_lengths):
        B, T_, N, D = x.shape
        x = x.reshape(-1, N, D).unsqueeze(1)                 # (B*T, 1, N, D)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)

        out = out.reshape(B * T_, -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(out.shape[-1], self._fc1_out).to(out.device)
        out = self.fc1(out)
        out = out.reshape(B, T_, -1)                          # (B, T, 512)

        lstm_out, _ = self.lstm(out)                          # (B, T, 128)

        # Per-t FC (shared)
        logits = self.fc2(lstm_out)                           # (B, T, n_cls)
        return logits, lstm_out


CNN_LSTM_Dense_classification = CNN_LSTM_Dense
