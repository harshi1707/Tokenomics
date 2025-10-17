import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, horizon: int = 7):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class GRUForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, horizon: int = 7):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def make_sequences(series: pd.Series, lookback: int, horizon: int):
    values = series.values.astype(np.float32)
    X, y = [], []
    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i : i + lookback])
        y.append(values[i + lookback : i + lookback + horizon])
    X = np.array(X)[:, :, None]
    y = np.array(y)
    return torch.from_numpy(X), torch.from_numpy(y)

