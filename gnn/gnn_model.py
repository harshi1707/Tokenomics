import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

