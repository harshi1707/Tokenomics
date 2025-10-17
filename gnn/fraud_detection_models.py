import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import numpy as np


class GraphSAGEFraudDetector(nn.Module):
    """GraphSAGE-based fraud detector for the Elliptic dataset"""
    
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GATFraudDetector(nn.Module):
    """Graph Attention Network for fraud detection"""
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, 
                 num_heads=8, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                           dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, 
                           heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            x1, att1 = self.gat1(x, edge_index, return_attention_weights=True)
            x1 = F.elu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2, att2 = self.gat2(x1, edge_index, return_attention_weights=True)
            logits = self.classifier(x2)
            return logits, (att1, att2)
        
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return self.classifier(x)


class TransformerFraudDetector(nn.Module):
    """Transformer-based GNN for fraud detection"""
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, 
                 num_heads=8, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        
        self.trans1 = TransformerConv(in_channels, hidden_channels, 
                                    heads=num_heads, dropout=dropout)
        self.trans2 = TransformerConv(hidden_channels * num_heads, hidden_channels,
                                    heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = F.elu(self.trans1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.trans2(x, edge_index)
        return self.classifier(x)


class HybridFraudDetector(nn.Module):
    """Hybrid model combining graph structure and temporal patterns"""
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 lstm_hidden=32, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        
        # Graph component
        self.graph_conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.graph_conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, dropout=dropout)
        
        # Temporal component (LSTM for sequence patterns)
        self.lstm = nn.LSTM(input_size=3, hidden_size=lstm_hidden, 
                           batch_first=True, dropout=dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels + lstm_hidden, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, sequences=None):
        # Graph embeddings
        g_emb = F.elu(self.graph_conv1(x, edge_index))
        g_emb = F.dropout(g_emb, p=self.dropout, training=self.training)
        g_emb = self.graph_conv2(g_emb, edge_index)
        
        # Temporal embeddings (if sequences provided)
        if sequences is not None:
            lstm_out, _ = self.lstm(sequences)
            t_emb = lstm_out[:, -1, :]  # Take last hidden state
        else:
            # Fallback: create dummy temporal embedding
            t_emb = torch.zeros(x.size(0), self.lstm.hidden_size, device=x.device)
        
        # Fuse embeddings
        combined = torch.cat([g_emb, t_emb], dim=1)
        return self.fusion(combined)


class EnsembleFraudDetector(nn.Module):
    """Ensemble of multiple GNN architectures for robust fraud detection"""
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        
        self.sage_detector = GraphSAGEFraudDetector(in_channels, hidden_channels, out_channels)
        self.gat_detector = GATFraudDetector(in_channels, hidden_channels, out_channels)
        self.trans_detector = TransformerFraudDetector(in_channels, hidden_channels, out_channels)
        
        # Ensemble weights (learnable)
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x, edge_index):
        sage_out = self.sage_detector(x, edge_index)
        gat_out = self.gat_detector(x, edge_index)
        trans_out = self.trans_detector(x, edge_index)
        
        # Weighted ensemble
        weights = F.softmax(self.weights, dim=0)
        ensemble_out = (weights[0] * sage_out + 
                       weights[1] * gat_out + 
                       weights[2] * trans_out)
        
        return ensemble_out


def create_fraud_detector(model_type, in_channels, **kwargs):
    """Factory function to create fraud detection models"""
    
    models = {
        'sage': GraphSAGEFraudDetector,
        'gat': GATFraudDetector,
        'transformer': TransformerFraudDetector,
        'hybrid': HybridFraudDetector,
        'ensemble': EnsembleFraudDetector
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](in_channels, **kwargs)


