import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import networkx as nx
from typing import Tuple, Optional


class EllipticDataProcessor:
    """Enhanced data processor for the Elliptic Bitcoin dataset"""
    
    def __init__(self, data_dir: str = "data/elliptic"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files from the Elliptic dataset"""
        
        edges_path = os.path.join(self.data_dir, "elliptic_txs_edgelist.csv")
        features_path = os.path.join(self.data_dir, "elliptic_txs_features.csv")
        classes_path = os.path.join(self.data_dir, "elliptic_txs_classes.csv")
        
        # Load data
        edges_df = pd.read_csv(edges_path)
        features_df = pd.read_csv(features_path)
        classes_df = pd.read_csv(classes_path)
        
        print(f"Loaded {len(edges_df)} edges, {len(features_df)} features, {len(classes_df)} classes")
        
        return edges_df, features_df, classes_df
    
    def process_labels(self, classes_df: pd.DataFrame) -> np.ndarray:
        """Process class labels: 1=licit, 2=illicit, 0=unknown"""
        
        # Create mapping: 1->0 (licit), 2->1 (illicit), 0->-1 (unknown)
        label_mapping = {1: 0, 2: 1, 0: -1}
        labels = classes_df['class'].map(label_mapping).values
        
        # Statistics
        licit_count = np.sum(labels == 0)
        illicit_count = np.sum(labels == 1)
        unknown_count = np.sum(labels == -1)
        
        print(f"Label distribution: {licit_count} licit, {illicit_count} illicit, {unknown_count} unknown")
        
        return labels
    
    def create_pytorch_geometric_data(self, edges_df: pd.DataFrame, 
                                    features_df: pd.DataFrame, 
                                    labels: np.ndarray) -> Data:
        """Convert to PyTorch Geometric Data object"""
        
        # Create node mapping
        all_nodes = set(edges_df['txId1'].unique()) | set(edges_df['txId2'].unique())
        node_mapping = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        
        # Create edge index
        edge_sources = [node_mapping[node] for node in edges_df['txId1']]
        edge_targets = [node_mapping[node] for node in edges_df['txId2']]
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        
        # Handle features (some nodes might not have features)
        feature_matrix = np.zeros((len(node_mapping), features_df.shape[1] - 1))  # -1 for txId column
        
        for idx, row in features_df.iterrows():
            tx_id = row['txId']
            if tx_id in node_mapping:
                node_idx = node_mapping[tx_id]
                feature_matrix[node_idx] = row.iloc[1:].values  # Skip txId column
        
        # Normalize features
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        x = torch.tensor(feature_matrix, dtype=torch.float)
        
        # Handle labels (some nodes might not have labels)
        node_labels = np.full(len(node_mapping), -1)  # -1 for unknown
        
        for idx, row in pd.DataFrame({'txId': features_df['txId'], 'class': labels}).iterrows():
            tx_id = row['txId']
            if tx_id in node_mapping:
                node_idx = node_mapping[tx_id]
                node_labels[node_idx] = row['class']
        
        y = torch.tensor(node_labels, dtype=torch.long)
        
        # Create masks for known labels
        known_mask = y != -1
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.known_mask = known_mask
        
        print(f"Created PyG Data: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features")
        
        return data
    
    def create_time_based_split(self, data: Data, train_ratio: float = 0.6, 
                               val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create time-based train/validation/test split"""
        
        # For Elliptic dataset, we can use node IDs as a proxy for time
        # (earlier transactions have smaller IDs)
        known_indices = torch.where(data.known_mask)[0].numpy()
        
        # Sort by node ID (proxy for time)
        sorted_indices = np.sort(known_indices)
        
        n_train = int(len(sorted_indices) * train_ratio)
        n_val = int(len(sorted_indices) * val_ratio)
        
        train_indices = sorted_indices[:n_train]
        val_indices = sorted_indices[n_train:n_train + n_val]
        test_indices = sorted_indices[n_train + n_val:]
        
        # Create boolean masks
        train_mask = np.zeros(data.num_nodes, dtype=bool)
        val_mask = np.zeros(data.num_nodes, dtype=bool)
        test_mask = np.zeros(data.num_nodes, dtype=bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def create_random_split(self, data: Data, train_ratio: float = 0.6,
                          val_ratio: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create random train/validation/test split"""
        
        known_indices = torch.where(data.known_mask)[0].numpy()
        
        train_indices, temp_indices = train_test_split(
            known_indices, train_size=train_ratio, random_state=random_state
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_ratio/(1-train_ratio), random_state=random_state
        )
        
        # Create boolean masks
        train_mask = np.zeros(data.num_nodes, dtype=bool)
        val_mask = np.zeros(data.num_nodes, dtype=bool)
        test_mask = np.zeros(data.num_nodes, dtype=bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def load_processed_data(self, split_type: str = 'time', **kwargs) -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
        """Load and process the complete Elliptic dataset"""
        
        # Load raw data
        edges_df, features_df, classes_df = self.load_raw_data()
        
        # Process labels
        labels = self.process_labels(classes_df)
        
        # Create PyG data
        data = self.create_pytorch_geometric_data(edges_df, features_df, labels)
        
        # Create splits
        if split_type == 'time':
            train_mask, val_mask, test_mask = self.create_time_based_split(data, **kwargs)
        else:
            train_mask, val_mask, test_mask = self.create_random_split(data, **kwargs)
        
        return data, train_mask, val_mask, test_mask


def create_elliptic_dataloader(data_dir: str = "data/elliptic", 
                             split_type: str = 'time',
                             batch_size: int = 1) -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to create Elliptic dataset loader"""
    
    processor = EllipticDataProcessor(data_dir)
    return processor.load_processed_data(split_type=split_type)


