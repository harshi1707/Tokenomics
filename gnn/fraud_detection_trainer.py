import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, precision_recall_fscore_support
)
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import json
import time


class FraudDetectionTrainer:
    """Comprehensive trainer for fraud detection models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self, data, train_mask, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        
        optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        
        # Only compute loss on known training labels
        train_out = out[train_mask]
        train_y = data.y[train_mask]
        
        # Convert -1 (unknown) to 0 for loss computation
        train_y_binary = (train_y == 1).float()
        
        loss = criterion(train_out, train_y_binary)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, data, val_mask, criterion):
        """Validate the model"""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            
            # Only evaluate on known validation labels
            val_out = out[val_mask]
            val_y = data.y[val_mask]
            
            # Convert -1 (unknown) to 0 for loss computation
            val_y_binary = (val_y == 1).float()
            
            loss = criterion(val_out, val_y_binary)
            
            # Compute probabilities
            probs = torch.sigmoid(val_out[:, 1] - val_out[:, 0])
            preds = (probs > 0.5).float()
            
            # Metrics
            auc = roc_auc_score(val_y_binary.cpu(), probs.cpu())
            ap = average_precision_score(val_y_binary.cpu(), probs.cpu())
            
            return loss.item(), auc, ap, probs.cpu().numpy(), preds.cpu().numpy()
    
    def train(self, data, train_mask, val_mask, 
              epochs: int = 200, lr: float = 0.01, weight_decay: float = 5e-4,
              patience: int = 20, save_path: Optional[str] = None):
        """Train the fraud detection model"""
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        
        best_val_auc = 0
        patience_counter = 0
        
        print(f"Training on {train_mask.sum()} nodes, validating on {val_mask.sum()} nodes")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            train_loss = self.train_epoch(data, train_mask, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_auc, val_ap, val_probs, val_preds = self.validate(data, val_mask, criterion)
            self.val_losses.append(val_loss)
            self.val_metrics.append({'auc': val_auc, 'ap': val_ap})
            
            # Update scheduler
            scheduler.step(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model with AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
        
        return best_val_auc
    
    def evaluate(self, data, test_mask) -> Dict:
        """Comprehensive evaluation on test set"""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            
            # Only evaluate on known test labels
            test_out = out[test_mask]
            test_y = data.y[test_mask]
            
            # Convert to binary
            test_y_binary = (test_y == 1).float()
            
            # Get predictions
            probs = torch.sigmoid(test_out[:, 1] - test_out[:, 0])
            preds = (probs > 0.5).float()
            
            # Metrics
            auc = roc_auc_score(test_y_binary.cpu(), probs.cpu())
            ap = average_precision_score(test_y_binary.cpu(), probs.cpu())
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_y_binary.cpu(), preds.cpu(), average='binary'
            )
            
            # Confusion Matrix
            cm = confusion_matrix(test_y_binary.cpu(), preds.cpu())
            
            # Precision at K
            k_values = [10, 50, 100, 500]
            precision_at_k = {}
            
            sorted_indices = torch.argsort(probs, descending=True)
            for k in k_values:
                if k <= len(sorted_indices):
                    top_k_indices = sorted_indices[:k]
                    top_k_labels = test_y_binary[top_k_indices]
                    precision_at_k[f'precision@{k}'] = top_k_labels.sum().item() / k
            
            return {
                'auc': auc,
                'average_precision': ap,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'precision_at_k': precision_at_k,
                'probabilities': probs.cpu().numpy(),
                'predictions': preds.cpu().numpy(),
                'true_labels': test_y_binary.cpu().numpy()
            }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Metrics curves
        epochs = range(len(self.val_metrics))
        aucs = [m['auc'] for m in self.val_metrics]
        aps = [m['ap'] for m in self.val_metrics]
        
        ax2.plot(epochs, aucs, label='Validation AUC')
        ax2.plot(epochs, aps, label='Validation AP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_pr_curves(self, eval_results: Dict, save_path: Optional[str] = None):
        """Plot ROC and Precision-Recall curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        y_true = eval_results['true_labels']
        y_probs = eval_results['probabilities']
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {eval_results["auc"]:.4f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ax2.plot(recall, precision, label=f'PR (AP = {eval_results["average_precision"]:.4f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, eval_results: Dict, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = eval_results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Licit', 'Illicit'],
                   yticklabels=['Licit', 'Illicit'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_fraud_report(self, eval_results: Dict, top_k: int = 100) -> Dict:
        """Generate comprehensive fraud detection report"""
        
        y_true = eval_results['true_labels']
        y_probs = eval_results['probabilities']
        
        # Top suspicious transactions
        top_indices = np.argsort(y_probs)[::-1][:top_k]
        top_suspicious = {
            'indices': top_indices.tolist(),
            'probabilities': y_probs[top_indices].tolist(),
            'true_labels': y_true[top_indices].tolist()
        }
        
        # Performance summary
        performance_summary = {
            'auc': eval_results['auc'],
            'average_precision': eval_results['average_precision'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1': eval_results['f1'],
            'precision_at_k': eval_results['precision_at_k']
        }
        
        # Fraud detection insights
        total_transactions = len(y_true)
        fraud_transactions = int(y_true.sum())
        detected_frauds = int((y_probs > 0.5).sum())
        true_frauds_detected = int(((y_probs > 0.5) & (y_true == 1)).sum())
        
        insights = {
            'total_transactions': int(total_transactions),
            'actual_fraud_count': fraud_transactions,
            'fraud_rate': float(fraud_transactions / total_transactions),
            'detected_suspicious': detected_frauds,
            'true_frauds_detected': true_frauds_detected,
            'detection_rate': float(true_frauds_detected / fraud_transactions) if fraud_transactions > 0 else 0
        }
        
        return {
            'performance_summary': performance_summary,
            'fraud_insights': insights,
            'top_suspicious_transactions': top_suspicious,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


def train_multiple_models(models_config: Dict, data, train_mask, val_mask, test_mask,
                         models_dir: str = "models") -> Dict:
    """Train multiple models and compare their performance"""
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create model
        from .fraud_detection_models import create_fraud_detector
        model = create_fraud_detector(config['type'], data.num_node_features, **config['params'])
        
        # Train
        trainer = FraudDetectionTrainer(model)
        save_path = os.path.join(models_dir, f"{model_name}.pt")
        
        best_auc = trainer.train(
            data, train_mask, val_mask,
            epochs=config.get('epochs', 200),
            lr=config.get('lr', 0.01),
            save_path=save_path
        )
        
        # Evaluate
        eval_results = trainer.evaluate(data, test_mask)
        
        # Generate report
        report = trainer.generate_fraud_report(eval_results)
        
        results[model_name] = {
            'best_val_auc': best_auc,
            'test_results': eval_results,
            'report': report,
            'trainer': trainer
        }
        
        print(f"\n{model_name} Results:")
        print(f"Best Validation AUC: {best_auc:.4f}")
        print(f"Test AUC: {eval_results['auc']:.4f}")
        print(f"Test AP: {eval_results['average_precision']:.4f}")
        print(f"Test F1: {eval_results['f1']:.4f}")

        # ...existing code...
def get_gnn_output():
    # Simulate GNN output
    return {"node_scores": [0.7, 0.5, 0.9], "graph_feature": 0.8}
# ...existing code...
    
    return results


