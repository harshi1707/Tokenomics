#!/usr/bin/env python3
"""
Comprehensive Fraud Detection System using Graph Neural Networks
Trains multiple GNN architectures on the Elliptic Bitcoin dataset
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elliptic_data_loader import create_elliptic_dataloader
from fraud_detection_models import create_fraud_detector
from fraud_detection_trainer import FraudDetectionTrainer, train_multiple_models


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection with GNNs')
    parser.add_argument('--data_dir', type=str, default='data/elliptic',
                       help='Directory containing Elliptic dataset')
    parser.add_argument('--model', type=str, default='all',
                       choices=['sage', 'gat', 'transformer', 'hybrid', 'ensemble', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Hidden channels for models')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results and plots')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    print("Loading Elliptic dataset...")
    data, train_mask, val_mask, test_mask = create_elliptic_dataloader(
        data_dir=args.data_dir, split_type='time'
    )
    
    print(f"Dataset loaded:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Training nodes: {train_mask.sum():,}")
    print(f"  Validation nodes: {val_mask.sum():,}")
    print(f"  Test nodes: {test_mask.sum():,}")
    
    # Define models to train
    models_config = {
        'graphsage': {
            'type': 'sage',
            'params': {
                'hidden_channels': args.hidden_channels,
                'num_layers': 3,
                'dropout': 0.5
            },
            'epochs': args.epochs,
            'lr': args.lr
        },
        'gat': {
            'type': 'gat',
            'params': {
                'hidden_channels': args.hidden_channels // 2,  # GAT uses more parameters
                'num_heads': 8,
                'dropout': 0.4
            },
            'epochs': args.epochs,
            'lr': args.lr
        },
        'transformer': {
            'type': 'transformer',
            'params': {
                'hidden_channels': args.hidden_channels // 2,
                'num_heads': 8,
                'dropout': 0.4
            },
            'epochs': args.epochs,
            'lr': args.lr
        },
        'hybrid': {
            'type': 'hybrid',
            'params': {
                'hidden_channels': args.hidden_channels // 2,
                'lstm_hidden': 32,
                'dropout': 0.4
            },
            'epochs': args.epochs,
            'lr': args.lr
        }
    }
    
    # Select models to train
    if args.model == 'all':
        models_to_train = models_config
    else:
        models_to_train = {args.model: models_config[args.model]}
    
    print(f"\nTraining models: {list(models_to_train.keys())}")
    
    # Train models
    results = train_multiple_models(
        models_to_train, data, train_mask, val_mask, test_mask,
        models_dir=args.models_dir
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.results_dir, f'fraud_detection_results_{timestamp}.json')
    
    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'best_val_auc': result['best_val_auc'],
            'test_results': {
                'auc': result['test_results']['auc'],
                'average_precision': result['test_results']['average_precision'],
                'precision': result['test_results']['precision'],
                'recall': result['test_results']['recall'],
                'f1': result['test_results']['f1'],
                'precision_at_k': result['test_results']['precision_at_k']
            },
            'report': result['report']
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate comparison plots
    plot_model_comparison(results, args.results_dir, timestamp)
    
    # Generate final report
    generate_final_report(results, args.results_dir, timestamp)
    
    print("\nFraud detection training completed!")


def plot_model_comparison(results: dict, results_dir: str, timestamp: str):
    """Generate comparison plots for all models"""
    
    models = list(results.keys())
    metrics = ['auc', 'average_precision', 'precision', 'recall', 'f1']
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model]['test_results'][metric] for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Precision at K comparison
    k_values = [10, 50, 100, 500]
    for k in k_values:
        values = []
        for model in models:
            precision_at_k = results[model]['test_results']['precision_at_k']
            if f'precision@{k}' in precision_at_k:
                values.append(precision_at_k[f'precision@{k}'])
            else:
                values.append(0)
        
        bars = axes[-1].bar([f'{model}\n@{k}' for model in models], values, 
                           alpha=0.7, label=f'P@{k}')
    
    axes[-1].set_title('Precision at K Comparison')
    axes[-1].set_ylabel('Precision')
    axes[-1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'model_comparison_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def generate_final_report(results: dict, results_dir: str, timestamp: str):
    """Generate a comprehensive final report"""
    
    report_file = os.path.join(results_dir, f'final_report_{timestamp}.txt')
    
    with open(report_file, 'w') as f:
        f.write("FRAUD DETECTION SYSTEM - FINAL REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model performance summary
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        for model_name, result in results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Best Validation AUC: {result['best_val_auc']:.4f}\n")
            f.write(f"  Test AUC: {result['test_results']['auc']:.4f}\n")
            f.write(f"  Test Average Precision: {result['test_results']['average_precision']:.4f}\n")
            f.write(f"  Test Precision: {result['test_results']['precision']:.4f}\n")
            f.write(f"  Test Recall: {result['test_results']['recall']:.4f}\n")
            f.write(f"  Test F1: {result['test_results']['f1']:.4f}\n")
            
            # Precision at K
            f.write("  Precision at K:\n")
            for k, precision in result['test_results']['precision_at_k'].items():
                f.write(f"    {k}: {precision:.4f}\n")
        
        # Best model recommendation
        best_model = max(results.keys(), key=lambda x: results[x]['test_results']['auc'])
        f.write(f"\nBEST PERFORMING MODEL: {best_model.upper()}\n")
        f.write(f"Test AUC: {results[best_model]['test_results']['auc']:.4f}\n")
        
        # Fraud detection insights
        f.write("\nFRAUD DETECTION INSIGHTS\n")
        f.write("-" * 25 + "\n")
        
        for model_name, result in results.items():
            insights = result['report']['fraud_insights']
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Total Transactions: {insights['total_transactions']:,}\n")
            f.write(f"  Actual Fraud Count: {insights['actual_fraud_count']:,}\n")
            f.write(f"  Fraud Rate: {insights['fraud_rate']:.2%}\n")
            f.write(f"  Detection Rate: {insights['detection_rate']:.2%}\n")
    
    print(f"Final report saved to: {report_file}")


if __name__ == "__main__":
    main()


