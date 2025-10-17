#!/usr/bin/env python3
"""
Simple script to train fraud detection models and run inference
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elliptic_data_loader import create_elliptic_dataloader
from fraud_detection_models import create_fraud_detector
from fraud_detection_trainer import FraudDetectionTrainer


def main():
    print("🕸️ Crypto Fraud Detection System")
    print("=" * 50)
    
    # Load data
    print("Loading Elliptic dataset...")
    data, train_mask, val_mask, test_mask = create_elliptic_dataloader()
    
    print(f"Dataset loaded:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Training nodes: {train_mask.sum():,}")
    print(f"  Test nodes: {test_mask.sum():,}")
    
    # Create model
    print("\nCreating GraphSAGE fraud detector...")
    model = create_fraud_detector('sage', data.num_node_features, 
                                 hidden_channels=128, num_layers=3)
    
    # Train model
    print("Training model...")
    trainer = FraudDetectionTrainer(model)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    best_auc = trainer.train(
        data, train_mask, val_mask,
        epochs=100,  # Reduced for demo
        lr=0.01,
        patience=15,
        save_path='models/sage_fraud_detector.pt'
    )
    
    print(f"Training completed. Best validation AUC: {best_auc:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate(data, test_mask)
    
    print(f"Test Results:")
    print(f"  AUC: {eval_results['auc']:.4f}")
    print(f"  Average Precision: {eval_results['average_precision']:.4f}")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall: {eval_results['recall']:.4f}")
    print(f"  F1: {eval_results['f1']:.4f}")
    
    # Generate fraud report
    print("\nGenerating fraud detection report...")
    report = trainer.generate_fraud_report(eval_results, top_k=50)
    
    print(f"\nFraud Detection Insights:")
    insights = report['fraud_insights']
    print(f"  Total Transactions: {insights['total_transactions']:,}")
    print(f"  Actual Fraud Count: {insights['actual_fraud_count']:,}")
    print(f"  Fraud Rate: {insights['fraud_rate']:.2%}")
    print(f"  Detection Rate: {insights['detection_rate']:.2%}")
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'node_id': range(data.num_nodes),
        'fraud_probability': eval_results['probabilities'],
        'prediction': eval_results['predictions'],
        'true_label': eval_results['true_labels']
    })
    
    results_df['fraud_probability'] = eval_results['probabilities']
    results_df['is_suspicious'] = eval_results['probabilities'] > 0.5
    results_df['risk_level'] = pd.cut(eval_results['probabilities'], 
                                     bins=[0, 0.3, 0.7, 1.0], 
                                     labels=['Low', 'Medium', 'High'])
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'fraud_detection_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    
    print(f"Results saved to: {results_file}")
    
    # Show top suspicious transactions
    print(f"\nTop 10 Most Suspicious Transactions:")
    suspicious = results_df.nlargest(10, 'fraud_probability')
    for idx, row in suspicious.iterrows():
        print(f"  Node {int(row['node_id'])}: {row['fraud_probability']:.4f} "
              f"(Risk: {row['risk_level']}, True: {'Fraud' if row['true_label'] == 1 else 'Legitimate'})")
    
    print("\n✅ Fraud detection completed successfully!")
    print(f"📊 Dashboard available at: streamlit run streamlit_fraud_dashboard.py")


if __name__ == "__main__":
    main()


