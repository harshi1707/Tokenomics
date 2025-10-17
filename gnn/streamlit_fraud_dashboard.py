import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elliptic_data_loader import create_elliptic_dataloader
from fraud_detection_models import create_fraud_detector


# Page configuration
st.set_page_config(
    page_title="Crypto Fraud Detection Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-metric {
        color: #4caf50;
        font-weight: bold;
    }
    .warning-metric {
        color: #ff9800;
        font-weight: bold;
    }
    .danger-metric {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the Elliptic dataset"""
    try:
        data, train_mask, val_mask, test_mask = create_elliptic_dataloader()
        return data, train_mask, val_mask, test_mask
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


@st.cache_resource
def load_model(model_type, data):
    """Load and cache trained models"""
    try:
        model = create_fraud_detector(model_type, data.num_node_features)
        
        # Try to load pre-trained weights
        model_path = f"models/{model_type}.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            st.success(f"✅ Loaded pre-trained {model_type} model")
        else:
            st.warning(f"⚠️ No pre-trained weights found for {model_type}. Using random initialization.")
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_network_visualization(data, node_indices, edge_index, node_colors=None, 
                                node_sizes=None, layout='spring'):
    """Create an interactive network visualization"""
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i, node_idx in enumerate(node_indices):
        G.add_node(node_idx, 
                  color=node_colors[i] if node_colors is not None else 'lightblue',
                  size=node_sizes[i] if node_sizes is not None else 10)
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src = node_indices[edge_index[0, i].item()]
        dst = node_indices[edge_index[1, i].item()]
        G.add_edge(src, dst)
    
    # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # Extract coordinates
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        hoverinfo='text',
        text=[f'Node: {node}<br>Fraud Score: {node_colors[i]:.3f}' 
              for i, node in enumerate(G.nodes())],
        marker=dict(
            size=node_sizes if node_sizes is not None else 15,
            color=node_colors if node_colors is not None else 'lightblue',
            colorscale='Viridis',
            colorbar=dict(title="Fraud Probability"),
            line=dict(width=2, color='black')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Transaction Network Visualization',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive network showing suspicious transactions",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='gray', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🕸️ Crypto Fraud Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides real-time fraud detection on the Elliptic Bitcoin transaction dataset 
    using state-of-the-art Graph Neural Networks. Analyze suspicious patterns, investigate 
    specific transactions, and visualize network structures.
    """)
    
    # Load data
    with st.spinner("Loading Elliptic dataset..."):
        data, train_mask, val_mask, test_mask = load_data()
    
    if data is None:
        st.error("Failed to load dataset. Please check your data directory.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["sage", "gat", "transformer", "hybrid"],
        help="Choose the GNN architecture for fraud detection"
    )
    
    # Threshold settings
    fraud_threshold = st.sidebar.slider(
        "Fraud Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold above which transactions are flagged as suspicious"
    )
    
    # Visualization settings
    st.sidebar.subheader("📊 Visualization Settings")
    
    max_nodes = st.sidebar.slider(
        "Max Nodes in Network View",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )
    
    k_hop = st.sidebar.slider(
        "Neighborhood Radius (k-hop)",
        min_value=1,
        max_value=3,
        value=2
    )
    
    # Load model
    model = load_model(model_type, data)
    if model is None:
        st.error("Failed to load model.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Fraud Detection Results")
        
        # Run inference
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
        
        # Calculate metrics
        test_probs = probabilities[test_mask]
        test_labels = data.y[test_mask]
        test_labels_binary = (test_labels == 1).float().numpy()
        
        # Filter out unknown labels
        known_mask = test_labels != -1
        known_probs = test_probs[known_mask]
        known_labels = test_labels_binary[known_mask]
        
        if len(known_probs) > 0:
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            auc = roc_auc_score(known_labels, known_probs)
            ap = average_precision_score(known_labels, known_probs)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("ROC-AUC", f"{auc:.4f}", help="Area under the ROC curve")
            
            with metric_col2:
                st.metric("Average Precision", f"{ap:.4f}", help="Area under the Precision-Recall curve")
            
            with metric_col3:
                fraud_count = int((probabilities > fraud_threshold).sum())
                st.metric("Suspicious Transactions", f"{fraud_count:,}", 
                         help=f"Transactions with fraud probability > {fraud_threshold}")
            
            with metric_col4:
                high_risk_count = int((probabilities > 0.8).sum())
                st.metric("High Risk Transactions", f"{high_risk_count:,}",
                         help="Transactions with fraud probability > 0.8")
        
        # Fraud probability distribution
        st.subheader("📈 Fraud Probability Distribution")
        
        fig_hist = px.histogram(
            x=probabilities,
            nbins=50,
            title="Distribution of Fraud Probabilities",
            labels={'x': 'Fraud Probability', 'y': 'Count'}
        )
        fig_hist.add_vline(x=fraud_threshold, line_dash="dash", 
                          annotation_text=f"Threshold: {fraud_threshold}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Top Suspicious Transactions")
        
        # Get top suspicious transactions
        suspicious_indices = np.argsort(probabilities)[::-1]
        top_k = 20
        
        suspicious_data = []
        for i in range(min(top_k, len(suspicious_indices))):
            idx = suspicious_indices[i]
            suspicious_data.append({
                'Rank': i + 1,
                'Node ID': int(idx),
                'Fraud Probability': probabilities[idx],
                'True Label': 'Unknown' if data.y[idx] == -1 else 
                            ('Illicit' if data.y[idx] == 1 else 'Licit')
            })
        
        df_suspicious = pd.DataFrame(suspicious_data)
        st.dataframe(df_suspicious, use_container_width=True)
    
    # Node investigation
    st.subheader("🔎 Transaction Investigation")
    
    col_invest1, col_invest2 = st.columns([1, 2])
    
    with col_invest1:
        # Node selector
        selected_node = st.number_input(
            "Enter Node ID to Investigate",
            min_value=0,
            max_value=data.num_nodes - 1,
            value=int(suspicious_indices[0]),
            step=1
        )
        
        if selected_node < data.num_nodes:
            node_prob = probabilities[selected_node]
            node_label = data.y[selected_node]
            
            st.markdown(f"""
            **Node {selected_node} Details:**
            - Fraud Probability: {node_prob:.4f}
            - True Label: {'Unknown' if node_label == -1 else ('Illicit' if node_label == 1 else 'Licit')}
            - Risk Level: {'🔴 High' if node_prob > 0.8 else '🟡 Medium' if node_prob > 0.5 else '🟢 Low'}
            """)
    
    with col_invest2:
        # Network visualization of selected node's neighborhood
        if selected_node < data.num_nodes:
            try:
                # Get k-hop subgraph
                subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                    selected_node, k_hop, data.edge_index, relabel_nodes=True
                )
                
                if len(subset) > 0:
                    # Limit nodes for visualization
                    if len(subset) > max_nodes:
                        subset = subset[:max_nodes]
                        edge_index_sub = edge_index_sub[:, 
                                                      torch.isin(edge_index_sub[0], torch.arange(len(subset))) &
                                                      torch.isin(edge_index_sub[1], torch.arange(len(subset)))]
                    
                    # Get node colors and sizes
                    node_probs = probabilities[subset.numpy()]
                    node_sizes = np.maximum(10, 30 * node_probs)
                    
                    # Create visualization
                    fig_network = create_network_visualization(
                        data, subset.numpy(), edge_index_sub, 
                        node_colors=node_probs, node_sizes=node_sizes
                    )
                    
                    st.plotly_chart(fig_network, use_container_width=True)
                    
                    # Subgraph statistics
                    st.markdown(f"""
                    **Neighborhood Statistics:**
                    - Nodes in subgraph: {len(subset)}
                    - Edges in subgraph: {edge_index_sub.shape[1]}
                    - Average fraud probability: {node_probs.mean():.4f}
                    - Max fraud probability: {node_probs.max():.4f}
                    """)
                else:
                    st.info("No neighbors found for this node.")
                    
            except Exception as e:
                st.error(f"Error creating network visualization: {e}")
    
    # Performance analysis
    st.subheader("📊 Model Performance Analysis")
    
    if len(known_probs) > 0:
        # ROC Curve
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        fpr, tpr, _ = roc_curve(known_labels, known_probs)
        precision, recall, _ = precision_recall_curve(known_labels, known_probs)
        
        # Create subplots
        fig_perf = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curve
        fig_perf.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {auc:.3f})', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig_perf.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random', 
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        fig_perf.add_trace(
            go.Scatter(x=recall, y=precision, name=f'PR (AP = {ap:.3f})', 
                      line=dict(color='green')),
            row=1, col=2
        )
        
        fig_perf.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig_perf.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig_perf.update_xaxes(title_text="Recall", row=1, col=2)
        fig_perf.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig_perf.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Download results
    st.subheader("💾 Export Results")
    
    # Create downloadable data
    results_df = pd.DataFrame({
        'Node_ID': range(data.num_nodes),
        'Fraud_Probability': probabilities,
        'True_Label': data.y.numpy(),
        'Label_Text': ['Unknown' if y == -1 else ('Illicit' if y == 1 else 'Licit') for y in data.y.numpy()]
    })
    
    # Add suspicious flag
    results_df['Suspicious'] = results_df['Fraud_Probability'] > fraud_threshold
    
    # Download buttons
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download All Results (CSV)",
        data=csv_data,
        file_name=f"fraud_detection_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.subheader("📋 Summary Statistics")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown(f"""
        **Dataset Overview:**
        - Total Transactions: {data.num_nodes:,}
        - Total Connections: {data.num_edges:,}
        - Feature Dimensions: {data.num_node_features}
        - Known Labels: {known_mask.sum():,}
        - Licit Transactions: {(data.y == 0).sum():,}
        - Illicit Transactions: {(data.y == 1).sum():,}
        """)
    
    with summary_col2:
        st.markdown(f"""
        **Detection Results:**
        - Model Used: {model_type.upper()}
        - Fraud Threshold: {fraud_threshold}
        - Suspicious Transactions: {fraud_count:,}
        - Detection Rate: {(fraud_count / data.num_nodes * 100):.2f}%
        - High Risk Transactions: {high_risk_count:,}
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        Crypto Fraud Detection Dashboard | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


