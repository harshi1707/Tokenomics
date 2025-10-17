import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import time
import json
from datetime import datetime, timedelta
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elliptic_data_loader import create_elliptic_dataloader
from fraud_detection_models import create_fraud_detector
from fraud_detection_trainer import FraudDetectionTrainer
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === Allocation utilities and fairness metrics ===

def project_with_bounds(weights, floor=0.01, cap=0.25):
    w = np.maximum(np.array(weights, dtype=float), 0.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    w = np.clip(w, floor, cap)
    w = w / w.sum()
    return w


def gini_coefficient(weights):
    w = np.maximum(np.array(weights, dtype=float), 0.0)
    if w.sum() == 0:
        return 0.0
    w = np.sort(w)
    n = len(w)
    cumw = np.cumsum(w)
    return (n + 1 - 2 * (cumw / cumw[-1]).sum() / n)


def risk_inverse_allocation(probs, alpha=1.0, floor=0.01, cap=0.2):
    base = np.maximum(1.0 - np.array(probs, dtype=float), 0.0) ** alpha
    return project_with_bounds(base, floor, cap)


def ensemble_blend(allocations, thetas=None, floor=0.01, cap=0.25):
    if thetas is None:
        thetas = [1.0 / len(allocations)] * len(allocations)
    w = np.zeros_like(allocations[0], dtype=float)
    for a, t in zip(allocations, thetas):
        w = w + float(t) * np.maximum(np.array(a, dtype=float), 0.0)
    return project_with_bounds(w, floor, cap)


def topk_equal_with_tail(scores, k=3, tau=0.1):
    scores = np.array(scores, dtype=float)
    n = len(scores)
    k = max(1, min(k, n))
    tau = min(max(tau, 0.0), 1.0 - 1e-8)
    topk_idx = np.argsort(scores)[-k:]
    alloc = np.full(n, tau / max(n - k, 1))
    alloc[topk_idx] = (1.0 - tau) / k
    return project_with_bounds(alloc, floor=0.0, cap=1.0)


def mean_variance_objective_allocation(probs, lam=1.0, beta=0.1, floor=0.01, cap=0.25):
    # Closed-form heuristic: start uniform then penalize risk and variance
    n = len(probs)
    a = np.ones(n, dtype=float) / n
    # Gradient-like step against risk
    a = a - beta * (np.array(probs, dtype=float) - np.mean(probs))
    a = project_with_bounds(a, floor, cap)
    # Blend with uniform for variance control
    a = (1.0 - min(max(lam, 0.0), 1.0)) * a + min(max(lam, 0.0), 1.0) * (np.ones(n) / n)
    return project_with_bounds(a, floor, cap)


def cvar_allocation(probs, scenarios=50, noise_std=0.05, alpha=0.9, lam=0.5, floor=0.01, cap=0.25, seed=42):
    rng = np.random.default_rng(seed)
    p = np.array(probs, dtype=float)
    S = rng.normal(0.0, noise_std, size=(scenarios, len(p))) + p
    S = np.clip(S, 0.0, 1.0)
    # Start from risk-inverse baseline
    a = risk_inverse_allocation(p, alpha=1.0, floor=floor, cap=cap)
    # Iterative projected subgradient descent on CVaR + fairness penalty
    step = 0.1
    for _ in range(50):
        losses = S @ a
        thresh = np.quantile(losses, alpha)
        indicators = (losses >= thresh).astype(float)
        # Subgradient of CVaR wrt a approximated by average of worst-case scenarios
        worst_idx = np.argsort(losses)[int(alpha * scenarios):]
        grad_cvar = S[worst_idx].mean(axis=0) if len(worst_idx) > 0 else S.mean(axis=0)
        # Fairness gradient via variance derivative ~ 2(a - mean)
        grad_var = 2.0 * (a - a.mean())
        grad = grad_cvar + lam * grad_var
        a = a - step * grad
        a = project_with_bounds(a, floor, cap)
        step *= 0.98
    return a


def graph_aware_allocation(probs, centrality, alpha=1.0, gamma=0.7, floor=0.01, cap=0.25):
    p = np.array(probs, dtype=float)
    c = np.array(centrality, dtype=float)
    if c.ptp() > 0:
        c = (c - c.min()) / (c.max() - c.min())
    w = np.maximum(1.0 - p, 0.0) ** alpha * (1.0 - c) ** gamma
    return project_with_bounds(w, floor, cap)


def dirichlet_projection(action, alpha=1.0, eps=1e-6):
    x = np.maximum(np.array(action, dtype=float), 0.0) + eps
    x = x ** max(alpha, 1e-6)
    x = x / x.sum()
    return x


# Page configuration
st.set_page_config(
    page_title="🕸️ Crypto Exchange Fraud Detection Dashboard",
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
    .api-status {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .api-online {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .api-offline {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .cluster-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .transaction-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class APIClient:
    """Client for interacting with the simulation API"""
    
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def check_health(self):
        """Check if API is online"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_simulation_status(self):
        """Get current simulation status"""
        try:
            response = requests.get(f"{self.base_url}/simulation/status")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"status": "no_simulation"}
    
    def create_simulation(self, config):
        """Create a new simulation"""
        try:
            response = requests.post(f"{self.base_url}/simulation/create", json=config)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"success": False, "error": "API connection failed"}
    
    def start_simulation(self):
        """Start the simulation"""
        try:
            response = requests.post(f"{self.base_url}/simulation/start")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"success": False, "error": "API connection failed"}
    
    def get_simulation_stats(self):
        """Get simulation statistics"""
        try:
            response = requests.get(f"{self.base_url}/simulation/stats")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"error": "API connection failed"}
    
    def get_market_prices(self):
        """Get current market prices"""
        try:
            response = requests.get(f"{self.base_url}/market/prices")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"prices": {}, "sentiment": "unknown"}
    
    def execute_trade(self, user_id, token_symbol, trade_type, amount):
        """Execute a trade"""
        try:
            response = requests.post(f"{self.base_url}/trading/execute", json={
                "user_id": user_id,
                "token_symbol": token_symbol,
                "trade_type": trade_type,
                "amount": amount
            })
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"success": False, "error": "API connection failed"}


class ExchangeSimulator:
    """Simulates exchange activities and generates transaction data"""
    
    def __init__(self):
        self.transactions = []
        self.users = []
        self.clusters = []
        
    def generate_users(self, num_users=100):
        """Generate synthetic users with different behaviors"""
        np.random.seed(42)
        
        # Create different user types
        user_types = ['whale', 'trader', 'hodler', 'bot', 'suspicious']
        users = []
        
        for i in range(num_users):
            user_type = np.random.choice(user_types, p=[0.05, 0.4, 0.3, 0.2, 0.05])
            
            if user_type == 'whale':
                balance = np.random.lognormal(12, 1)  # Large balances
                activity = np.random.beta(2, 5)  # Low activity
            elif user_type == 'trader':
                balance = np.random.lognormal(8, 0.5)  # Medium balances
                activity = np.random.beta(8, 2)  # High activity
            elif user_type == 'hodler':
                balance = np.random.lognormal(9, 0.3)  # Medium balances
                activity = np.random.beta(1, 9)  # Very low activity
            elif user_type == 'bot':
                balance = np.random.lognormal(7, 0.4)  # Small-medium balances
                activity = np.random.beta(9, 1)  # Very high activity
            else:  # suspicious
                balance = np.random.lognormal(6, 1)  # Variable balances
                activity = np.random.beta(5, 3)  # Medium-high activity
            
            users.append({
                'user_id': i,
                'address': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
                'user_type': user_type,
                'balance': balance,
                'activity_level': activity,
                'fraud_probability': np.random.beta(1, 9) if user_type != 'suspicious' else np.random.beta(5, 2)
            })
        
        self.users = users
        return users
    
    def simulate_exchange_activity(self, duration_hours=24):
        """Simulate exchange activity and generate transactions"""
        transactions = []
        current_time = datetime.now()
        
        for user in self.users:
            # Generate transactions based on user type and activity level
            num_transactions = int(user['activity_level'] * 50 * (duration_hours / 24))
            
            for _ in range(num_transactions):
                # Random time within the duration
                transaction_time = current_time + timedelta(
                    hours=np.random.uniform(0, duration_hours),
                    minutes=np.random.uniform(0, 60)
                )
                
                # Generate transaction features
                amount = np.random.lognormal(6, 2) if user['user_type'] == 'whale' else np.random.lognormal(4, 1)
                
                # Create transaction
                transaction = {
                    'tx_id': len(transactions),
                    'user_id': user['user_id'],
                    'from_address': user['address'],
                    'to_address': np.random.choice([u['address'] for u in self.users if u['user_id'] != user['user_id']]),
                    'amount': amount,
                    'token_symbol': np.random.choice(['BTC', 'ETH', 'USDT', 'BNB', 'ADA']),
                    'timestamp': transaction_time,
                    'gas_price': np.random.lognormal(2, 0.5),
                    'gas_used': np.random.lognormal(8, 0.3),
                    'block_number': np.random.randint(18000000, 19000000),
                    'transaction_fee': np.random.lognormal(1, 0.5),
                    'fraud_probability': user['fraud_probability'] + np.random.normal(0, 0.1),
                    'user_type': user['user_type']
                }
                
                transactions.append(transaction)
        
        # Sort by timestamp
        self.transactions = sorted(transactions, key=lambda x: x['timestamp'])
        return self.transactions
    
    def perform_clustering(self, method='dbscan'):
        """Perform clustering on transaction data"""
        if not self.transactions:
            return []
        
        # Prepare features for clustering
        features = []
        for tx in self.transactions:
            features.append([
                tx['amount'],
                tx['gas_price'],
                tx['gas_used'],
                tx['transaction_fee'],
                tx['fraud_probability'],
                len(tx['from_address']),  # Address length as feature
                len(tx['to_address']),
                tx['timestamp'].hour,  # Time-based features
                tx['timestamp'].minute
            ])
        
        features = np.array(features)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Perform clustering
        if method == 'dbscan':
            clustering = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'spectral':
            clustering = SpectralClustering(n_clusters=8, random_state=42)
        else:  # agglomerative
            clustering = AgglomerativeClustering(n_clusters=8)
        
        cluster_labels = clustering.fit_predict(features)
        
        # Add cluster labels to transactions
        for i, tx in enumerate(self.transactions):
            tx['cluster_id'] = int(cluster_labels[i])
            tx['is_outlier'] = cluster_labels[i] == -1
        
        # Analyze clusters
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip outliers
                continue
                
            cluster_transactions = [tx for tx in self.transactions if tx['cluster_id'] == cluster_id]
            
            cluster_analysis = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_transactions),
                'avg_fraud_prob': np.mean([tx['fraud_probability'] for tx in cluster_transactions]),
                'avg_amount': np.mean([tx['amount'] for tx in cluster_transactions]),
                'dominant_user_type': max(set([tx['user_type'] for tx in cluster_transactions]), 
                                        key=[tx['user_type'] for tx in cluster_transactions].count),
                'risk_level': 'High' if np.mean([tx['fraud_probability'] for tx in cluster_transactions]) > 0.7 else
                             'Medium' if np.mean([tx['fraud_probability'] for tx in cluster_transactions]) > 0.4 else 'Low',
                'transactions': cluster_transactions
            }
            clusters.append(cluster_analysis)
        
        self.clusters = clusters
        return clusters


@st.cache_data
def load_fraud_detection_data():
    """Load fraud detection models and data"""
    try:
        # Load Elliptic data
        data, train_mask, val_mask, test_mask = create_elliptic_dataloader()
        
        # Load or create model
        model = create_fraud_detector('gat', data.num_node_features, 
                                    hidden_channels=64, num_heads=8)
        
        # Try to load pre-trained weights
        model_path = "models/gat_fraud_detector.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        model.eval()
        
        # Run inference
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
        
        return data, probabilities, model
    except Exception as e:
        st.error(f"Error loading fraud detection data: {e}")
        return None, None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">🕸️ Crypto Exchange Fraud Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides real-time fraud detection, exchange simulation, and cluster analysis 
    for crypto transactions. It integrates with the simulation API to provide live updates.
    """)
    
    # Initialize API client and simulator
    api_client = APIClient()
    simulator = ExchangeSimulator()
    
    # Check API status
    api_online = api_client.check_health()
    if api_online:
        st.markdown('<div class="api-status api-online">✅ API Online - Connected to Simulation Server</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-offline">❌ API Offline - Running in Demo Mode</div>', 
                   unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Simulation controls
    st.sidebar.subheader("🔄 Exchange Simulation")
    
    if st.sidebar.button("🚀 Start New Simulation", type="primary"):
        with st.spinner("Starting simulation..."):
            config = {
                "name": f"fraud_detection_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "duration_hours": 24,
                "user_count": 100,
                "trading_frequency": 0.2,
                "auto_trading": True
            }
            
            if api_online:
                result = api_client.create_simulation(config)
                if result.get('success'):
                    st.success("✅ Simulation created successfully!")
                    api_client.start_simulation()
                else:
                    st.error(f"❌ Failed to create simulation: {result.get('error')}")
            else:
                st.info("Running in demo mode - generating synthetic data...")
                simulator.generate_users(100)
                simulator.simulate_exchange_activity(24)
                st.success("✅ Demo simulation completed!")
    
    # Clustering options
    st.sidebar.subheader("🔍 Clustering Options")
    clustering_method = st.sidebar.selectbox(
        "Clustering Method",
        ["dbscan", "spectral", "agglomerative"],
        help="Choose the clustering algorithm"
    )
    
    if st.sidebar.button("🔍 Analyze Clusters"):
        with st.spinner("Performing cluster analysis..."):
            clusters = simulator.perform_clustering(method=clustering_method)
            st.success(f"✅ Found {len(clusters)} clusters!")
    
    # Fraud detection options
    st.sidebar.subheader("🕸️ Fraud Detection")
    fraud_threshold = st.sidebar.slider(
        "Fraud Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    # Main dashboard content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔄 Exchange Simulation", "🔍 Cluster Analysis", "🕸️ Fraud Detection"])
    
    with tab1:
        st.subheader("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Get simulation stats
        if api_online:
            stats = api_client.get_simulation_stats()
            if 'error' not in stats:
                with col1:
                    st.metric("Active Users", stats.get('active_users', 0))
                with col2:
                    st.metric("Total Transactions", stats.get('total_transactions', 0))
                with col3:
                    st.metric("Market Volume", f"${stats.get('total_volume', 0):,.2f}")
                with col4:
                    st.metric("API Status", "🟢 Online")
            else:
                st.info("No active simulation running")
        else:
            with col1:
                st.metric("Demo Users", len(simulator.users))
            with col2:
                st.metric("Demo Transactions", len(simulator.transactions))
            with col3:
                st.metric("API Status", "🔴 Offline")
            with col4:
                st.metric("Mode", "Demo")
        
        # Market prices
        if api_online:
            market_data = api_client.get_market_prices()
            if market_data.get('prices'):
                st.subheader("💰 Current Market Prices")
                prices_df = pd.DataFrame(list(market_data['prices'].items()), 
                                       columns=['Token', 'Price'])
                st.dataframe(prices_df, use_container_width=True)
    
    with tab2:
        st.subheader("🔄 Exchange Simulation")
        
        if simulator.transactions:
            st.success(f"✅ Generated {len(simulator.transactions)} transactions")
            
            # Transaction timeline
            st.subheader("📈 Transaction Timeline")
            
            # Create timeline data
            timeline_data = []
            for tx in simulator.transactions[:100]:  # Show first 100 for performance
                timeline_data.append({
                    'timestamp': tx['timestamp'],
                    'amount': tx['amount'],
                    'user_type': tx['user_type'],
                    'fraud_prob': tx['fraud_probability'],
                    'token': tx['token_symbol']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            # Timeline chart
            fig_timeline = px.scatter(
                timeline_df, 
                x='timestamp', 
                y='amount',
                color='user_type',
                size='fraud_prob',
                hover_data=['token', 'fraud_prob'],
                title="Transaction Timeline - Amount vs Time"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Recent transactions table
            st.subheader("📋 Recent Transactions")
            recent_txs = pd.DataFrame(simulator.transactions[-20:])  # Last 20 transactions
            st.dataframe(recent_txs[['tx_id', 'user_id', 'amount', 'token_symbol', 
                                   'fraud_probability', 'user_type']], use_container_width=True)
            
            # Execute manual trade
            st.subheader("💱 Execute Manual Trade")
            col1, col2 = st.columns(2)
            
            with col1:
                user_id = st.selectbox("User ID", options=list(range(len(simulator.users))))
                token_symbol = st.selectbox("Token", ["BTC", "ETH", "USDT", "BNB", "ADA"])
                trade_type = st.selectbox("Trade Type", ["buy", "sell"])
            
            with col2:
                amount = st.number_input("Amount", min_value=0.01, value=1.0, step=0.1)
                
                if st.button("Execute Trade"):
                    if api_online:
                        result = api_client.execute_trade(user_id, token_symbol, trade_type, amount)
                        if result.get('success'):
                            st.success(f"✅ Trade executed! Trade ID: {result.get('trade_id')}")
                        else:
                            st.error(f"❌ Trade failed: {result.get('error')}")
                    else:
                        # Simulate trade execution
                        new_tx = {
                            'tx_id': len(simulator.transactions),
                            'user_id': user_id,
                            'amount': amount,
                            'token_symbol': token_symbol,
                            'trade_type': trade_type,
                            'timestamp': datetime.now(),
                            'fraud_probability': simulator.users[user_id]['fraud_probability']
                        }
                        simulator.transactions.append(new_tx)
                        st.success("✅ Demo trade executed!")
        else:
            st.info("No simulation data available. Start a simulation to see exchange activity.")
    
    with tab3:
        st.subheader("🔍 Cluster Analysis")
        
        if simulator.clusters:
            st.success(f"✅ Analyzed {len(simulator.clusters)} clusters")
            
            # Cluster overview
            cluster_summary = []
            for cluster in simulator.clusters:
                cluster_summary.append({
                    'Cluster ID': cluster['cluster_id'],
                    'Size': cluster['size'],
                    'Avg Fraud Prob': f"{cluster['avg_fraud_prob']:.3f}",
                    'Avg Amount': f"${cluster['avg_amount']:.2f}",
                    'Dominant Type': cluster['dominant_user_type'],
                    'Risk Level': cluster['risk_level']
                })
            
            st.subheader("📊 Cluster Summary")
            st.dataframe(pd.DataFrame(cluster_summary), use_container_width=True)
            
            # Cluster visualization
            st.subheader("🎯 Cluster Visualization")
            
            # Prepare data for visualization
            all_features = []
            all_labels = []
            all_fraud_probs = []
            
            for cluster in simulator.clusters:
                for tx in cluster['transactions']:
                    all_features.append([tx['amount'], tx['fraud_probability']])
                    all_labels.append(cluster['cluster_id'])
                    all_fraud_probs.append(tx['fraud_probability'])
            
            features_df = pd.DataFrame(all_features, columns=['Amount', 'Fraud_Probability'])
            features_df['Cluster'] = all_labels
            features_df['Fraud_Prob'] = all_fraud_probs
            
            # Create scatter plot
            fig_clusters = px.scatter(
                features_df,
                x='Amount',
                y='Fraud_Probability',
                color='Cluster',
                size='Fraud_Prob',
                title="Transaction Clusters - Amount vs Fraud Probability",
                hover_data=['Fraud_Prob']
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Detailed cluster analysis
            st.subheader("🔬 Detailed Cluster Analysis")
            
            for cluster in simulator.clusters:
                with st.expander(f"Cluster {cluster['cluster_id']} - {cluster['risk_level']} Risk ({cluster['size']} transactions)"):
                    st.markdown(f"""
                    **Cluster Statistics:**
                    - Size: {cluster['size']} transactions
                    - Average Fraud Probability: {cluster['avg_fraud_prob']:.3f}
                    - Average Amount: ${cluster['avg_amount']:.2f}
                    - Dominant User Type: {cluster['dominant_user_type']}
                    - Risk Level: {cluster['risk_level']}
                    """)
                    
                    # Show sample transactions
                    sample_txs = pd.DataFrame(cluster['transactions'][:10])
                    st.dataframe(sample_txs[['tx_id', 'amount', 'token_symbol', 'fraud_probability', 'user_type']])
        else:
            st.info("No cluster data available. Run cluster analysis after starting a simulation.")
    
    with tab4:
        st.subheader("🕸️ Fraud Detection Analysis")
        
        # Load fraud detection data
        data, probabilities, model = load_fraud_detection_data()
        
        if data is not None and probabilities is not None:
            st.success("✅ Fraud detection model loaded successfully")
            
            # Fraud statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_risk = (probabilities > 0.8).sum()
                st.metric("High Risk Transactions", f"{high_risk:,}")
            
            with col2:
                medium_risk = ((probabilities > 0.5) & (probabilities <= 0.8)).sum()
                st.metric("Medium Risk Transactions", f"{medium_risk:,}")
            
            with col3:
                suspicious = (probabilities > fraud_threshold).sum()
                st.metric("Suspicious Transactions", f"{suspicious:,}")
            
            with col4:
                avg_fraud_prob = probabilities.mean()
                st.metric("Average Fraud Probability", f"{avg_fraud_prob:.3f}")
            
            # Fraud probability distribution
            st.subheader("📈 Fraud Probability Distribution")
            
            fig_fraud_dist = px.histogram(
                x=probabilities,
                nbins=50,
                title="Distribution of Fraud Probabilities",
                labels={'x': 'Fraud Probability', 'y': 'Count'}
            )
            fig_fraud_dist.add_vline(x=fraud_threshold, line_dash="dash", 
                                   annotation_text=f"Threshold: {fraud_threshold}")
            st.plotly_chart(fig_fraud_dist, use_container_width=True)
            
            # Top suspicious transactions
            st.subheader("🚨 Top Suspicious Transactions")
            
            # Create suspicious transactions dataframe
            suspicious_indices = np.argsort(probabilities)[::-1]
            top_k = 20
            
            suspicious_data = []
            for i in range(min(top_k, len(suspicious_indices))):
                idx = suspicious_indices[i]
                suspicious_data.append({
                    'Rank': i + 1,
                    'Transaction ID': int(idx),
                    'Fraud Probability': probabilities[idx],
                    'Risk Level': 'High' if probabilities[idx] > 0.8 else 
                                 'Medium' if probabilities[idx] > 0.5 else 'Low'
                })
            
            st.dataframe(pd.DataFrame(suspicious_data), use_container_width=True)
            
            # Network visualization (if we have graph data)
            st.subheader("🕸️ Transaction Network")
            
            if hasattr(data, 'edge_index') and data.edge_index.shape[1] > 0:
                # Create a sample subgraph for visualization
                sample_nodes = np.random.choice(data.num_nodes, min(100, data.num_nodes), replace=False)
                
                # Create network visualization
                G = nx.Graph()
                
                # Add nodes
                for node in sample_nodes:
                    G.add_node(node, fraud_prob=probabilities[node])
                
                # Add edges (sample)
                edge_sample = data.edge_index[:, :min(200, data.edge_index.shape[1])]
                for i in range(edge_sample.shape[1]):
                    src, dst = edge_sample[0, i].item(), edge_sample[1, i].item()
                    if src in sample_nodes and dst in sample_nodes:
                        G.add_edge(src, dst)
                
                # Create network plot
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Extract coordinates
                x_coords = [pos[node][0] for node in G.nodes()]
                y_coords = [pos[node][1] for node in G.nodes()]
                node_colors = [G.nodes[node]['fraud_prob'] for node in G.nodes()]
                
                fig_network = go.Figure()
                
                # Add edges
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                fig_network.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    name='edges'
                ))
                
                # Add nodes
                fig_network.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers',
                    hoverinfo='text',
                    text=[f'Node: {node}<br>Fraud Score: {node_colors[i]:.3f}' 
                          for i, node in enumerate(G.nodes())],
                    marker=dict(
                        size=15,
                        color=node_colors,
                        colorscale='Viridis',
                        colorbar=dict(title="Fraud Probability"),
                        line=dict(width=2, color='black')
                    ),
                    name='nodes'
                ))
                
                fig_network.update_layout(
                    title='Transaction Network - Fraud Risk Visualization',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("Network visualization not available for current dataset")
        else:
            st.error("❌ Failed to load fraud detection data")
        
        # === Token Optimization (Reinforcement + Heuristic Strategies) ===
        st.subheader("🧮 Token Optimization Strategies")
        st.markdown("Select a strategy to derive fair, risk-aware allocations for top suspicious nodes.")
        
        # Prepare top suspicious nodes from probabilities if available
        if data is not None and probabilities is not None:
            n_show = st.slider("Number of top suspicious nodes", min_value=5, max_value=50, value=10, step=1)
            sorted_idx = np.argsort(probabilities)[::-1][:n_show]
            top_probs = probabilities[sorted_idx]
            top_nodes = sorted_idx
            
            # Strategy selection
            strategy = st.selectbox(
                "Allocation strategy",
                [
                    "Risk-inverse proportional",
                    "Mean-variance fairness",
                    "CVaR-constrained",
                    "Graph-aware fairness",
                    "Dirichlet projection of RL action",
                    "Ensemble of strategies"
                ]
            )
            
            colA, colB, colC = st.columns(3)
            with colA:
                floor = st.number_input("Floor (ε)", min_value=0.0, max_value=0.2, value=0.01, step=0.005, format="%.3f")
                cap = st.number_input("Cap (c)", min_value=0.05, max_value=1.0, value=0.25, step=0.05, format="%.2f")
            with colB:
                alpha_val = st.number_input("Alpha (risk/Dirichlet power)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
                lam = st.number_input("Lambda (fairness strength)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            with colC:
                beta = st.number_input("Beta (risk penalty)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
                tau = st.number_input("Tail mass (for discrete/top-k)", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
            
            allocation = None
            details = {}
            
            # Graph centrality if needed
            centrality_values = np.zeros_like(top_probs)
            if strategy in ("Graph-aware fairness", "Ensemble of strategies"):
                # Build a small k-NN graph in feature space of probabilities as a proxy if true graph is not easily subselected
                G_tmp = nx.Graph()
                for i, node in enumerate(top_nodes):
                    G_tmp.add_node(int(node))
                # connect by nearest neighbors in index order for simplicity
                for i in range(len(top_nodes) - 1):
                    G_tmp.add_edge(int(top_nodes[i]), int(top_nodes[i+1]))
                cent = nx.betweenness_centrality(G_tmp, normalized=True)
                centrality_values = np.array([cent.get(int(n), 0.0) for n in top_nodes])
            
            if strategy == "Risk-inverse proportional":
                allocation = risk_inverse_allocation(top_probs, alpha=alpha_val, floor=floor, cap=cap)
                details["Gini"] = gini_coefficient(allocation)
            elif strategy == "Mean-variance fairness":
                allocation = mean_variance_objective_allocation(top_probs, lam=lam, beta=beta, floor=floor, cap=cap)
                details["Gini"] = gini_coefficient(allocation)
            elif strategy == "CVaR-constrained":
                allocation = cvar_allocation(top_probs, scenarios=100, noise_std=0.07, alpha=0.9, lam=lam, floor=floor, cap=cap)
                details["Gini"] = gini_coefficient(allocation)
            elif strategy == "Graph-aware fairness":
                allocation = graph_aware_allocation(top_probs, centrality_values, alpha=alpha_val, gamma=0.7, floor=floor, cap=cap)
                details["Gini"] = gini_coefficient(allocation)
            elif strategy == "Dirichlet projection of RL action":
                # If RL actions were available from another module, we would import them; here derive a proxy from inverse risk
                base = 1.0 - top_probs
                a0 = np.maximum(base, 1e-6) / np.maximum(base.sum(), 1e-6)
                a0 = dirichlet_projection(a0, alpha=alpha_val)
                allocation = project_with_bounds(a0, floor=floor, cap=cap)
                details["Gini"] = gini_coefficient(allocation)
            else:  # Ensemble of strategies
                a1 = risk_inverse_allocation(top_probs, alpha=alpha_val, floor=floor, cap=cap)
                a2 = mean_variance_objective_allocation(top_probs, lam=lam, beta=beta, floor=floor, cap=cap)
                a3 = graph_aware_allocation(top_probs, centrality_values, alpha=alpha_val, gamma=0.7, floor=floor, cap=cap)
                allocation = ensemble_blend([a1, a2, a3], thetas=[0.5, 0.25, 0.25], floor=floor, cap=cap)
                details["Gini"] = gini_coefficient(allocation)
            
            # Display results
            if allocation is not None:
                res_df = pd.DataFrame({
                    'Node': top_nodes.astype(int),
                    'Fraud_Prob': top_probs,
                    'Allocation': allocation
                }).sort_values('Fraud_Prob', ascending=False)
                st.markdown("**Optimized Allocations**")
                st.dataframe(res_df, use_container_width=True)
                
                fig_alloc = px.bar(res_df, x=res_df['Node'].astype(str), y='Allocation', title=f"Allocation - {strategy}")
                st.plotly_chart(fig_alloc, use_container_width=True)
                
                st.caption(f"Gini fairness index: {details.get('Gini', 0.0):.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        Crypto Exchange Fraud Detection Dashboard | 
        API Status: {'🟢 Online' if api_online else '🔴 Offline'} | 
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

