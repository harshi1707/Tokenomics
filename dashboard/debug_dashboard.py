import streamlit as st
import requests
import yaml
import os
import time
import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Page configuration
st.set_page_config(
    page_title='🚀 Debug Crypto Dashboard', 
    page_icon='🚀',
    layout='wide'
)

st.title('🚀 Debug Crypto Trading & Fraud Detection Dashboard')

# Debug information
st.header("🔧 Debug Information")

# Check imports
st.subheader("Import Status")
try:
    import plotly.graph_objects as go
    import plotly.express as px
    st.success("✅ Plotly imported successfully")
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Plotly import failed: {e}")
    PLOTLY_AVAILABLE = False

try:
    sys.path.append(os.path.join(project_root, 'gnn'))
    from elliptic_data_loader import create_elliptic_dataloader
    st.success("✅ GNN modules imported successfully")
    GNN_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ GNN modules not available: {e}")
    GNN_AVAILABLE = False

# Check config
st.subheader("Configuration")
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    st.success("✅ Config loaded successfully")
    API_BASE = f"http://{cfg['api']['host']}:{cfg['api']['port']}"
    st.info(f"API Base URL: {API_BASE}")
except Exception as e:
    st.error(f"❌ Config loading failed: {e}")
    API_BASE = "http://127.0.0.1:8000"

# Check API connection
st.subheader("API Connection")
try:
    response = requests.get(f"{API_BASE}/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ API is online and responding")
        api_online = True
    else:
        st.warning(f"⚠️ API responded with status {response.status_code}")
        api_online = False
except Exception as e:
    st.error(f"❌ API connection failed: {e}")
    api_online = False

# Simple demo functionality
st.header("🎮 Simple Demo")

# Basic simulation
if st.button("Generate Demo Data"):
    st.success("✅ Demo data generated!")
    
    # Create some sample data
    np.random.seed(42)
    demo_users = []
    demo_transactions = []
    
    for i in range(20):
        user_type = np.random.choice(['whale', 'trader', 'hodler', 'bot', 'suspicious'])
        demo_users.append({
            'user_id': i,
            'user_type': user_type,
            'balance': np.random.lognormal(8, 1),
            'activity_level': np.random.beta(3, 3),
            'fraud_probability': np.random.beta(2, 8) if user_type != 'suspicious' else np.random.beta(6, 2)
        })
    
    for i in range(100):
        user_id = np.random.randint(0, len(demo_users))
        demo_transactions.append({
            'tx_id': i,
            'user_id': user_id,
            'amount': np.random.lognormal(4, 1),
            'token_symbol': np.random.choice(['BTC', 'ETH', 'USDT', 'BNB']),
            'timestamp': datetime.now() - timedelta(hours=np.random.uniform(0, 24)),
            'fraud_probability': demo_users[user_id]['fraud_probability'] + np.random.normal(0, 0.1)
        })
    
    st.session_state.demo_users = demo_users
    st.session_state.demo_transactions = demo_transactions

# Display demo data if available
if 'demo_users' in st.session_state:
    st.subheader("📊 Demo Data Analysis")
    
    users_df = pd.DataFrame(st.session_state.demo_users)
    transactions_df = pd.DataFrame(st.session_state.demo_transactions)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(users_df))
    
    with col2:
        st.metric("Total Transactions", len(transactions_df))
    
    with col3:
        high_risk = sum(1 for tx in st.session_state.demo_transactions if tx['fraud_probability'] > 0.7)
        st.metric("High Risk Transactions", high_risk)
    
    with col4:
        avg_fraud_prob = np.mean([tx['fraud_probability'] for tx in st.session_state.demo_transactions])
        st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.3f}")
    
    # User type distribution
    st.subheader("👥 User Type Distribution")
    user_type_counts = users_df['user_type'].value_counts()
    st.bar_chart(user_type_counts)
    
    # Transaction overview
    st.subheader("💱 Transaction Overview")
    st.dataframe(transactions_df.head(10))
    
    # Fraud probability distribution
    if PLOTLY_AVAILABLE:
        st.subheader("📈 Fraud Probability Distribution")
        fraud_probs = [tx['fraud_probability'] for tx in st.session_state.demo_transactions]
        
        fig = px.histogram(
            x=fraud_probs,
            nbins=20,
            title="Distribution of Fraud Probabilities",
            labels={'x': 'Fraud Probability', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top suspicious transactions
    st.subheader("🚨 Top Suspicious Transactions")
    suspicious_txs = sorted(st.session_state.demo_transactions, 
                           key=lambda x: x['fraud_probability'], reverse=True)[:10]
    
    suspicious_data = []
    for i, tx in enumerate(suspicious_txs):
        suspicious_data.append({
            'Rank': i + 1,
            'TX ID': tx['tx_id'],
            'User Type': users_df.iloc[tx['user_id']]['user_type'],
            'Amount': f"${tx['amount']:.2f}",
            'Token': tx['token_symbol'],
            'Fraud Probability': f"{tx['fraud_probability']:.3f}",
            'Risk Level': 'High' if tx['fraud_probability'] > 0.8 else 
                         'Medium' if tx['fraud_probability'] > 0.5 else 'Low'
        })
    
    st.dataframe(pd.DataFrame(suspicious_data), use_container_width=True)

# API testing
if api_online:
    st.header("🔗 API Testing")
    
    if st.button("Test API Endpoints"):
        endpoints = [
            ("Health Check", "/health"),
            ("Simulation Status", "/simulation/status"),
            ("Market Prices", "/market/prices")
        ]
        
        for name, endpoint in endpoints:
            try:
                response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
                if response.status_code == 200:
                    st.success(f"✅ {name}: OK")
                    if endpoint == "/market/prices":
                        data = response.json()
                        if data.get("prices"):
                            st.info(f"Available tokens: {list(data['prices'].keys())}")
                else:
                    st.warning(f"⚠️ {name}: Status {response.status_code}")
            except Exception as e:
                st.error(f"❌ {name}: {e}")

# System information
st.header("💻 System Information")
st.info(f"""
**Python Version:** {sys.version}
**Working Directory:** {os.getcwd()}
**Project Root:** {project_root}
**API Base URL:** {API_BASE}
**Plotly Available:** {PLOTLY_AVAILABLE}
**GNN Available:** {GNN_AVAILABLE}
**API Online:** {api_online}
""")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    🚀 Debug Dashboard | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)


