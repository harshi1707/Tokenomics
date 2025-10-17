import streamlit as st
import requests
import yaml
import os
import time
import json
import pandas as pd
import sys
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Try to import plotly, but don't fail if it's not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Charts will be disabled.")


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


cfg = load_config()
API_BASE = f"http://{cfg['api']['host']}:{cfg['api']['port']}"


st.set_page_config(page_title='Crypto Trading & DAO Simulation Dashboard', layout='wide')
st.title('🚀 AI-Powered Crypto Trading & DAO Simulation Platform')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Simulation Control", 
    "Trading Interface", 
    "DAO Governance", 
    "Market Analysis", 
    "User Management",
    "Legacy Features"
])

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

def make_api_request(method, endpoint, data=None, params=None):
    """Helper function to make API requests with error handling"""
    try:
        if method.upper() == 'GET':
            response = requests.get(f"{API_BASE}{endpoint}", params=params)
        elif method.upper() == 'POST':
            response = requests.post(f"{API_BASE}{endpoint}", json=data)
        else:
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


if page == "Simulation Control":
    st.header("🎮 Simulation Control Center")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Create New Simulation")
        with st.form("create_simulation"):
            sim_name = st.text_input("Simulation Name", value="demo_simulation")
            user_count = st.number_input("Number of Users", min_value=10, max_value=1000, value=100)
            duration_hours = st.number_input("Duration (hours)", min_value=1, max_value=168, value=24)
            trading_freq = st.slider("Trading Frequency", 0.01, 0.5, 0.1)
            auto_trading = st.checkbox("Auto Trading", value=True)
            auto_voting = st.checkbox("Auto Voting", value=True)
            
            if st.form_submit_button("Create & Start Simulation"):
                config_data = {
                    "name": sim_name,
                    "user_count": user_count,
                    "duration_hours": duration_hours,
                    "trading_frequency": trading_freq,
                    "auto_trading": auto_trading,
                    "auto_voting": auto_voting
                }
                result = make_api_request("POST", "/simulation/create", config_data)
                if result:
                    st.success(f"Simulation created: {result.get('message', 'Success')}")
    
    with col2:
        st.subheader("Simulation Status")
        status = make_api_request("GET", "/simulation/status")
        if status:
            if status.get("status") == "running":
                st.success("🟢 Simulation Running")
                st.metric("Runtime (hours)", f"{status.get('runtime_hours', 0):.2f}")
                st.metric("Active Users", status.get('users', 0))
                st.metric("Active Proposals", status.get('active_proposals', 0))
            elif status.get("status") == "stopped":
                st.warning("🔴 Simulation Stopped")
            else:
                st.info("⚪ No Simulation")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("Start Simulation"):
                result = make_api_request("POST", "/simulation/start")
                if result:
                    st.success("Simulation started!")
        
        with col2_2:
            if st.button("Stop Simulation"):
                result = make_api_request("POST", "/simulation/stop")
                if result:
                    st.success("Simulation stopped!")
    
    with col3:
        st.subheader("Quick Stats")
        stats = make_api_request("GET", "/simulation/stats")
        if stats and "simulation" in stats:
            sim_stats = stats["simulation"]
            if "stats" in sim_stats:
                st.metric("Total Trades", sim_stats["stats"].get("total_trades", 0))
                st.metric("Total Proposals", sim_stats["stats"].get("total_proposals", 0))
                st.metric("Total Votes", sim_stats["stats"].get("total_votes", 0))
                st.metric("Recommendations", sim_stats["stats"].get("recommendations_generated", 0))
    
    # Market Prices Display
    st.subheader("📈 Current Market Prices")
    prices_data = make_api_request("GET", "/market/prices")
    if prices_data:
        prices = prices_data.get("prices", {})
        sentiment = prices_data.get("sentiment", {})
        
        price_cols = st.columns(len(prices))
        for i, (token, price) in enumerate(prices.items()):
            with price_cols[i]:
                st.metric(
                    f"{token} Price", 
                    f"${price:.4f}",
                    delta=f"{sentiment.get(token, 'neutral')}"
                )

elif page == "Trading Interface":
    st.header("💹 Trading Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Execute Trade")
        with st.form("execute_trade"):
            user_id = st.number_input("User ID", min_value=1, value=1)
            token_symbol = st.selectbox("Token", ["BTC", "ETH", "DAOTOKEN", "GOVERNANCE"])
            trade_type = st.selectbox("Trade Type", ["buy", "sell"])
            amount = st.number_input("Amount", min_value=0.01, value=1.0)
            
            if st.form_submit_button("Execute Trade"):
                trade_data = {
                    "user_id": user_id,
                    "token_symbol": token_symbol,
                    "trade_type": trade_type,
                    "amount": amount
                }
                result = make_api_request("POST", "/trading/execute", trade_data)
                if result:
                    if result.get("success"):
                        st.success(f"Trade executed! Trade ID: {result.get('trade_id')}")
                        st.info(f"Price: ${result.get('price'):.4f}, Total: ${result.get('total_cost'):.2f}")
                    else:
                        st.error(f"Trade failed: {result.get('error')}")
    
    with col2:
        st.subheader("Get AI Recommendation")
        with st.form("get_recommendation"):
            rec_user_id = st.number_input("User ID for Recommendation", min_value=1, value=1)
            rec_token = st.selectbox("Token for Recommendation", ["BTC", "ETH", "DAOTOKEN", "GOVERNANCE"])
            
            if st.form_submit_button("Get Recommendation"):
                rec_data = make_api_request("GET", f"/trading/recommendation/{rec_user_id}/{rec_token}")
                if rec_data:
                    rec = rec_data.get("recommendation", {})
                    st.success("🤖 AI Recommendation Generated")
                    st.write(f"**Action:** {rec.get('action', 'N/A')}")
                    st.write(f"**Amount:** {rec.get('amount', 0):.4f}")
                    st.write(f"**Confidence:** {rec.get('confidence', 0):.2%}")
                    st.write(f"**Reasoning:** {rec.get('reasoning', 'N/A')}")
                    st.write(f"**Model:** {rec.get('model_source', 'N/A')}")

elif page == "Market Analysis":
    st.header("📊 Market Analysis Dashboard")
    
    # Market Overview
    st.subheader("Market Overview")
    market_data = make_api_request("GET", "/market/prices")
    if market_data:
        prices = market_data.get("prices", {})
        sentiment = market_data.get("sentiment", {})
        
        # Display prices in a simple table
        if prices:
            df = pd.DataFrame(list(prices.items()), columns=['Token', 'Price'])
            st.dataframe(df, use_container_width=True)
        
        # Display sentiment
        if sentiment:
            sentiment_df = pd.DataFrame(list(sentiment.items()), columns=['Token', 'Sentiment'])
            st.dataframe(sentiment_df, use_container_width=True)
    
    # Detailed Statistics
    st.subheader("Detailed Statistics")
    stats_data = make_api_request("GET", "/simulation/stats")
    if stats_data:
        if "market" in stats_data:
            market_stats = stats_data["market"]
            st.write("**Recent Trading Activity:**")
            recent_trades = market_stats.get("recent_trades", [])
            if recent_trades:
                trades_df = pd.DataFrame(recent_trades)
                st.dataframe(trades_df)
            else:
                st.info("No recent trading activity")
        
        if "governance" in stats_data:
            gov_stats = stats_data["governance"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Proposals", gov_stats.get("active_proposals", 0))
            with col2:
                st.metric("Active Voters", gov_stats.get("active_voters", 0))
            with col3:
                st.metric("Recent Proposals", gov_stats.get("recent_proposals", 0))

elif page == "User Management":
    st.header("👥 User Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New User")
        with st.form("create_user"):
            address = st.text_input("User Address", value="user_0001")
            initial_balance = st.number_input("Initial Balance", min_value=0.0, value=10000.0)
            
            if st.form_submit_button("Create User"):
                user_data = {
                    "address": address,
                    "initial_balance": initial_balance
                }
                result = make_api_request("POST", "/users/create", user_data)
                if result:
                    if result.get("success"):
                        st.success(f"User created! ID: {result.get('user_id')}")
                    else:
                        st.error(f"Failed to create user: {result.get('error')}")
    
    with col2:
        st.subheader("User Statistics")
        user_id = st.number_input("User ID", min_value=1, value=1)
        if st.button("Get User Stats"):
            stats = make_api_request("GET", f"/users/{user_id}/stats")
            if stats:
                st.write(f"**Address:** {stats.get('address', 'N/A')}")
                st.write(f"**Balance:** ${stats.get('balance', 0):.2f}")
                st.write(f"**Reputation:** {stats.get('reputation_score', 0):.1f}")
                st.write(f"**Total Trades:** {stats.get('total_trades', 0)}")
                st.write(f"**Votes Cast:** {stats.get('votes_cast', 0)}")
                st.write(f"**Proposals Created:** {stats.get('proposals_created', 0)}")
    
    # User Leaderboard
    st.subheader("🏆 User Leaderboard")
    leaderboard_data = make_api_request("GET", "/simulation/leaderboard")
    if leaderboard_data:
        leaderboard = leaderboard_data.get("leaderboard", [])
        if leaderboard:
            leaderboard_df = pd.DataFrame(leaderboard)
            st.dataframe(leaderboard_df, use_container_width=True)
        else:
            st.info("No leaderboard data available")

elif page == "Legacy Features":
    st.header("🔧 Legacy Features")

col1, col2 = st.columns(2)
with col1:
    st.subheader('Strategy Recommendation')
    if st.button('Get Recommendation'):
        try:
            resp = requests.post(f"{API_BASE}/recommend", json={})
            data = resp.json()
            st.json(data)
        except Exception as e:
            st.error(str(e))

with col2:
    st.subheader('Cluster Summary')
    try:
        resp = requests.get(f"{API_BASE}/cluster/summary")
        st.json(resp.json())
    except Exception as e:
        st.error(str(e))

st.subheader('Price Forecast')
horizon = st.slider('Horizon (days)', 1, 30, value=cfg['forecast']['horizon'])
if st.button('Get Forecast'):
    try:
        resp = requests.get(f"{API_BASE}/forecast", params={'horizon': horizon})
        st.json(resp.json())
    except Exception as e:
        st.error(str(e))

