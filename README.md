# Tokenomics AI Platform

A comprehensive AI-powered platform for token distribution optimization, featuring Reinforcement Learning (RL), Graph Neural Networks (GNN), and time series forecasting for intelligent tokenomics decisions.

##  Running the Platform Locally

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd tokenomics-platform

# Run automated setup (recommended)
python setup.py

# Or manual setup:
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Load Pre-trained AI Models
The platform includes pre-trained RL models. To ensure optimal performance:

```bash
# Activate virtual environment (if not already activated)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# The pre-trained models are located in:
# rl/saved_models/
# - a3c_crypto_opt.zip
# - dqn_crypto_opt.zip
# - ppo_crypto_opt.zip
# - sac_crypto_opt.zip

# Models are automatically loaded when the API server starts
# No additional setup required - they're ready to use!
```

### Step 3: Start the Platform
```bash
# Start the Flask API server
python api/simple_app.py

# Server will start on: http://localhost:3000
```

### Step 4: Access the Platform
1. **Open your browser** and navigate to: `http://localhost:3000`
2. **Login Page**: Enter any email/password or click "Continue as Guest"
3. **Main Dashboard**: Access all features from the navigation

### Step 5: Test AI Features
- **Token Launch Simulator**: Configure parameters and run AI simulations
- **Model Performance**: Each simulation dynamically selects the best RL model (PPO/A3C/SAC/DQN)
- **Fraud Detection**: Analyze suspicious network activity
- **Price Forecasting**: View LSTM predictions

##  Project Structure
```
tokenomics-platform/
├── api/                          # Flask API server
│   ├── simple_app.py            # Main API with all endpoints
│   ├── rl_optimization_service.py
│   ├── forecasting_service.py
│   └── coingecko_service.py
├── rl/                           # Reinforcement Learning
│   └── saved_models/            # Pre-trained RL models
│       ├── a3c_crypto_opt.zip
│       ├── dqn_crypto_opt.zip
│       ├── ppo_crypto_opt.zip
│       └── sac_crypto_opt.zip
├── dashboard/                    # Streamlit dashboards
├── data/                         # Datasets (Elliptic for GNN)
├── configs/                      # Configuration files
├── login.html                    # Login page (entry point)
├── index.html                    # Main dashboard
├── token_launch_simulator.html   # Token launch simulator
├── dashboard.html               # Analytics dashboard
├── suspicious_nodes.html        # Fraud detection interface
├── simple_website.html          # Simple website
├── requirements.txt             # Python dependencies
├── setup.py                     # Automated setup script
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```
