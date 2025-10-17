from flask import Flask, jsonify, request, send_from_directory
import os
import yaml
from datetime import datetime
import json
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()

@app.route('/')
def root():
    return send_from_directory('.', 'login.html')

@app.route('/app')
def app_route():
    return send_from_directory('CryptoVerse-master/client/build', 'index.html')

@app.route('/dashboard')
def dashboard_route():
    return send_from_directory('.', 'dashboard.html')

@app.route('/network')
def network_route():
    return send_from_directory('.', 'suspicious_nodes.html')

@app.route('/simulator')
def simulator_route():
    return send_from_directory('.', 'token_launch_simulator.html')

@app.route('/login')
def login_route():
    return send_from_directory('.', 'login.html')


@app.route('/<path:path>')
def serve_static(path):
    # Try to serve from React build first
    try:
        return send_from_directory('CryptoVerse-master/client/build', path)
    except:
        # Fall back to serving from root directory
        try:
            return send_from_directory('..', path)
        except:
            return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/v1/coins/<coin_id>/forecast')
def get_coin_forecast(coin_id):
    """Mock forecast endpoint"""
    horizon = int(request.args.get('horizon', 7))
    model_type = request.args.get('model', 'lstm')

    # Generate mock forecast data
    current_price = 0.05
    predictions = []
    for i in range(horizon):
        day = i + 1
        predicted_price = current_price * (1 + np.random.normal(0, 0.1))
        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1
        confidence_level = 0.8 - (i * 0.05)

        predictions.append({
            'day': day,
            'predicted_price': predicted_price,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': max(0.5, confidence_level)
        })

    return jsonify({
        'coin_id': coin_id,
        'model_type': model_type,
        'horizon': horizon,
        'current_price': current_price,
        'forecast': {
            'dates': [f'2024-01-{i+1:02d}' for i in range(horizon)],
            'predictions': predictions
        },
        'metadata': {
            'last_updated': datetime.now().isoformat()
        }
    })

@app.route('/api/v1/coins/<coin_id>/fraud-risk')
def get_coin_fraud_risk(coin_id):
    """Mock fraud risk endpoint"""
    return jsonify({
        'coin_id': coin_id,
        'timestamp': datetime.now().isoformat(),
        'risk_score': 0.15,
        'risk_level': 'low',
        'risk_color': '#4caf50',
        'confidence': 0.87,
        'model_scores': {
            'gnn_model': 0.12,
            'transaction_analysis': 0.18,
            'pattern_recognition': 0.14
        },
        'cluster_info': {
            'total_clusters': 8,
            'clusters': [
                {'id': 0, 'size': 45, 'avg_risk': 0.08, 'description': 'Low risk trading cluster'},
                {'id': 1, 'size': 23, 'avg_risk': 0.25, 'description': 'Medium risk cluster'},
                {'id': 2, 'size': 12, 'avg_risk': 0.65, 'description': 'High risk suspicious cluster'}
            ],
            'high_risk_clusters': 1
        },
        'suspicious_patterns': [
            {
                'type': 'unusual_volume',
                'severity': 'medium',
                'description': 'Unusual trading volume detected',
                'confidence': 0.72
            }
        ],
        'analysis_method': 'GNN-based pattern analysis',
        'data_source': 'Transaction network data'
    })

@app.post('/api/v1/coins/<coin_id>/distribution')
def optimize_coin_distribution(coin_id):
    """Mock token distribution optimization"""
    body = request.get_json(force=True) or {}
    total_supply = body.get('total_supply', 1000000)
    target_market_cap = body.get('target_market_cap', 10000000)
    risk_preference = body.get('risk_preference', 'moderate')

    # Mock optimization result
    import uuid
    task_id = str(uuid.uuid4())

    # Store mock result
    mock_result = {
        'coin_id': coin_id,
        'optimal_allocation': {
            'community_airdrop': 0.35,
            'liquidity_mining': 0.25,
            'team_advisors': 0.20,
            'governance_rewards': 0.15,
            'marketing_partnerships': 0.05
        },
        'performance_metrics': {
            'stability_score': 0.87,
            'fairness_score': 0.92,
            'adoption_score': 0.78,
            'risk_adjusted_score': 0.83,
            'projected_market_cap': target_market_cap * 0.9,
            'market_cap_achievement': 0.9,
            'concentration_index': 0.23,
            'diversity_index': 0.76,
            'allocation_variance': 0.045,
            'allocation_entropy': 1.45
        },
        'simulation_results': {
            'ppo': {'total_reward': 1250.5, 'final_info': {'price_stability': 0.89}},
            'a3c': {'total_reward': 1180.3, 'final_info': {'price_stability': 0.91}},
            'sac': {'total_reward': 1320.7, 'final_info': {'price_stability': 0.87}}
        },
        'distribution_strategies': [
            'community_airdrop',
            'liquidity_mining',
            'team_advisors',
            'governance_rewards',
            'marketing_partnerships'
        ],
        'risk_preference': risk_preference,
        'timestamp': datetime.now().isoformat()
    }

    app.config[f'task_{task_id}'] = mock_result

    return jsonify({
        'task_id': task_id,
        'status': 'processing',
        'message': 'Token distribution optimization started',
        'coin_id': coin_id,
        'parameters': {
            'total_supply': total_supply,
            'target_market_cap': target_market_cap,
            'risk_preference': risk_preference
        }
    }), 202

@app.get('/api/v1/coins/<coin_id>/distribution/<task_id>')
def get_distribution_result(coin_id, task_id):
    """Get mock distribution result"""
    result = app.config.get(f'task_{task_id}')
    if result is None:
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'message': 'Optimization still in progress',
            'coin_id': coin_id
        }), 202

    return jsonify({
        'task_id': task_id,
        'status': 'completed',
        'result': result,
        'coin_id': coin_id
    })

@app.post('/api/v1/rl/optimize')
def rl_optimize():
    """Dynamic RL optimization endpoint based on simulation parameters"""
    body = request.get_json(force=True) or {}
    fraud_probabilities = body.get('fraud_probabilities', [0.1, 0.15, 0.08, 0.22, 0.12])
    model_type = body.get('model_type', 'ensemble')
    risk_preference = body.get('risk_preference', 'moderate')
    token_supply = body.get('token_supply', 1000000)
    target_market_cap = body.get('target_market_cap', 10000000)
    launch_duration = body.get('launch_duration', 30)
    market_condition = body.get('market_condition', 'sideways')
    expected_users = body.get('expected_users', 1000)

    # Calculate dynamic model performance based on parameters
    import numpy as np
    np.random.seed(int(token_supply) % 1000)  # Deterministic seed based on token supply

    # Base performance modifiers
    risk_modifier = {'conservative': 0.8, 'moderate': 1.0, 'aggressive': 1.2}[risk_preference]
    market_modifier = {'bull': 1.3, 'bear': 0.7, 'sideways': 1.0, 'volatile': 0.9}[market_condition]

    # Supply scaling factor (larger supply = more stable but lower reward)
    supply_factor = min(1.0, 1000000 / max(token_supply, 100000))
    market_cap_factor = min(1.0, target_market_cap / 10000000)

    # Calculate model performances
    base_performances = {
        'ppo': {'reward': 1200, 'stability': 0.85, 'convergence_time': 45, 'sample_efficiency': 0.75},
        'a3c': {'reward': 1150, 'stability': 0.88, 'convergence_time': 52, 'sample_efficiency': 0.80},
        'sac': {'reward': 1180, 'stability': 0.82, 'convergence_time': 38, 'sample_efficiency': 0.70},
        'dqn': {'reward': 1100, 'stability': 0.80, 'convergence_time': 41, 'sample_efficiency': 0.78}
    }

    # Apply modifiers
    model_performance = {}
    for model, perf in base_performances.items():
        modified_reward = perf['reward'] * risk_modifier * market_modifier * supply_factor * market_cap_factor
        modified_stability = min(0.95, perf['stability'] * risk_modifier * market_modifier)
        modified_conv_time = perf['convergence_time'] * (2 - supply_factor)  # Larger supply = longer training
        modified_efficiency = perf['sample_efficiency'] * market_modifier

        # Add some randomness
        reward_noise = np.random.normal(0, 50)
        stability_noise = np.random.normal(0, 0.02)

        model_performance[model] = {
            'reward': round(modified_reward + reward_noise, 1),
            'stability': round(min(0.95, max(0.7, modified_stability + stability_noise)), 3),
            'convergence_time': round(modified_conv_time + np.random.normal(0, 5), 1),
            'sample_efficiency': round(min(0.9, max(0.6, modified_efficiency + np.random.normal(0, 0.05))), 3)
        }

    # Determine best model based on weighted score
    def calculate_score(perf):
        return (perf['reward'] * 0.4 + perf['stability'] * 100 * 0.3 +
                (1 - perf['convergence_time']/60) * 100 * 0.2 + perf['sample_efficiency'] * 100 * 0.1)

    model_scores = {model: calculate_score(perf) for model, perf in model_performance.items()}
    best_model = max(model_scores, key=model_scores.get)

    # Calculate confidence based on score difference
    scores = list(model_scores.values())
    max_score = max(scores)
    second_max = sorted(scores)[-2]
    confidence = min(0.95, 0.7 + (max_score - second_max) / max_score * 0.25)

    # Generate dynamic allocations based on parameters
    base_allocations = np.array([0.35, 0.25, 0.20, 0.15, 0.05])

    # Adjust based on fraud probabilities and other factors
    risk_adjustment = np.array(fraud_probabilities) * 0.1
    supply_adjustment = np.array([0.05, 0.03, -0.02, -0.03, -0.03]) * (token_supply / 1000000)
    market_adjustment = {
        'bull': np.array([0.05, 0.03, -0.02, -0.03, -0.03]),
        'bear': np.array([-0.02, -0.03, 0.05, 0.03, -0.03]),
        'sideways': np.array([0, 0, 0, 0, 0]),
        'volatile': np.array([-0.03, 0.05, 0.02, -0.02, -0.02])
    }[market_condition]

    allocations = base_allocations + risk_adjustment[:5] + supply_adjustment + market_adjustment

    # Ensure allocations are positive and sum to 1
    allocations = np.maximum(allocations, 0.02)  # Minimum 2%
    allocations = allocations / allocations.sum()

    return jsonify({
        'best_model': best_model.upper(),
        'confidence': round(confidence, 3),
        'allocations': allocations.tolist(),
        'strategy': f'AI-Optimized Distribution using {best_model.upper()} Model',
        'model_performance': model_performance,
        'simulation_parameters': {
            'token_supply': token_supply,
            'target_market_cap': target_market_cap,
            'risk_preference': risk_preference,
            'market_condition': market_condition,
            'expected_users': expected_users
        }
    })

@app.post('/api/v1/fraud/analyze')
def fraud_analyze():
    """Mock fraud analysis endpoint"""
    body = request.get_json(force=True) or {}
    transaction_data = body.get('transaction_data', [0.1, 0.15, 0.08, 0.22, 0.12])
    analysis_type = body.get('analysis_type', 'comprehensive')

    # Mock fraud analysis results
    suspicious_nodes = sum(1 for x in transaction_data if x > 0.15)
    high_risk_clusters = sum(1 for x in transaction_data if x > 0.2)

    return jsonify({
        'risk_level': 'low' if suspicious_nodes < 2 else 'medium',
        'risk_score': sum(transaction_data) / len(transaction_data),
        'suspicious_nodes': suspicious_nodes,
        'high_risk_clusters': high_risk_clusters,
        'confidence': 0.94,
        'analysis_type': analysis_type,
        'clusters': [
            {'id': 0, 'size': 45, 'avg_risk': 0.08, 'description': 'Low risk trading cluster'},
            {'id': 1, 'size': 23, 'avg_risk': 0.25, 'description': 'Medium risk cluster'},
            {'id': 2, 'size': 12, 'avg_risk': 0.65, 'description': 'High risk suspicious cluster'}
        ]
    })

@app.post('/api/v1/forecast/predict')
def forecast_predict():
    """Mock price forecasting endpoint"""
    body = request.get_json(force=True) or {}
    symbol = body.get('symbol', 'BTC')
    horizon = body.get('horizon', 7)
    model_type = body.get('model_type', 'lstm')

    # Mock price predictions
    current_price = 0.05
    predictions = []
    for i in range(horizon):
        day = i + 1
        # Generate realistic price movement
        import numpy as np
        change = np.random.normal(0, 0.02)  # Small random changes
        predicted_price = current_price * (1 + change)
        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1
        confidence_level = max(0.5, 0.95 - (i * 0.05))  # Decreasing confidence over time

        predictions.append({
            'day': day,
            'predicted_price': predicted_price,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        })

    return jsonify({
        'symbol': symbol,
        'current_price': current_price,
        'predictions': predictions,
        'model_type': model_type,
        'horizon': horizon,
        'confidence_interval': 0.95
    })

@app.post('/api/v1/fraud/suspicious-nodes')
def get_suspicious_nodes():
    """Get top suspicious nodes with interactive graph data"""
    body = request.get_json(force=True) or {}
    limit = body.get('limit', 10)
    include_graph = body.get('include_graph', True)

    # Mock suspicious nodes data (from GNN model results)
    suspicious_nodes = [
        {'id': 1812, 'fraud_prob': 0.845212, 'connections': 45, 'risk_level': 'high', 'cluster': 0},
        {'id': 1121, 'fraud_prob': 0.823431, 'connections': 32, 'risk_level': 'high', 'cluster': 1},
        {'id': 1857, 'fraud_prob': 0.810875, 'connections': 28, 'risk_level': 'high', 'cluster': 2},
        {'id': 245, 'fraud_prob': 0.782664, 'connections': 67, 'risk_level': 'high', 'cluster': 0},
        {'id': 711, 'fraud_prob': 0.777921, 'connections': 23, 'risk_level': 'high', 'cluster': 1},
        {'id': 380, 'fraud_prob': 0.747604, 'connections': 41, 'risk_level': 'medium', 'cluster': 2},
        {'id': 1218, 'fraud_prob': 0.731719, 'connections': 19, 'risk_level': 'medium', 'cluster': 0},
        {'id': 1938, 'fraud_prob': 0.723713, 'connections': 55, 'risk_level': 'medium', 'cluster': 1},
        {'id': 886, 'fraud_prob': 0.721449, 'connections': 33, 'risk_level': 'medium', 'cluster': 2},
        {'id': 474, 'fraud_prob': 0.719876, 'connections': 27, 'risk_level': 'medium', 'cluster': 0}
    ]

    # Limit results
    suspicious_nodes = suspicious_nodes[:limit]

    # Prepare graph data for interactive visualization
    graph_data = None
    if include_graph:
        # Create network graph data
        nodes = []
        links = []

        # Add suspicious nodes
        for node in suspicious_nodes:
            nodes.append({
                'id': str(node['id']),
                'name': f"Node {node['id']}",
                'fraud_prob': node['fraud_prob'],
                'connections': node['connections'],
                'risk_level': node['risk_level'],
                'cluster': node['cluster'],
                'group': node['cluster'] + 1,  # For visualization grouping
                'size': max(5, node['connections'] / 2),  # Node size based on connections
                'color': '#ff4444' if node['risk_level'] == 'high' else '#ff8800' if node['risk_level'] == 'medium' else '#44aa44'
            })

        # Add some connected nodes (neighbors)
        neighbor_nodes = [
            {'id': 505, 'connections': 12, 'cluster': 0},
            {'id': 744, 'connections': 8, 'cluster': 0},
            {'id': 993, 'connections': 15, 'cluster': 1},
            {'id': 1047, 'connections': 9, 'cluster': 1},
            {'id': 1108, 'connections': 11, 'cluster': 2},
            {'id': 1328, 'connections': 7, 'cluster': 2},
            {'id': 1624, 'connections': 13, 'cluster': 0},
            {'id': 1663, 'connections': 6, 'cluster': 1},
            {'id': 1745, 'connections': 10, 'cluster': 2},
            {'id': 1834, 'connections': 14, 'cluster': 0}
        ]

        for neighbor in neighbor_nodes:
            nodes.append({
                'id': str(neighbor['id']),
                'name': f"Node {neighbor['id']}",
                'fraud_prob': 0.1 + (neighbor['connections'] * 0.01),  # Lower fraud prob for neighbors
                'connections': neighbor['connections'],
                'risk_level': 'low',
                'cluster': neighbor['cluster'],
                'group': neighbor['cluster'] + 1,
                'size': max(3, neighbor['connections'] / 3),
                'color': '#666666'
            })

        # Create links between suspicious nodes and their neighbors
        for i, suspicious in enumerate(suspicious_nodes):
            cluster_neighbors = [n for n in neighbor_nodes if n['cluster'] == suspicious['cluster']]
            for neighbor in cluster_neighbors[:3]:  # Connect to 3 neighbors max
                links.append({
                    'source': str(suspicious['id']),
                    'target': str(neighbor['id']),
                    'value': 1,
                    'strength': 0.5
                })

        graph_data = {
            'nodes': nodes,
            'links': links
        }

    return jsonify({
        'suspicious_nodes': suspicious_nodes,
        'total_count': len(suspicious_nodes),
        'graph_data': graph_data,
        'metadata': {
            'model': 'GNN_Fraud_Detector',
            'threshold': 0.7,
            'last_updated': '2025-10-17T13:30:00Z',
            'data_source': 'elliptic_dataset'
        }
    })

@app.route('/simulation/create', methods=['POST'])
def create_simulation():
    """Create and start a new simulation"""
    try:
        body = request.get_json(force=True) or {}
    except:
        body = {}

    # Mock simulation creation
    simulation_id = f"sim_{int(datetime.now().timestamp())}"

    return jsonify({
        'success': True,
        'simulation_id': simulation_id,
        'message': 'Simulation created and started successfully',
        'parameters': body
    })

@app.route('/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop the current simulation"""
    return jsonify({
        'success': True,
        'message': 'Simulation stopped successfully'
    })

@app.route('/simulation/status')
def get_simulation_status():
    """Get current simulation status"""
    # Mock status - in real implementation this would check actual simulation state
    return jsonify({
        'status': 'stopped',  # or 'running'
        'runtime_hours': 0.0,
        'users': 0,
        'active_proposals': 0
    })

@app.post('/api/v1/tokenomics/distribution-strategies')
def get_dynamic_token_distribution_strategies():
    """Generate dynamic token distribution strategies based on real-time simulation data"""
    body = request.get_json(force=True) or {}
    simulation_context = body.get('simulation_context', {})

    # Get current simulation status to inform strategy selection
    sim_status = None
    try:
        sim_status = make_api_request("GET", "/simulation/status")
    except:
        pass

    if sim_status and sim_status.get('status') == 'running':
        # Use real simulation data
        runtime_hours = sim_status.get('runtime_hours', 0)
        active_users = sim_status.get('users', 100)
        market_sentiment = sim_status.get('market_sentiment', {'BTC': 'neutral'})
    else:
        # Use defaults for non-running simulation
        runtime_hours = 0
        active_users = 100
        market_sentiment = {'BTC': 'neutral'}

    # Get fraud analysis for risk assessment
    try:
        fraud_data = make_api_request("POST", "/api/v1/fraud/suspicious-nodes", {
            'limit': 10,
            'include_graph': False
        })
        high_risk_count = len([n for n in fraud_data.get('suspicious_nodes', [])
                              if n.get('risk_level') == 'high']) if fraud_data else 0
    except:
        high_risk_count = 2

    # Get RL optimization results
    try:
        rl_results = make_api_request("POST", "/api/v1/rl/optimize", {
            'fraud_probabilities': [0.1, 0.15, 0.08, 0.22, 0.12],
            'model_type': 'ensemble',
            'risk_preference': 'moderate'
        })
        rl_confidence = rl_results.get('confidence', 0.8) if rl_results else 0.8
    except:
        rl_confidence = 0.8

    # Dynamic strategy selection based on simulation context
    strategies = []

    # Base strategies with dynamic weighting
    base_strategies = [
        {
            'name': 'ICO/IEO/IDO (Token Sale)',
            'description': 'Tokens sold to investors via exchange/platform',
            'current_issues': 'Whale domination, scams, pump & dump',
            'ai_improvement': 'Reinforcement Learning (PPO/DQN) optimizes sale pricing & allocation fairness',
            'allocation_percentage': 25 + (high_risk_count * 2),  # Increase with fraud risk
            'confidence_score': rl_confidence,
            'adaptability': 'high'
        },
        {
            'name': 'Airdrops',
            'description': 'Free tokens given to wallets/users',
            'current_issues': 'Sybil attacks, token dumping',
            'ai_improvement': 'ML fraud detection + behavior scoring ensures only genuine users',
            'allocation_percentage': 15 + (active_users // 20),  # Scale with user count
            'confidence_score': 0.85,
            'adaptability': 'medium'
        },
        {
            'name': 'Vesting Schedules',
            'description': 'Gradual token release to team/investors',
            'current_issues': 'Unlock events cause price crashes',
            'ai_improvement': 'RL adaptive vesting schedules based on market conditions',
            'allocation_percentage': 20 - (runtime_hours // 10),  # Decrease over time
            'confidence_score': 0.78,
            'adaptability': 'high'
        },
        {
            'name': 'Liquidity Mining / Staking Rewards',
            'description': 'Tokens given for providing liquidity',
            'current_issues': 'Centralization (whales dominate pools)',
            'ai_improvement': 'GNN detects concentration, ensures fair reward scaling',
            'allocation_percentage': 18 + (active_users // 50),
            'confidence_score': 0.82,
            'adaptability': 'medium'
        },
        {
            'name': 'Governance Incentives',
            'description': 'Tokens given for voting/participation',
            'current_issues': 'Sybil wallets, vote buying',
            'ai_improvement': 'Reputation-weighted + GNN scoring ensures real user participation',
            'allocation_percentage': 12 + (high_risk_count * 1.5),
            'confidence_score': 0.75,
            'adaptability': 'high'
        },
        {
            'name': 'Buyback & Burn',
            'description': 'Project buys tokens and burns them',
            'current_issues': 'Market unpredictability',
            'ai_improvement': 'Predictive ML optimizes timing for buybacks to stabilize price',
            'allocation_percentage': 5 + (runtime_hours // 20),
            'confidence_score': 0.70,
            'adaptability': 'low'
        },
        {
            'name': 'Quadratic Distribution',
            'description': 'Rewards grow with square root of contribution',
            'current_issues': 'Still gameable with multiple wallets',
            'ai_improvement': 'GNN + Sybil detection prevents wallet splitting',
            'allocation_percentage': 8,
            'confidence_score': 0.88,
            'adaptability': 'medium'
        },
        {
            'name': 'AI-Powered Hybrid Distribution ⭐',
            'description': 'Mix of multiple strategies tuned by AI',
            'current_issues': 'No such system exists yet',
            'ai_improvement': 'Our novelty: adaptive blending of strategies',
            'allocation_percentage': 30 + (rl_confidence * 10),
            'confidence_score': rl_confidence,
            'adaptability': 'maximum'
        }
    ]

    # Adjust allocations based on simulation context
    total_allocation = sum(s['allocation_percentage'] for s in base_strategies)

    # Normalize to 100%
    for strategy in base_strategies:
        strategy['allocation_percentage'] = round((strategy['allocation_percentage'] / total_allocation) * 100, 1)

    # Sort by confidence and adaptability
    strategies = sorted(base_strategies, key=lambda x: (x['confidence_score'], x['adaptability']), reverse=True)

    # Add market context insights
    market_context = {
        'simulation_runtime': runtime_hours,
        'active_users': active_users,
        'high_risk_nodes': high_risk_count,
        'market_sentiment': market_sentiment,
        'rl_model_confidence': rl_confidence,
        'recommended_primary_strategy': strategies[0]['name'],
        'risk_adjustment_factor': high_risk_count / max(active_users, 1),
        'adaptability_score': sum(s['confidence_score'] for s in strategies) / len(strategies)
    }

    # Get detailed model performance metrics
    model_performance = {}
    try:
        rl_response = make_api_request("POST", "/api/v1/rl/optimize", {
            'fraud_probabilities': [0.1, 0.15, 0.08, 0.22, 0.12],
            'model_type': 'ensemble',
            'risk_preference': 'moderate'
        })
        if rl_response and 'model_performance' in rl_response:
            model_performance = rl_response['model_performance']
    except:
        # Fallback model performance data
        model_performance = {
            'ppo': {'reward': 1250.5, 'stability': 0.89, 'convergence_time': 45.2, 'sample_efficiency': 0.76},
            'a3c': {'reward': 1180.3, 'stability': 0.91, 'convergence_time': 52.1, 'sample_efficiency': 0.82},
            'sac': {'reward': 1320.7, 'stability': 0.87, 'convergence_time': 38.9, 'sample_efficiency': 0.71},
            'dqn': {'reward': 1150.2, 'stability': 0.85, 'convergence_time': 41.7, 'sample_efficiency': 0.79}
        }

    return jsonify({
        'strategies': strategies,
        'market_context': market_context,
        'simulation_status': {
            'is_running': sim_status is not None and sim_status.get('status') == 'running',
            'runtime_hours': runtime_hours,
            'active_users': active_users
        },
        'ai_insights': {
            'fraud_risk_level': 'high' if high_risk_count > 3 else 'medium' if high_risk_count > 1 else 'low',
            'recommended_allocation_adjustment': f"{'Increase' if high_risk_count > 2 else 'Maintain'} security-focused strategies",
            'market_adaptability': 'high' if rl_confidence > 0.8 else 'medium',
            'next_best_action': 'Monitor fraud patterns' if high_risk_count > 2 else 'Scale user acquisition'
        },
        'model_performance': model_performance,
        'timestamp': '2025-10-17T13:59:50Z',
        'model_versions': {
            'rl_model': 'PPO-v2.1',
            'gnn_model': 'GraphSAGE-v1.3',
            'forecasting_model': 'LSTM-v2.0'
        }
    })

@app.post('/recommendations/token-distribution/<int:user_id>')
def get_token_distribution_recommendation(user_id):
    """Mock token distribution recommendation"""
    body = request.get_json(force=True) or {}
    suspicious_probs = body.get('suspicious_probs', [0.1, 0.15, 0.08, 0.22, 0.12])

    # Mock recommendation
    return jsonify({
        'user_id': user_id,
        'recommendation': {
            'type': 'token_distribution',
            'model_source': 'rl_ppo',
            'confidence': 0.89,
            'reasoning': 'Based on PPO reinforcement learning analysis of fraud probabilities and market conditions.',
            'strategy': 'AI-Optimized Distribution',
            'secondary_strategy': 'Risk-Adjusted Allocation',
            'allocations': [0.35, 0.25, 0.20, 0.15, 0.05],
            'fairness_score': 0.92,
            'stability_score': 0.87,
            'risk_score': 0.13,
            'model_confidence': 0.89,
            'market_conditions': {
                'high_risk_nodes': 2,
                'total_nodes': 5,
                'avg_risk': 0.134,
                'risk_variance': 0.003
            }
        }
    })

if __name__ == '__main__':
    port = 3000  # Run on port 3000 as requested
    print(f"Starting simplified API server on port {port}")
    app.run(host='127.0.0.1', port=port, debug=True)