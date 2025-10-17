import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Import RL components
from rl.env import TokenDistributionEnv
from rl.trainers import train_ppo, train_dqn, train_sac, train_a3c, get_ensemble_prediction
from stable_baselines3 import PPO, DQN, SAC, A2C

logger = logging.getLogger(__name__)

# Distribution strategies
DISTRIBUTION_STRATEGIES = [
    'equal_distribution',
    'vesting_schedule',
    'airdrop_allocation',
    'liquidity_mining',
    'governance_rewards',
    'buyback_mechanism'
]

class TokenDistributionOptimizationEnv(gym.Env):
    """Enhanced environment for token distribution optimization with real parameters"""

    def __init__(self, total_supply: float, target_market_cap: float, risk_preference: str = 'moderate'):
        super(TokenDistributionOptimizationEnv, self).__init__()

        self.total_supply = total_supply
        self.target_market_cap = target_market_cap
        self.risk_preference = risk_preference

        # Risk preference mapping
        self.risk_multipliers = {
            'conservative': {'stability': 1.5, 'fairness': 1.2, 'adoption': 0.8},
            'moderate': {'stability': 1.0, 'fairness': 1.0, 'adoption': 1.0},
            'aggressive': {'stability': 0.7, 'fairness': 0.8, 'adoption': 1.3}
        }

        # Action space: allocation across 6 strategies
        self.action_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        # Observation space: market state + allocation state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.allocation = np.ones(6, dtype=np.float32) / 6
        self.state = np.zeros(12, dtype=np.float32)
        self.current_step = 0
        self.max_steps = 365  # 1 year simulation

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.allocation = np.ones(6, dtype=np.float32) / 6
        self.state = np.random.normal(0.0, 0.1, size=self.state.shape).astype(np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Normalize action to allocation
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones(6) / 6
        else:
            action = action / np.sum(action)

        self.allocation = action.astype(np.float32)

        # Simulate market response based on allocation and parameters
        price_stability, fairness, adoption = self._simulate_market_response()

        # Calculate reward based on risk preference
        risk_mult = self.risk_multipliers[self.risk_preference]
        reward = (
            risk_mult['stability'] * price_stability +
            risk_mult['fairness'] * fairness +
            risk_mult['adoption'] * adoption
        )

        # Update state
        self.state = self._next_state()

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            'price_stability': float(price_stability),
            'fairness': float(fairness),
            'adoption': float(adoption),
            'allocation': self.allocation.tolist(),
            'market_cap_projection': float(self._calculate_market_cap_projection())
        }

        return self.state, reward, terminated, truncated, info

    def _simulate_market_response(self) -> Tuple[float, float, float]:
        """Simulate market response to allocation strategy"""
        # Strategy effectiveness weights
        strategy_weights = {
            'equal_distribution': [0.8, 0.9, 0.6],  # stability, fairness, adoption
            'vesting_schedule': [0.9, 0.7, 0.8],
            'airdrop_allocation': [0.6, 0.8, 0.9],
            'liquidity_mining': [0.7, 0.6, 0.9],
            'governance_rewards': [0.8, 0.8, 0.7],
            'buyback_mechanism': [0.9, 0.7, 0.6]
        }

        price_stability = 0
        fairness = 0
        adoption = 0

        for i, strategy in enumerate(DISTRIBUTION_STRATEGIES):
            weight = self.allocation[i]
            weights = strategy_weights[strategy]
            price_stability += weight * weights[0]
            fairness += weight * weights[1]
            adoption += weight * weights[2]

        # Add some noise and time-based decay
        noise = np.random.normal(0, 0.05)
        time_decay = max(0.5, 1.0 - self.current_step / self.max_steps)

        price_stability = np.clip(price_stability + noise, 0, 1) * time_decay
        fairness = np.clip(fairness + noise, 0, 1) * time_decay
        adoption = np.clip(adoption + noise, 0, 1) * time_decay

        return price_stability, fairness, adoption

    def _calculate_market_cap_projection(self) -> float:
        """Project market cap based on current allocation and market conditions"""
        base_price = self.target_market_cap / self.total_supply
        stability_factor = self._simulate_market_response()[0]
        adoption_factor = self._simulate_market_response()[2]

        # Price adjustment based on strategy effectiveness
        price_multiplier = 0.8 + 0.4 * (stability_factor + adoption_factor) / 2
        projected_price = base_price * price_multiplier

        return projected_price * self.total_supply

    def _next_state(self) -> np.ndarray:
        """Update market state"""
        noise = np.random.normal(0.0, 0.05, size=self.state.shape)
        influence = np.concatenate([self.allocation, np.array([self.current_step / self.max_steps])])
        influence = np.pad(influence, (0, max(0, self.state.shape[0] - len(influence))), mode='constant')

        return (0.9 * self.state + 0.1 * influence + noise).astype(np.float32)


class RLOptimizationService:
    """Service for RL-based token distribution optimization"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active_simulations = {}

    def optimize_token_distribution(
        self,
        coin_id: str,
        total_supply: float,
        target_market_cap: float,
        risk_preference: str = 'moderate',
        simulation_steps: int = 1000,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run RL optimization for token distribution"""

        logger.info(f"Starting RL optimization for {coin_id}")

        # Create environment
        env = TokenDistributionOptimizationEnv(
            total_supply=total_supply,
            target_market_cap=target_market_cap,
            risk_preference=risk_preference
        )

        # Train models
        models = {}
        model_types = ['ppo', 'sac', 'dqn']

        for model_type in model_types:
            try:
                if progress_callback:
                    progress_callback(f"Training {model_type.upper()} model", 0)

                if model_type == 'ppo':
                    model = train_ppo(total_timesteps=simulation_steps)
                elif model_type == 'sac':
                    model = train_sac(total_timesteps=simulation_steps)
                elif model_type == 'dqn':
                    model = train_dqn(total_timesteps=simulation_steps)

                models[model_type] = model

                if progress_callback:
                    progress_callback(f"Completed {model_type.upper()} training", 100)

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue

        # Get ensemble prediction
        ensemble_result = get_ensemble_prediction(np.ones(6) / 6)  # Dummy probs for ensemble

        # Run simulations with each model
        simulation_results = {}
        for model_type, model in models.items():
            try:
                result = self._run_model_simulation(env, model, model_type)
                simulation_results[model_type] = result
            except Exception as e:
                logger.error(f"Simulation failed for {model_type}: {e}")
                continue

        # Calculate final recommendations
        final_allocation = self._calculate_optimal_allocation(simulation_results)

        # Generate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            final_allocation, total_supply, target_market_cap, risk_preference
        )

        return {
            'coin_id': coin_id,
            'optimal_allocation': final_allocation,
            'performance_metrics': performance_metrics,
            'simulation_results': simulation_results,
            'distribution_strategies': DISTRIBUTION_STRATEGIES,
            'risk_preference': risk_preference,
            'timestamp': datetime.now().isoformat()
        }

    def _run_model_simulation(self, env, model, model_type: str) -> Dict[str, Any]:
        """Run simulation with a specific model"""
        obs, _ = env.reset()
        total_reward = 0
        rewards = []
        allocations = []

        for _ in range(50):  # Shorter simulation for evaluation
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            rewards.append(float(reward))
            allocations.append(env.allocation.tolist())

            if terminated or truncated:
                break

        return {
            'total_reward': float(total_reward),
            'average_reward': float(np.mean(rewards)),
            'final_allocation': env.allocation.tolist(),
            'allocation_history': allocations,
            'reward_history': rewards,
            'final_info': info
        }

    def _calculate_optimal_allocation(self, simulation_results: Dict) -> Dict[str, float]:
        """Calculate optimal allocation from simulation results"""
        if not simulation_results:
            # Fallback to equal distribution
            allocation = {strategy: 1.0 / len(DISTRIBUTION_STRATEGIES)
                         for strategy in DISTRIBUTION_STRATEGIES}
            return allocation

        # Weight by total reward
        total_rewards = {model: result['total_reward'] for model, result in simulation_results.items()}
        max_reward = max(total_rewards.values())

        if max_reward == 0:
            # Equal weights if all rewards are zero
            weights = {model: 1.0 / len(simulation_results) for model in simulation_results}
        else:
            weights = {model: reward / max_reward for model, reward in total_rewards.items()}

        # Weighted average allocation
        final_allocation = np.zeros(6)
        for model, weight in weights.items():
            allocation = np.array(simulation_results[model]['final_allocation'])
            final_allocation += weight * allocation

        # Normalize
        final_allocation = final_allocation / np.sum(final_allocation)

        return {DISTRIBUTION_STRATEGIES[i]: float(final_allocation[i]) for i in range(6)}

    def _calculate_performance_metrics(
        self,
        allocation: Dict[str, float],
        total_supply: float,
        target_market_cap: float,
        risk_preference: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""

        allocation_array = np.array([allocation[strategy] for strategy in DISTRIBUTION_STRATEGIES])

        # Strategy effectiveness (based on predefined weights)
        strategy_effectiveness = {
            'equal_distribution': [0.8, 0.9, 0.6],
            'vesting_schedule': [0.9, 0.7, 0.8],
            'airdrop_allocation': [0.6, 0.8, 0.9],
            'liquidity_mining': [0.7, 0.6, 0.9],
            'governance_rewards': [0.8, 0.8, 0.7],
            'buyback_mechanism': [0.9, 0.7, 0.6]
        }

        stability_score = sum(allocation[strategy] * strategy_effectiveness[strategy][0]
                              for strategy in DISTRIBUTION_STRATEGIES)
        fairness_score = sum(allocation[strategy] * strategy_effectiveness[strategy][1]
                            for strategy in DISTRIBUTION_STRATEGIES)
        adoption_score = sum(allocation[strategy] * strategy_effectiveness[strategy][2]
                             for strategy in DISTRIBUTION_STRATEGIES)

        # Risk-adjusted scores
        risk_mult = {
            'conservative': {'stability': 1.5, 'fairness': 1.2, 'adoption': 0.8},
            'moderate': {'stability': 1.0, 'fairness': 1.0, 'adoption': 1.0},
            'aggressive': {'stability': 0.7, 'fairness': 0.8, 'adoption': 1.3}
        }[risk_preference]

        risk_adjusted_score = (
            risk_mult['stability'] * stability_score +
            risk_mult['fairness'] * fairness_score +
            risk_mult['adoption'] * adoption_score
        ) / sum(risk_mult.values())

        # Market cap projection
        base_price = target_market_cap / total_supply
        market_multiplier = 0.8 + 0.4 * (stability_score + adoption_score) / 2
        projected_market_cap = base_price * market_multiplier * total_supply

        # Token distribution analysis
        concentration_index = max(allocation.values())  # Higher = more concentrated
        diversity_index = len([v for v in allocation.values() if v > 0.05]) / len(DISTRIBUTION_STRATEGIES)

        return {
            'stability_score': float(stability_score),
            'fairness_score': float(fairness_score),
            'adoption_score': float(adoption_score),
            'risk_adjusted_score': float(risk_adjusted_score),
            'projected_market_cap': float(projected_market_cap),
            'market_cap_achievement': float(projected_market_cap / target_market_cap),
            'concentration_index': float(concentration_index),
            'diversity_index': float(diversity_index),
            'allocation_variance': float(np.var(allocation_array)),
            'allocation_entropy': float(-sum(p * np.log(p + 1e-10) for p in allocation_array if p > 0))
        }

    async def optimize_async(
        self,
        coin_id: str,
        total_supply: float,
        target_market_cap: float,
        risk_preference: str = 'moderate',
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Async version of optimization"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.optimize_token_distribution,
            coin_id, total_supply, target_market_cap, risk_preference, 1000, progress_callback
        )


# Global service instance
rl_service = RLOptimizationService()


def optimize_token_distribution(
    coin_id: str,
    total_supply: float,
    target_market_cap: float,
    risk_preference: str = 'moderate'
) -> Dict[str, Any]:
    """Main function to optimize token distribution using RL"""
    return rl_service.optimize_token_distribution(
        coin_id=coin_id,
        total_supply=total_supply,
        target_market_cap=target_market_cap,
        risk_preference=risk_preference
    )


async def optimize_token_distribution_async(
    coin_id: str,
    total_supply: float,
    target_market_cap: float,
    risk_preference: str = 'moderate',
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Async version for API use"""
    return await rl_service.optimize_async(
        coin_id=coin_id,
        total_supply=total_supply,
        target_market_cap=target_market_cap,
        risk_preference=risk_preference,
        progress_callback=progress_callback
    )