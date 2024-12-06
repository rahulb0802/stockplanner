import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, historical_data, initial_value, initial_allocations, phase_goals, benchmark_returns=None, is_training=True):
        super(PortfolioEnv, self).__init__()

        self.historical_data = historical_data
        self.phase_goals = phase_goals  # phase_goals: {year: goal_value}
        self.benchmark_returns = benchmark_returns
        self.current_year = 0
        self.portfolio_value = initial_value
        self.allocations = np.array(initial_allocations, dtype=np.float32)

        if not np.isclose(np.sum(self.allocations), 1):
            raise ValueError("Initial allocations must sum to 1.")

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(15,), dtype=np.float32  # portfolio value, allocations, inflation rates
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(12,), dtype=np.float32  # Adjust allocations for 12 assets
        )

        self.history = []
        self.is_training = is_training

    def reset(self, seed=None, options=None):
        self.current_year = 0
        self.portfolio_value = 100000
        state = self._get_state()
        return state, {}

    def _get_state(self):
        inflation_us = self.historical_data.iloc[self.current_year]['US Inflation']
        inflation_nigeria = self.historical_data.iloc[self.current_year]['Nigeria Inflation']

        state = np.concatenate([np.array([self.portfolio_value]), self.allocations, np.array([inflation_us, inflation_nigeria])])

        return state

    def step(self, action):
        # Ensure the action is clipped within bounds and adjust allocations
        action = np.clip(action, -0.5, 0.5)
        self.allocations = np.clip(self.allocations + action, 0, 1)
        self.allocations /= np.sum(self.allocations)

        asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)
        portfolio_return = np.dot(self.allocations, asset_returns)

        # Update the portfolio value based on the return
        self.portfolio_value *= (1 + portfolio_return)
        self.history.append({
            "year": self.historical_data.index[self.current_year],
            "portfolio_value": self.portfolio_value,
            "allocations": self.allocations.tolist(),
            "reward": self._calculate_reward()  # Get reward based on phase goals and other factors
        })

        reward = self._calculate_reward()
        self.current_year += 1
        done = self.current_year >= len(self.historical_data) - 1
        truncated = False
        return self._get_state(), reward, done, truncated, {}

    def _calculate_reward(self):
        """
        Enhanced reward function to meet phase goals, encourage consistent growth, 
        reduce volatility, and ensure the portfolio aligns with the final target.
        """
        current_year = self.historical_data.index[self.current_year]
        inflation_us = self.historical_data.iloc[self.current_year]['US Inflation']
        inflation_nigeria = self.historical_data.iloc[self.current_year]['Nigeria Inflation']
        
        # Initialize reward
        reward = 0
        scaling_factor = self.portfolio_value / 100000  # Scale penalties by portfolio size

        if current_year in [2008, 2009, 2022]:  # Handle known periods of crisis
            reward -= 5000  # Penalize significantly for being unable to weather the storm
            if self.portfolio_value > 100000:  # Reward those who don't lose too much
                reward += 2000

        # 1. Phase Goal Reward/Penalty
        if current_year in self.phase_goals:
            phase_goal = self.phase_goals[current_year]
            if self.portfolio_value >= phase_goal:
                reward += 1000 * scaling_factor  # Reward for meeting the phase goal
            else:
                reward -= (phase_goal - self.portfolio_value) * 3 * scaling_factor  # Penalty for falling short

        # 2. Final Goal Penalty/Reward for Year 18
        if current_year == 2047:  # Final year
            target_min = 700000
            target_max = 850000
            if self.portfolio_value < target_min:  # Penalize if below target
                reward -= (target_min - self.portfolio_value) * 1.0
            elif self.portfolio_value > target_max:  # Penalize if above target
                reward -= (self.portfolio_value - target_max) * 1.0
            else:  # Reward if within target range
                reward += 1000  # Reward for staying within the 700k to 850k range

        # 3. Inflation Adjustment Penalty
        inflation_adjusted_value = self.portfolio_value / (1 + (inflation_us + inflation_nigeria) / 2)
        inflation_impact = self.portfolio_value - inflation_adjusted_value
        reward -= inflation_impact * 0.025  # Light penalty for inflation erosion

        # 4. Smooth Growth Penalty
        if len(self.history) > 1:
            last_year_value = self.history[-2]['portfolio_value']
            growth_rate = (self.portfolio_value - last_year_value) / last_year_value
            if growth_rate > 0.12:  # Penalize excessive growth
                reward -= (growth_rate - 0.12) * 5000 * scaling_factor
            elif growth_rate < 0.04:  # Penalize insufficient growth
                reward -= (0.04 - growth_rate) * 4000 * scaling_factor

        # 5. Volatility Penalty
        asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)
        portfolio_return = np.dot(self.allocations, asset_returns)
        volatility = np.std(asset_returns)
        reward -= volatility * 500 * scaling_factor  # Penalize high volatility
        risk_adjusted_return = portfolio_return / (volatility + 1e-8)  # Positive reward for risk-adjusted return
        reward += risk_adjusted_return * 1

        # 6. Trajectory Target Penalty
        target_value = 100000 * (1 + 0.12) ** self.current_year  # Assume 7% annual growth
        deviation = abs(self.portfolio_value - target_value)
        reward -= deviation * 0.5 * scaling_factor  # Penalize deviation from trajectory

        # 7. Risky Allocations Penalty (Final Phase)
        if current_year > 2035:  # Encourage safer investments in the final phase
            asset_volatilities = np.std(asset_returns)  # Replace with actual volatilities if available
            risky_asset_weight = sum([w for w, vol in zip(self.allocations, asset_volatilities) if vol > 0.2])
            reward -= risky_asset_weight * 1000  # Penalize high-risk allocations

        # 8. Drawdown Penalty
        if len(self.history) > 1:
            last_year_value = self.history[-2]['portfolio_value']
            drawdown = (last_year_value - self.portfolio_value) / last_year_value
            if drawdown > 0.03:  # Penalize drops exceeding 3%
                reward -= drawdown * 10 * scaling_factor

        return reward

