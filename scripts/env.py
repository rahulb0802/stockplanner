import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, historical_data, initial_value, initial_allocations, phase_goals, benchmark_returns=None, is_training=True):
        super(PortfolioEnv, self).__init__()

        # Historical data (12 assets + 2 inflation columns)
        self.historical_data = historical_data
        self.phase_goals = phase_goals  # {year: goal_value}
        self.benchmark_returns = benchmark_returns
        self.current_year = 0
        self.portfolio_value = initial_value
        self.allocations = np.array(initial_allocations, dtype=np.float32)  # Initial allocations for 12 assets

        # Validate that initial allocations sum to 1
        if not np.isclose(np.sum(self.allocations), 1):
            raise ValueError("Initial allocations must sum to 1.")

        # State space: portfolio value, allocations, inflation rates
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(15,), dtype=np.float32  # 12 assets + 2 inflations + portfolio value
        )

        # Action space: adjustments to portfolio allocations
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(12,), dtype=np.float32  # Adjust allocations for 12 assets
        )

        self.history = []
        self.is_training = is_training
    def reset(self, seed=None, options=None):
        self.current_year = 0
        self.portfolio_value = 100000  # Entirely invested initially
        state = self._get_state()
        if not self.is_training:
            info = {
                'reward': None,
                'done': False,
            }
            return state, info
        return state, {}

    def _get_state(self):
        # Extract inflation rates
        inflation_us = self.historical_data.iloc[self.current_year]['US Inflation']
        inflation_nigeria = self.historical_data.iloc[self.current_year]['Nigeria Inflation']

        state = np.concatenate(
            ([self.portfolio_value], self.allocations, [inflation_us, inflation_nigeria])
        )
        return state

    def step(self, action):
        # Adjust allocations based on action
        action = np.clip(action, -0.4, 0.4)  # Ensure changes are within allowable bounds
        self.allocations = np.clip(self.allocations + action, 0, 1)
        self.allocations /= np.sum(self.allocations)  # Normalize to sum to 1

        # Simulate portfolio growth
        asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)  # First 12 columns
        portfolio_return = np.dot(self.allocations, asset_returns)
        
        self.portfolio_value *= (1 + portfolio_return)
        self.history.append({
            "year": self.historical_data.index[self.current_year],  # Get the year from the index
            "portfolio_value": self.portfolio_value,
            "allocations": self.allocations.tolist(),
            "reward": self._calculate_reward()
        })
        # Check phase goals and compute rewards
        reward = self._calculate_reward()
        self.current_year += 1

        done = self.current_year >= len(self.historical_data) - 1
        truncated = False
        return self._get_state(), reward, done, truncated, {}

    def _calculate_reward(self):
        """
        Reward function with:
        1. Phase goals: 50k by 2030, 75k by 2040, 100k by 2050.
        2. Maximum drop limit (1k) from one year to the next.
        3. Gradual, stable growth.
        4. Risk-adjusted return and minimizing volatility.
        """
        current_year = self.historical_data.index[self.current_year]
        inflation_us = self.historical_data.iloc[self.current_year]['US Inflation']
        inflation_nigeria = self.historical_data.iloc[self.current_year]['Nigeria Inflation']
        
        # Initialize reward
        reward = 0

        # Phase Goal Penalty (50k by 2030, 75k by 2040, 100k by 2050)
        phase_goals = {2030: 50000, 2040: 75000, 2050: 100000}
        
        if current_year in phase_goals:
            phase_goal = phase_goals[current_year]
            goal_diff = abs(self.portfolio_value - phase_goal)
            reward -= goal_diff * 1.5  # Stronger penalty if deviating from phase goals

        # Long-term goal (700k-800k after 17-18 years)
        if current_year == 2050:  # Final year (2050 for 18 years)
            if 700000 <= self.portfolio_value <= 800000:
                reward += 500  # Large reward for hitting the final goal
            else:
                reward -= abs(self.portfolio_value - 750000) * 0.2  # Penalty for missing the target

        # Inflation adjustment (penalize inflation erosion)
        inflation_adjusted_value = self.portfolio_value / (1 + (inflation_us + inflation_nigeria) / 2)
        reward -= (self.portfolio_value - inflation_adjusted_value) * 0.01  # Light penalty for inflation impact

        # Risk-Adjusted Return (reduce the penalty for volatility)
        asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)
        portfolio_return = np.dot(self.allocations, asset_returns)
        volatility = np.std(asset_returns)
        risk_adjusted_return = portfolio_return / (volatility + 1e-6)  # Avoid division by zero
        reward += risk_adjusted_return * 0.3  # Encourage positive returns with lower volatility

        # Minimize Drawdowns (penalize if the portfolio value drops by more than 10% from last year)
        if len(self.history) > 1:
            last_year_value = self.history[-2]['portfolio_value']
            drawdown = (last_year_value - self.portfolio_value) / last_year_value
            if drawdown > 0.1:  # Penalize if drop is greater than 10%
                reward -= drawdown * 2  # Larger penalty for significant drawdown

        # Smooth Growth (penalize excessive growth in a year)
        if len(self.history) > 1:
            growth_ratio = self.portfolio_value / self.history[-2]['portfolio_value']
            if growth_ratio > 1.2:  # Penalize excessive growth in a year (20% increase)
                reward -= (growth_ratio - 1.2) * 0.5  # Penalize large increases

        # Max Drop Penalty (limit the drop from one year to the next)
        if len(self.history) > 1:
            last_year_value = self.history[-2]['portfolio_value']
            drop = last_year_value - self.portfolio_value
            if drop > 1000:  # If the drop exceeds 1k, apply a penalty
                reward -= (drop - 1000) * 1  # Moderate penalty for exceeding drop threshold

        # Gradual Growth Penalty (penalize drastic increases from one year to another)
        if len(self.history) > 1:
            previous_value = self.history[-2]['portfolio_value']
            value_increase = self.portfolio_value - previous_value
            if value_increase > 1000:  # If the portfolio increases by more than 1k, apply a penalty
                reward -= (value_increase - 1000) * 0.5  # Moderate penalty for large increases

        return reward










