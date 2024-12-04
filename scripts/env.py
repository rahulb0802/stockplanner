import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, historical_data, initial_value, initial_allocations, phase_goals):
        super(PortfolioEnv, self).__init__()

        # Historical data (e.g., stock prices, inflation rates, returns)
        self.historical_data = historical_data
        self.phase_goals = phase_goals  # {year: goal_value}
        self.current_year = 0
        self.portfolio_value = initial_value
        self.allocations = np.array(initial_allocations)  # Initial allocations for 18 assets

        # Validate that initial allocations sum to 1
        if not np.isclose(np.sum(self.allocations), 1):
            raise ValueError("Initial allocations must sum to 1.")

        # State space: portfolio value, allocations, inflation rates
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(15,), dtype=np.float32  # 18 assets + 2 inflations
        )

        # Action space: adjustments to portfolio allocations
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(12,), dtype=np.float32  # Adjust allocations for 18 assets
        )

    def reset(self, seed=None, options=None):
        self.current_year = 0
        self.portfolio_value = 100000  # Entirely invested initially
        return self._get_state(), {}

    def _get_state(self):
        # Extract relevant features from historical data
        inflation_us = self.historical_data['US Inflation'][self.current_year]
        inflation_nigeria = self.historical_data['Nigeria Inflation'][self.current_year]

        state = np.concatenate(
            ([self.portfolio_value], self.allocations, [inflation_us, inflation_nigeria])
        )
        return state

    def step(self, action):
        # Adjust allocations based on action
        action = np.clip(action, -0.1, 0.1)  # Ensure changes are within allowable bounds
        self.allocations = np.clip(self.allocations + action, 0, 1)
        self.allocations /= np.sum(self.allocations)  # Normalize to sum to 1

        # Simulate portfolio growth
        tickers = self.historical_data.columns
        asset_returns = self.historical_data.iloc[self.current_year][tickers]
        portfolio_return = np.dot(self.allocations, asset_returns)
        
        self.portfolio_value *= (1 + portfolio_return)

        # Check phase goals and compute rewards
        reward = self._calculate_reward()
        self.current_year += 1

        done = self.current_year >= len(self.historical_data['returns'])
        return self._get_state(), reward, done, False, {}

    def _calculate_reward(self):
        # Reward based on proximity to the phase goal
        current_year = self.historical_data['year'][self.current_year]
        if current_year in self.phase_goals:
            phase_goal = self.phase_goals[current_year]
            reward = -abs(self.portfolio_value - phase_goal)
        else:
            reward = 0  # Neutral reward for non-phase goal years
        return reward
