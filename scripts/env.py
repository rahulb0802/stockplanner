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
        action = np.clip(action, -0.3, 0.3)
        self.allocations = np.clip(self.allocations + action, 0.05, 0.20)
        self.allocations /= np.sum(self.allocations)

        # Bootstrapping: Sample asset returns with replacement
        asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)
        
        # Extreme Value Replacement: Replace extreme values in the returns
        asset_returns = self._replace_extreme_values(asset_returns)
        

        # Calculate portfolio return using the adjusted returns
        portfolio_return = np.dot(self.allocations, asset_returns)
        target_value = 900000  # Target for year 18
        years_left = max(1, 18 - self.current_year)  # Avoid division by 0
        growth_factor = (target_value / self.portfolio_value) ** (1 / years_left) - 1

        # Scale returns toward growth factor (soft limit)
        if portfolio_return > growth_factor:
            portfolio_return = (portfolio_return + growth_factor) / 2  # Smoothly scale down


        # Update the portfolio value based on the return
        self.portfolio_value *= (1 + portfolio_return)
        self.history.append({
            "year": self.historical_data.index[self.current_year],
            "portfolio_value": self.portfolio_value,
            "growth_rate": portfolio_return * 100,
            "allocations": self.allocations.tolist(),
            "reward": self._calculate_reward()  # Get reward based on phase goals and other factors
        })

        reward = self._calculate_reward()
        self.current_year += 1
        done = self.current_year >= len(self.historical_data) - 1
        truncated = False
        return self._get_state(), reward, done, truncated, {}

    def _replace_extreme_values(self, returns, lower_percentile=10, upper_percentile=90, std_dev_threshold=1.5):
        # Step 1: Percentile Clipping
        lower_bound = np.percentile(returns, lower_percentile)
        upper_bound = np.percentile(returns, upper_percentile)
        returns = np.clip(returns, lower_bound, upper_bound)

        # Step 2: Mean and Standard Deviation Clipping
        mean_return = np.mean(returns)
        std_dev_return = np.std(returns)
        strict_lower_bound = mean_return - std_dev_threshold * std_dev_return
        strict_upper_bound = mean_return + std_dev_threshold * std_dev_return
        returns = np.clip(returns, strict_lower_bound, strict_upper_bound)

        # Step 3: Scale Extreme Values
        returns = np.where(
            returns > upper_bound,
            mean_return + (returns - mean_return) * 0.2,  # Compress positives
            returns
        )
        returns = np.where(
            returns < lower_bound,
            mean_return + (returns - mean_return) * 0.2,  # Compress negatives
            returns
        )

        # Step 4: Normalize Returns
        returns = (returns - np.min(returns)) / (np.max(returns) - np.min(returns) + 0.1)
        returns *= 0.63

        return returns

    



    # def _bootstrap_returns(self):
    #     """
    #     Resample the returns with replacement from the historical data.
    #     """
    #     asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)
    #     return np.random.choice(asset_returns, size=len(asset_returns), replace=True)

    def _calculate_reward(self, growth_rate=None):
        """
        Enhanced reward function to encourage:
        - Meeting phase goals
        - Consistent, smooth growth
        - Low volatility
        - Staying within the $700k–$900k range by year 18
        """
        current_year = self.historical_data.index[self.current_year]
        inflation_us = self.historical_data.iloc[self.current_year]['US Inflation']
        inflation_nigeria = self.historical_data.iloc[self.current_year]['Nigeria Inflation']
        
        # Initialize reward
        reward = 0

        ### 1. Phase Goal Reward
        if current_year in self.phase_goals:
            phase_goal = self.phase_goals[current_year]
            if self.portfolio_value >= phase_goal:
                reward += 100  # Fixed reward for meeting the phase goal
            else:
                reward -= 100 * (phase_goal - self.portfolio_value) / phase_goal  # Scaled penalty for missing goal

        ### 2. Final Goal Reward for Year 18
        if current_year == 2047:  # Final year
            target_min = 700000
            target_max = 850000
            if self.portfolio_value < target_min:
                reward -= 100 * (target_min - self.portfolio_value) / target_min  # Scaled penalty for being below range
            elif self.portfolio_value > target_max:
                reward -= 100 * (self.portfolio_value - target_max) / target_max  # Scaled penalty for exceeding range
            else:
                reward += 500  # Large reward for staying within target range

        ### 3. Inflation Adjustment Penalty
        inflation_adjusted_value = self.portfolio_value / (1 + (inflation_us + inflation_nigeria) / 2)
        inflation_impact = self.portfolio_value - inflation_adjusted_value
        reward -= inflation_impact * 0.01  # Light penalty for inflation erosion

        ### 4. Smooth Growth Penalty
        if len(self.history) > 1:
            last_year_value = self.history[-2]['portfolio_value']
            growth_rate = (self.portfolio_value - last_year_value) / last_year_value
            if growth_rate > 0.10:  # Penalize excessive growth
                reward -= 50 * (growth_rate - 0.10) ** 2
            elif growth_rate < 0.05:  # Penalize insufficient growth
                reward -= 50 * (0.05 - growth_rate) ** 2

        ### 5. Volatility Penalty
        asset_returns = np.array(self.historical_data.iloc[self.current_year][1:13], dtype=np.float32)
        portfolio_return = np.dot(self.allocations, asset_returns)
        volatility = np.std(asset_returns)
        reward -= volatility * 10  # Scaled penalty for high volatility

        ### 6. Trajectory Target Penalty
        target_value = 100000 * (1 + 0.07) ** self.current_year  # Assume 7% annual growth
        deviation = abs(self.portfolio_value - target_value)
        reward -= deviation * 0.005  # Scaled penalty for deviation from trajectory

        ### 7. Drawdown Penalty
        if len(self.history) > 1:
            last_year_value = self.history[-2]['portfolio_value']
            drawdown = (last_year_value - self.portfolio_value) / last_year_value
            if drawdown > 0.03:  # Penalize large portfolio drops
                reward -= drawdown * 500

        ### 8. Risk-Adjusted Returns Bonus
        risk_adjusted_return = portfolio_return / (volatility + 1e-8)  # Avoid division by zero
        reward += risk_adjusted_return * 5  # Positive reward for better return-to-risk ratio

        
        return reward
    def get_average_return(self, method="arithmetic"):
        # Extract portfolio returns from the history
        returns = np.array([entry["growth_rate"] / 100 for entry in self.history])  # Convert % to decimals
        
        if method == "arithmetic":
            return np.mean(returns)
        elif method == "geometric":
            return np.prod(1 + returns) ** (1 / len(returns)) - 1
        else:
            raise ValueError("Method must be 'arithmetic' or 'geometric'")


