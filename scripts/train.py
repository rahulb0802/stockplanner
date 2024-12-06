from stable_baselines3 import PPO
from env import PortfolioEnv
import pandas as pd

historical_data = pd.read_csv('combined_return_annual.csv')
phase_goals = {2030: 50000, 2040: 75000, 2042: 100000}
initial_allocations = [0.15, 0.12693, 0.15, 0.04, 0.01, 0.01, 0.15, 0.00229, 0.0511, 0.01947, 0.15, 0.14021]
env = PortfolioEnv(historical_data=historical_data, initial_value=100000, initial_allocations=initial_allocations, phase_goals=phase_goals, is_training=True)
state, _ = env.reset()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

model.save("models/portfolio_rl_model.zip")