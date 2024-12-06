from stable_baselines3 import PPO
from env import PortfolioEnv
import pandas as pd

# Load historical data and phase goals
historical_data = pd.read_csv("combined_return_annual.csv")
phase_goals = {2030: 50000, 2040: 75000, 2042: 100000}
initial_allocations = [0.15, 0.12693, 0.15, 0.04, 0.01, 0.01, 0.15, 0.00229, 0.0511, 0.01947, 0.15, 0.14021]
# Initialize environment
env = PortfolioEnv(historical_data=historical_data, initial_value=100000, initial_allocations=initial_allocations, phase_goals=phase_goals, is_training=False)

# Load the model
model = PPO.load("models/portfolio_rl_model.zip")

# Test the model
state, info = env.reset()
done = False
while not done:
    action, _ = model.predict(state)
    state, reward, done, truncated, info = env.step(action)
    print(f"Year: {env.current_year}, Portfolio Value: {env.portfolio_value}")
print(f"Average return: {env.get_average_return('arithmetic')}")
# Save test results
pd.DataFrame(env.history).to_csv("portfolio_performance.csv", index=False)
