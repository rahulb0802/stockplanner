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
histories = []
for i in range(20):
    # Test the model
    state, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, reward, done, truncated, info = env.step(action)
        # print(f"Year: {env.current_year}, Portfolio Value: {env.portfolio_value}")
    # print(f"Average return: {env.get_average_return('arithmetic')}")
    # print(f"Total Dividends: {env.total_dividends}")
    histories.append(pd.DataFrame(env.history))
    # Save test results
    # pd.DataFrame(env.history).to_csv("portfolio_performance.csv", index=False)
for idx, history in enumerate(histories):
    history["Simulation"] = idx

# Combine all histories
combined_history = pd.concat(histories)

numeric_history = combined_history.select_dtypes(include=["number"])
average_results = numeric_history.groupby("year").mean()
std_results = numeric_history.groupby("year").std()

# Merge results for analysis
summary = average_results.copy()
for col in std_results.columns:
    summary[f"{col} Std"] = std_results[col]

# Example: Extract and expand the allocations column
allocations_df = combined_history["allocations"].apply(pd.Series)
allocations_df.columns = [f"Asset {i+1}" for i in range(allocations_df.shape[1])]
allocations_df["year"] = combined_history["year"]  # Add year for grouping

# Compute averages and standard deviations for allocations
average_allocations = allocations_df.groupby("year").mean()
std_allocations = allocations_df.groupby("year").std()

# Merge allocation stats with numeric stats
summary_allocations = average_allocations.copy()
for col in std_allocations.columns:
    summary_allocations[f"{col} Std"] = std_allocations[col]

summary.to_csv('numsummarywithdraw.csv')
summary_allocations.to_csv('allocationssummarywithdraw.csv')
