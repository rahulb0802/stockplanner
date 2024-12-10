import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv('allocationssummarywithdraw.csv')

allocations_df = pd.DataFrame({
    "Year": history['year'] + 1,
    "AMZN": history['Asset 1'],
    "EVRG": history['Asset 2'],
    "EXR": history['Asset 3'],
    "KMI": history['Asset 4'],
    "MAA": history['Asset 5'],
    "NU": history['Asset 6'],
    "NVDA": history['Asset 7'],
    "PSA": history['Asset 8'],
    "AVGO": history['Asset 9'],
    "T": history['Asset 10'],
    "VZ": history['Asset 11'],
    "XLF": history['Asset 12']
}).set_index("Year")

window_size = 10  # e.g., 10 episodes
for asset in allocations_df.columns:
    if asset != 'Year':  # Assuming 'episode' is the index column and doesn't need volatility calculation
        allocations_df[f'{asset} STD'] = allocations_df[asset].expanding().std()

# Now we have new columns 'std_asset_1', 'std_asset_2', etc., containing the rolling std dev

# Plot the volatility (rolling standard deviation) of each asset
plt.figure(figsize=(10, 6))

# Plot the volatility for each asset
for asset in allocations_df.columns:
    if asset.endswith('STD'):  # Only plot standard deviation columns
        sns.lineplot(data=allocations_df, x='Year', y=asset, label=asset)

plt.title('Volatility of Portfolio Allocations (Standard Deviation) Over Time')
plt.xlabel('Year')
plt.ylabel('Standard Deviation of Allocation')
plt.legend(title="Assets")
plt.xticks(range(1, 19, 1))
plt.ylim(0, 0.05)
plt.savefig('withdrawvisuals/allocationvolatilitywith.png')