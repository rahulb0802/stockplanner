import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv('numsummary.csv')
data = pd.DataFrame({
    "Year": history['year'] + 1,
    "Portfolio Return": history['growth_rate']
})

# Plot yearly returns
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="Year", y="Portfolio Return", hue="Year", legend=False, palette='crest')

plt.title("Yearly Returns")
plt.xlabel("Year")
plt.ylabel("Return (%)")
plt.savefig('visuals/yearly_returns.png')
