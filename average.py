import pandas as pd

withdraw = pd.read_csv('numsummarywithdraw.csv')

print(withdraw['growth_rate'].mean())