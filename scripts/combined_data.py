import inflation
import stock_data
import pandas as pd

combined_data_annual = pd.concat([stock_data.portfolio_annual, inflation.us_inflation, inflation.nigeria_inflation], axis=1)
combined_data_annual.dropna(inplace=True)
combined_data_annual.columns = list(stock_data.portfolio_data.columns) + ['US Inflation', 'Nigeria Inflation']
combined_data_annual.to_csv('combined_data_annual.csv')

combined_return_annual = pd.concat([stock_data.annual_returns, inflation.us_inflation, inflation.nigeria_inflation], axis=1)
combined_return_annual.dropna(inplace=True)
combined_return_annual.columns = list(stock_data.portfolio_data.columns) + ['US Inflation', 'Nigeria Inflation']
combined_return_annual.to_csv('combined_return_annual.csv')