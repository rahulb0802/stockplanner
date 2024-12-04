import yfinance as yf
import pandas as pd

assets = ['AMZN', 'EVRG', 'EXR', 'KMI', 'MAA', 'NU', 'NVDA', 'PSA', 'AVGO', 'T', 'VZ', 'XLF']
data_frames = []

for asset in assets:
    try:
        stock_data = yf.download(asset, start='2005-01-01', end='2024-12-01')
        if not stock_data.empty:
            stock_data = stock_data[['Close']].rename(columns={'Close': asset})
            data_frames.append(stock_data)
        else:
            print(f"No data found for {asset}.")
    except Exception as e:
        print(f"Failed to download data for {asset}: {e}")

portfolio_data = pd.concat(data_frames, axis=1)
portfolio_data.fillna(method='ffill', inplace=True)
portfolio_data.fillna(method='bfill', inplace=True)

portfolio_data.to_csv('portfolio_data.csv', index=True)
print(portfolio_data.head())
portfolio_annual = portfolio_data.resample('YS').mean()
portfolio_annual.to_csv('portfolio_annual_averages.csv', index=True)
annual_returns = portfolio_annual.pct_change().dropna()