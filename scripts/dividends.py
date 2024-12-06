import yfinance as yf
import pandas as pd
import math

assets = ['AMZN', 'EVRG', 'EXR', 'KMI', 'MAA', 'NU', 'NVDA', 'PSA', 'AVGO', 'T', 'VZ', 'XLF']


dividend_yields = []

for ticker in assets:
    stock = yf.Ticker(ticker)
    try:
        # Fetch dividend yield (as a fraction of stock price)
        yield_data = stock.info.get('dividendYield', None)
        if yield_data is not None:
            dividend_yields.append(yield_data)  # Convert to percentage
        else:
            dividend_yields.append(0)  # If no yield data, assume 0
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        dividend_yields.append(None)  # Handle exceptions
