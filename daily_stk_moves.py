import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the stock ticker and period
ticker = 'MSFT'
period = 'max'

# Fetch historical data from Yahoo Finance
msft_data = yf.download(ticker, period=period, interval='1d')

# Calculate daily returns
msft_data['Daily Return'] = msft_data['Adj Close'].pct_change()

# Drop NaN values that may result from calculating returns
msft_data.dropna(inplace=True)

# Create a 2x1 grid for plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Plot the MSFT price history on the top subplot
axs[0].plot(msft_data.index, msft_data['Adj Close'])
axs[0].set_title('MSFT Price History (Last 20 Years)')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Adjusted Close Price')

# Plot the histogram of daily returns on the bottom subplot
axs[1].hist(msft_data['Daily Return'], bins=100, edgecolor='k')
axs[1].set_title(f'Histogram of Daily Returns for MSFT\n'
                 f'Period={period}')
axs[1].set_xlabel('Daily Return')
axs[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()
