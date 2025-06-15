import yfinance as yf
import pandas as pd

# Set ticker symbols and date range
tickers = ["GOOGL", "AMZN", "AAPL", "AVGO", "META", "MSFT", "NVDA"]
start_date = "2012-05-01"
end_date = "2025-06-15"
try:
	# Download price data with adjustments
	data = yf.download(tickers, 
					   start=start_date, 
					   end=end_date, 
					   auto_adjust=True, 
					   progress=True)
finally:
	# # Get only the 'Close' prices
	# close_prices = data['Close']
	#
	# # Save to CSV
	# close_prices.to_csv("new_data/nvda_avgo_close_adjusted_2006_2023.csv")

    data.to_csv("new_data/ohlc_adjusted_mag7.csv")

	# Optional: Save to Parquet instead
	# close_prices.to_parquet("nvda_avgo_prices_2006_2023.parquet")

    print("Saved to ohlc_adjusted_mag7.csv")

