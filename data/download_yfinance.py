import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# df_con = pd.read_csv("data-csv/snp-constituents.csv")
# tickers = df_con[df_con.date == '2017-01-03'].tickers.iloc[0].split(',')

url = "https://web.archive.org/web/20170129054116/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
table = soup.find("table", {"class": "wikitable"})
df = pd.read_html(str(table))[0]
tickers = df["Ticker symbol"].to_list()

# tickers = ['BF-B', 'BRK-B']

start_date = "2006-01-03"
end_date = "2025-08-01"
try:
    # Download price data with adjustments
    data = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=True, progress=True
    )
finally:
    # Get only the 'Close' prices
    # close_prices = data['Close']

    # Save to CSV
    # close_prices.to_csv("temp/close_adjusted_fixed_17.csv")

    data.to_csv("temp/ohlc_adjusted_fixed_17_WIKI.csv")
    # data.to_csv("temp/ohlc_bfb_brkb.csv")

    # Optional: Save to Parquet instead
    # close_prices.to_parquet("nvda_avgo_prices_2006_2023.parquet")

    print("Saved to temp/ohlc_adjusted_fixed_17_WIKI.csv")
