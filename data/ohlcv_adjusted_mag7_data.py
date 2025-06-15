import numpy as np
import pandas as pd

df_raw = pd.read_csv("data-csv/ohlcv_adjusted_mag7.csv")

features = df_raw.columns.values[1:]
tickers = df_raw.iloc[0].values[1:]
# Build MultiIndex: [(feature, ticker), ...]
multi_index = pd.MultiIndex.from_tuples(list(zip(features, tickers)))

# Step 3: Slice the actual data
df_clean = df_raw.iloc[2:].copy()
df_clean.columns = ['Date'] + list(multi_index)
df_clean.set_index('Date', inplace=True)
df_clean.index = pd.to_datetime(df_clean.index)

# Optional: convert all values to numeric (from strings)
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# Assume you already have df_clean and its columns are MultiIndex (feature, ticker)
df_clean.columns = pd.MultiIndex.from_tuples(df_clean.columns)

# Step 1: Normalize feature names by stripping trailing ".1", ".2", etc
new_columns = [(feat.split('.')[0], ticker) for feat, ticker in df_clean.columns]
df_clean.columns = pd.MultiIndex.from_tuples(new_columns)

# Step 2 (optional): Sort the index for clarity
df_clean = df_clean.sort_index(axis=1)
df_clean["date"] = pd.to_datetime(df_clean.index, errors="coerce")
df_clean.set_index("date", inplace=True)
df_clean = df_clean.round(4)

def get_open():
    return df_clean['Open']
def get_high():
    return df_clean['High']
def get_low():
    return df_clean['Low']
def get_close():
    return df_clean['Close']
def get_volume():
    return df_clean['Volume']
