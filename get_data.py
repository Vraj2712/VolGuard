"""
Step 1 — Get the data.

Pulls a few years of daily prices for a handful of stocks using yfinance
and saves them to a tidy CSV file. Run it, and it prints proof it worked.

Usage:
    pip install -U yfinance pandas
    python get_data.py
"""

import yfinance as yf
import pandas as pd
from datetime import date, timedelta

# --- Settings you can change ---------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # a handful of stocks
YEARS   = 3                                          # how far back to go
OUTFILE = "stock_prices.csv"
# -------------------------------------------------------------------------

END   = date.today()
START = END - timedelta(days=365 * YEARS)

print(f"Downloading {TICKERS}")
print(f"From {START} to {END} (daily)...\n")

# auto_adjust=True -> the "Close" column is already adjusted for splits/dividends.
# group_by="ticker" -> columns are organized as (Ticker, Field).
raw = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    progress=False,
    threads=True,
)

if raw is None or raw.empty:
    raise SystemExit(
        "No data came back. Usually this means:\n"
        "  1. yfinance is out of date  ->  pip install -U yfinance\n"
        "  2. No internet connection\n"
        "  3. A ticker symbol is wrong"
    )

# Reshape the wide MultiIndex table into a clean 'long' format:
# columns = Date, Ticker, Open, High, Low, Close, Volume
try:
    df = raw.stack(level=0, future_stack=True)   # pandas >= 2.1
except TypeError:
    df = raw.stack(level=0)                       # older pandas
df = (
    df.rename_axis(["Date", "Ticker"])
      .reset_index()
      .sort_values(["Ticker", "Date"])
      .reset_index(drop=True)
)

# Save it
df.to_csv(OUTFILE, index=False)

# --- Proof it actually worked --------------------------------------------
got = sorted(df["Ticker"].unique())
missing = [t for t in TICKERS if t not in got]

print("Success!\n")
print(f"Rows saved      : {len(df):,}")
print(f"Date range      : {df['Date'].min().date()}  ->  {df['Date'].max().date()}")
print(f"Tickers returned: {got}")
if missing:
    print(f"WARNING missing : {missing}  (check the symbol)")
print(f"\nFirst few rows:\n{df.head()}\n")
print(f"Saved to: {OUTFILE}")