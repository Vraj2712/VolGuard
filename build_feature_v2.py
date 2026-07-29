"""
Step 4 — Add better features.

Your database already has High, Low, and Volume (Step 2 only used Close), so we
can add genuinely new information WITHOUT downloading anything:

    range_hl     : (High - Low) / Close      -> how wide the day's swing was
    range_5      : 5-day average of range_hl  -> smoothed recent swing size
    parkinson_5  : Parkinson volatility (5d)  -> a range-based volatility estimate
    vol_ratio    : Volume / its 20-day average -> was trading unusually heavy?

These are computed in pandas (simpler than SQL for rolling range math), then a
new table `features_v2` is written back to the same database.

Usage:
    python build_features_v2.py
"""

import sqlite3
import numpy as np
import pandas as pd

DB      = "volguard.db"
OUTCSV  = "features_clean_v2.csv"

# 1) Pull the raw prices (OHLCV) back out of the database
conn = sqlite3.connect(DB)
px = pd.read_sql("SELECT * FROM prices", conn)
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Loaded {len(px):,} price rows, tickers: {sorted(px['ticker'].unique())}")

# 2) Build features per ticker (groupby keeps stocks from bleeding into each other)
def make_features(g):
    g = g.sort_values("date").copy()
    g["ret"]    = g["close"].pct_change()
    g["vol_5"]  = g["ret"].rolling(5).std()
    g["vol_10"] = g["ret"].rolling(10).std()
    g["vol_30"] = g["ret"].rolling(30).std()

    # --- NEW range-based features ---
    g["range_hl"] = (g["high"] - g["low"]) / g["close"]
    g["range_5"]  = g["range_hl"].rolling(5).mean()
    # Parkinson estimator: uses the high/low range, a classic vol proxy
    park = (np.log(g["high"] / g["low"]) ** 2) / (4 * np.log(2))
    g["parkinson_5"] = np.sqrt(park.rolling(5).mean())

    # --- NEW volume feature ---
    g["vol_ratio"] = g["volume"] / g["volume"].rolling(20).mean()

    # target: forward 5-day realized vol (same definition as Step 2)
    g["target_vol_fwd"] = g["vol_5"].shift(-5)
    return g

feat = pd.concat([make_features(g) for _, g in px.groupby("ticker")],
                 ignore_index=True)

# 3) Keep model-ready rows (all features present + a target)
cols = ["date", "ticker", "close", "ret",
        "vol_5", "vol_10", "vol_30",
        "range_hl", "range_5", "parkinson_5", "vol_ratio",
        "target_vol_fwd"]
clean = feat[cols].dropna().reset_index(drop=True)

# 4) Save back to the database and to CSV
clean.to_sql("features_v2", conn, if_exists="replace", index=False)
clean.to_csv(OUTCSV, index=False)
conn.close()

print(f"\nBuilt features_v2: {len(clean):,} clean rows")
print("New columns added: range_hl, range_5, parkinson_5, vol_ratio")
print("\nSample:")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(clean.head(5).to_string(index=False))
print(f"\nSaved -> table 'features_v2' in {DB}  and  {OUTCSV}")