"""
Step 2 — Build features + target in SQL (SQLite).

Reads the CSV from Step 1, loads it into a local SQLite database, then uses SQL
to compute:
    - daily returns
    - rolling volatility over 5, 10, and 30-day windows
    - the target: forward 5-day realized volatility (next-period vol)

Outputs two tables in volguard.db:
    prices          -> the raw data you loaded
    features        -> every row, including warm-up rows that are still NULL
    features_clean  -> model-ready rows only (no NULLs) + a CSV copy

Usage:
    python build_features.py
"""

import math
import sqlite3
import pandas as pd

# --- Settings ------------------------------------------------------------
INFILE   = "stock_prices.csv"   # from Step 1
DB       = "volguard.db"
HORIZON  = 5                     # target = realized vol over the NEXT 5 days
OUTCSV   = "features_clean.csv"
# -------------------------------------------------------------------------

# 1) Load the CSV and clean the Date column into plain ISO strings
#    (so SQLite sorts them correctly as text: '2022-07-28' < '2022-07-29')
print(f"Loading {INFILE} ...")
df = pd.read_csv(INFILE)
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = df.rename(columns=str.lower)          # Date->date, Close->close, etc.
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"  {len(df):,} rows, tickers: {sorted(df['ticker'].unique())}")

# 2) Connect to SQLite and load the prices table
conn = sqlite3.connect(DB)

# SQLite has no built-in STDDEV, and sqrt() isn't guaranteed to be compiled in,
# so we register our own safe sqrt. We compute variance in pure SQL and only
# take the square root here.
def safe_sqrt(x):
    if x is None:
        return None
    return math.sqrt(x) if x > 0 else 0.0
conn.create_function("sqrt", 1, safe_sqrt)

df.to_sql("prices", conn, if_exists="replace", index=False)

# 3) Build the features table entirely in SQL
#    variance (sample) = (Σx² − (Σx)²/n) / (n−1),  vol = sqrt(variance)
build_sql = f"""
DROP TABLE IF EXISTS features;
CREATE TABLE features AS
WITH rets AS (
    SELECT
        date,
        ticker,
        close,
        close / LAG(close) OVER w - 1.0 AS ret
    FROM prices
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
vols AS (
    SELECT
        date, ticker, close, ret,
        CASE WHEN COUNT(ret) OVER w5 = 5 THEN
            sqrt(MAX(0.0,
                (SUM(ret*ret) OVER w5 - SUM(ret) OVER w5 * SUM(ret) OVER w5 / 5.0) / 4.0))
        END AS vol_5,
        CASE WHEN COUNT(ret) OVER w10 = 10 THEN
            sqrt(MAX(0.0,
                (SUM(ret*ret) OVER w10 - SUM(ret) OVER w10 * SUM(ret) OVER w10 / 10.0) / 9.0))
        END AS vol_10,
        CASE WHEN COUNT(ret) OVER w30 = 30 THEN
            sqrt(MAX(0.0,
                (SUM(ret*ret) OVER w30 - SUM(ret) OVER w30 * SUM(ret) OVER w30 / 30.0) / 29.0))
        END AS vol_30
    FROM rets
    WINDOW
        w5  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN  4 PRECEDING AND CURRENT ROW),
        w10 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN  9 PRECEDING AND CURRENT ROW),
        w30 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
)
SELECT
    date, ticker, close, ret, vol_5, vol_10, vol_30,
    LEAD(vol_5, {HORIZON}) OVER (PARTITION BY ticker ORDER BY date) AS target_vol_fwd
FROM vols
ORDER BY ticker, date;
"""
conn.executescript(build_sql)

# 4) Model-ready table: keep only rows with a full 30-day window AND a target
conn.executescript("""
DROP TABLE IF EXISTS features_clean;
CREATE TABLE features_clean AS
SELECT * FROM features
WHERE vol_30 IS NOT NULL AND target_vol_fwd IS NOT NULL
ORDER BY ticker, date;
""")
conn.commit()

# 5) Proof it worked
n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
n_feat   = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
n_clean  = conn.execute("SELECT COUNT(*) FROM features_clean").fetchone()[0]

print("\nSuccess!")
print(f"  prices         : {n_prices:,} rows")
print(f"  features       : {n_feat:,} rows (includes NULL warm-up rows)")
print(f"  features_clean : {n_clean:,} rows (model-ready, no NULLs)")

print("\nSample of the clean feature table:")
sample = pd.read_sql(
    "SELECT * FROM features_clean ORDER BY ticker, date LIMIT 6", conn
)
with pd.option_context("display.width", 160, "display.max_columns", 20):
    print(sample.to_string(index=False))

# Export the clean table to CSV for convenience
pd.read_sql("SELECT * FROM features_clean ORDER BY ticker, date", conn).to_csv(
    OUTCSV, index=False
)
print(f"\nSaved model-ready table to: {OUTCSV}")
conn.close()