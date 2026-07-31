"""
Build the Tableau dashboard CSV.

Produces ONE tidy file (volguard_dashboard.csv) where each row is one stock on
one day, with the model's predicted volatility next to the actual volatility that
happened. That single table feeds all four dashboard panels.

To keep the accuracy honest, the model is trained on the OLDER data only, then
used to predict every day — so the "test" rows are true out-of-sample forecasts.

Volatility is shown ANNUALISED as a percent (e.g. 22.0 = ~22% a year), which is
the form people recognise. It's the daily figure x sqrt(252) x 100.

Usage:
    python build_dashboard_csv.py
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DB        = "volguard.db"
FEATURES  = ["ret", "vol_5", "vol_10", "vol_30"]
TARGET    = "target_vol_fwd"
TEST_FRAC = 0.20
ANN       = np.sqrt(252) * 100        # daily volatility -> annualised %
OUT       = "volguard_dashboard.csv"

# 1) Load the clean feature table (every row has a known actual outcome)
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_clean ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])

# 2) Time split, train on the older part only, predict everything
cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - TEST_FRAC))]
train = df[df["date"] < cutoff]
model = LinearRegression().fit(train[FEATURES], train[TARGET])
df["pred"] = model.predict(df[FEATURES])

# 3) Assemble the tidy table
df["set"] = np.where(df["date"] < cutoff, "train", "test")
latest_per_ticker = df.groupby("ticker")["date"].transform("max")
df["is_latest"] = df["date"] == latest_per_ticker

out = pd.DataFrame({
    "date":          df["date"].dt.strftime("%Y-%m-%d"),
    "ticker":        df["ticker"],
    "actual_vol":    (df[TARGET] * ANN).round(2),
    "predicted_vol": (df["pred"] * ANN).round(2),
    "set":           df["set"],
    "is_latest":     df["is_latest"],
})
out["error"]     = (out["predicted_vol"] - out["actual_vol"]).round(2)
out["abs_error"] = out["error"].abs().round(2)

# tidy column order
out = out[["date", "ticker", "actual_vol", "predicted_vol",
           "error", "abs_error", "set", "is_latest"]]
out.to_csv(OUT, index=False)

# 4) Quick report
print(f"Wrote {OUT}: {len(out):,} rows, {out['ticker'].nunique()} tickers")
print(f"Date range: {out['date'].min()} -> {out['date'].max()}")
print(f"Test-set mean absolute error: {out.loc[out['set']=='test','abs_error'].mean():.2f}% (annualised)")
print("\nFirst rows:")
print(out.head(6).to_string(index=False))
print("\nMost recent row per ticker (is_latest = True) — powers the risk ranking:")
print(out[out["is_latest"]].to_string(index=False))