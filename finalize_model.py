"""
Finalize the model.

Across all experiments (range/volume features, VIX, random forest, GARCH), the
simple linear model on the original features was never beaten. So that's the one
we ship. This script:

  1. reproduces the honest hold-out test score (for the record),
  2. retrains the same model on ALL available data (so the shipped model uses
     every day of history when predicting the future),
  3. saves it as volguard_model.pkl with its documented performance.

Usage:
    python finalize_model.py
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DB        = "volguard.db"
FEATURES  = ["ret", "vol_5", "vol_10", "vol_30"]   # the simple, winning set
TARGET    = "target_vol_fwd"
BASELINE  = "vol_5"
TEST_FRAC = 0.20
MODEL_OUT = "volguard_model.pkl"

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

# 1) Load the clean feature table
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_clean ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])

# 2) Honest hold-out score (time split, no shuffle) — for documentation
cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - TEST_FRAC))]
train, test = df[df.date < cutoff], df[df.date >= cutoff]
scorer = LinearRegression().fit(train[FEATURES], train[TARGET])
pred = scorer.predict(test[FEATURES])

metrics = {
    "test_RMSE": round(rmse(test[TARGET], pred), 6),
    "test_MAE":  round(float(mean_absolute_error(test[TARGET], pred)), 6),
    "test_R2":   round(float(r2_score(test[TARGET], pred)), 4),
    "baseline_RMSE": round(rmse(test[TARGET], test[BASELINE]), 6),
}
metrics["improvement_vs_baseline_%"] = round(
    (metrics["baseline_RMSE"] - metrics["test_RMSE"]) / metrics["baseline_RMSE"] * 100, 1)

print("Documented hold-out performance:")
for k, v in metrics.items():
    print(f"  {k:26s}: {v}")

# 3) Retrain on ALL data for the shipped model
final = LinearRegression().fit(df[FEATURES], df[TARGET])

# 4) Save with full metadata
joblib.dump({
    "model": final,
    "features": FEATURES,
    "target": TARGET,
    "target_horizon_days": 5,
    "trained_on": "all rows",
    "n_rows": len(df),
    "tickers": sorted(df["ticker"].unique().tolist()),
    "date_range": [str(df.date.min().date()), str(df.date.max().date())],
    "documented_metrics": metrics,
    "note": "Predicts forward 5-day realized daily volatility (std of daily returns).",
}, MODEL_OUT)
print(f"\nSaved final model -> {MODEL_OUT}  (trained on all {len(df):,} rows)")