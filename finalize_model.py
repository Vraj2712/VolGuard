"""
Finalize the model.

Across all experiments (range/volume features, VIX, random forest, XGBoost,
GARCH), the simple linear model on the original features was never beaten —
confirmed with WALK-FORWARD validation across 4 independent out-of-sample
periods (see walk_forward_validate.py), not just one lucky train/test split.
So that's the one we ship. This script:

  1. reproduces the honest walk-forward score (for the record),
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
N_FOLDS   = 5          # 1 seed fold + 4 out-of-sample scored folds
MODEL_OUT = "volguard_model.pkl"

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

# 1) Load the clean feature table
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_clean ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])

# 2) Honest walk-forward score — for documentation.
#    Expanding window: fold 0 seeds training only; folds 1..N_FOLDS-1 are each
#    scored using a model trained on every fold strictly before it (no
#    look-ahead), so every reported number is genuinely out-of-sample.
dates = np.sort(df["date"].unique())
edges = [dates[0]] + [dates[int(len(dates) * i / N_FOLDS)] for i in range(1, N_FOLDS)] \
        + [dates[-1] + pd.Timedelta(days=1)]
df["fold"] = pd.cut(df["date"], bins=edges, labels=range(N_FOLDS),
                     right=False, include_lowest=True).astype(int)

fold_rmse, fold_mae, fold_r2, base_rmse = [], [], [], []
for test_fold in range(1, N_FOLDS):
    train, test = df[df.fold < test_fold], df[df.fold == test_fold]
    scorer = LinearRegression().fit(train[FEATURES], train[TARGET])
    pred = scorer.predict(test[FEATURES])
    fold_rmse.append(rmse(test[TARGET], pred))
    fold_mae.append(float(mean_absolute_error(test[TARGET], pred)))
    fold_r2.append(float(r2_score(test[TARGET], pred)))
    base_rmse.append(rmse(test[TARGET], test[BASELINE]))

metrics = {
    "method": f"walk-forward, {N_FOLDS - 1} out-of-sample folds",
    "test_RMSE": round(float(np.mean(fold_rmse)), 6),
    "test_RMSE_std": round(float(np.std(fold_rmse)), 6),
    "test_MAE": round(float(np.mean(fold_mae)), 6),
    "test_R2": round(float(np.mean(fold_r2)), 4),
    "baseline_RMSE": round(float(np.mean(base_rmse)), 6),
    "fold_RMSEs": [round(r, 6) for r in fold_rmse],
}
metrics["improvement_vs_baseline_%"] = round(
    (metrics["baseline_RMSE"] - metrics["test_RMSE"]) / metrics["baseline_RMSE"] * 100, 1)

print("Documented walk-forward performance (mean across out-of-sample folds):")
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