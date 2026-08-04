"""
Build the Tableau dashboard CSV.

Produces ONE tidy file (volguard_dashboard.csv) where each row is one stock on
one day, with the model's predicted volatility next to the actual volatility that
happened. That single table feeds all four dashboard panels.

To keep the accuracy honest, this uses the same WALK-FORWARD scheme as
finalize_model.py / walk_forward_validate.py: an expanding window trained only
on data strictly before each fold. Fold 0 (oldest ~20%) has no prior data, so
its predictions are in-sample ("train"); folds 1-4 are each predicted by a model
trained only on earlier folds, so "test" rows are genuine out-of-sample
forecasts across 4 different time periods — not one lucky/unlucky split.

Volatility is shown ANNUALISED as a percent (e.g. 22.0 = ~22% a year), which is
the form people recognise. It's the daily figure x sqrt(252) x 100.

Usage:
    python build_dashboard_csv.py
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DB        = "volguard.db"
FEATURES  = ["ret", "vol_5", "vol_10", "vol_30"]
TARGET    = "target_vol_fwd"
N_FOLDS   = 5          # 1 seed (in-sample) fold + 4 out-of-sample folds
ANN       = np.sqrt(252) * 100        # daily volatility -> annualised %
OUT       = "volguard_dashboard.csv"

# 1) Load the clean feature table (every row has a known actual outcome)
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_clean ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])

# 2) Walk-forward: expanding window, predict each fold with a model trained
#    only on strictly earlier folds. Fold 0 has no "earlier" data, so it's
#    scored in-sample and labelled "train"; folds 1-4 are true out-of-sample.
dates = np.sort(df["date"].unique())
edges = [dates[0]] + [dates[int(len(dates) * i / N_FOLDS)] for i in range(1, N_FOLDS)] \
        + [dates[-1] + pd.Timedelta(days=1)]
df["fold"] = pd.cut(df["date"], bins=edges, labels=range(N_FOLDS),
                     right=False, include_lowest=True).astype(int)

df["pred"] = np.nan
df["set"] = ""
for f in range(N_FOLDS):
    train = df[df.fold == 0] if f == 0 else df[df.fold < f]
    idx = df.fold == f
    model = LinearRegression().fit(train[FEATURES], train[TARGET])
    df.loc[idx, "pred"] = model.predict(df.loc[idx, FEATURES])
    df.loc[idx, "set"] = "train" if f == 0 else "test"

latest_per_ticker = df.groupby("ticker")["date"].transform("max")
df["is_latest"] = df["date"] == latest_per_ticker

out = pd.DataFrame({
    "date":          df["date"].dt.strftime("%Y-%m-%d"),
    "ticker":        df["ticker"],
    "fold":          df["fold"],
    "actual_vol":    (df[TARGET] * ANN).round(2),
    "predicted_vol": (df["pred"] * ANN).round(2),
    "set":           df["set"],
    "is_latest":     df["is_latest"],
})
out["error"]     = (out["predicted_vol"] - out["actual_vol"]).round(2)
out["abs_error"] = out["error"].abs().round(2)

# tidy column order
out = out[["date", "ticker", "fold", "actual_vol", "predicted_vol",
           "error", "abs_error", "set", "is_latest"]]
out.to_csv(OUT, index=False)

# 4) Quick report
print(f"Wrote {OUT}: {len(out):,} rows, {out['ticker'].nunique()} tickers")
print(f"Date range: {out['date'].min()} -> {out['date'].max()}")
print("\nWalk-forward MAE per fold (annualised %, out-of-sample folds only):")
for f in sorted(out.loc[out["set"] == "test", "fold"].unique()):
    fold_out = out[out["fold"] == f]
    mae = mean_absolute_error(fold_out["actual_vol"], fold_out["predicted_vol"])
    print(f"  fold {f}: MAE={mae:.2f}%  ({fold_out['date'].min()} -> {fold_out['date'].max()})")
print(f"\nOverall test-set mean absolute error: {out.loc[out['set']=='test','abs_error'].mean():.2f}% (annualised)")
print("\nFirst rows:")
print(out.head(6).to_string(index=False))
print("\nMost recent row per ticker (is_latest = True) — powers the risk ranking:")
print(out[out["is_latest"]].to_string(index=False))