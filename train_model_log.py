"""
Step 3b — Same as train_model.py, but predicts LOG-volatility.

Volatility is always positive and spike-prone, which linear regression handles
poorly. Modelling log(vol) fixes the skew; we train in log space, then convert
predictions back with exp() so all scores are in the SAME units as before and
stay directly comparable to the naive baseline and GARCH.

Usage:
    pip install -U scikit-learn joblib arch
    python train_model_log.py
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Settings ------------------------------------------------------------
DB          = "volguard.db"
FEATURES    = ["ret", "vol_5", "vol_10", "vol_30"]
TARGET      = "target_vol_fwd"
BASELINE    = "vol_5"
TEST_FRAC   = 0.20
MODEL_OUT   = "volguard_model_log.pkl"
RUN_GARCH   = True
EPS         = 1e-6           # guards log(0)
# -------------------------------------------------------------------------

def rmse(y, p): return np.sqrt(mean_squared_error(y, p))
def report(name, y, p):
    return {"model": name, "RMSE": rmse(y, p), "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p)}

# 1) Load
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_clean ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers, "
      f"{df['date'].min().date()} -> {df['date'].max().date()}")

# 2) Time-based split (old = train, new = test, never shuffled)
cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - TEST_FRAC))]
train = df[df["date"] < cutoff]
test  = df[df["date"] >= cutoff]
print(f"Split at {cutoff.date()}  ->  train={len(train):,} rows, test={len(test):,} rows\n")

X_tr, X_te = train[FEATURES], test[FEATURES]
y_te = test[TARGET].values                     # actual vol (decimal units) for scoring

# Train on LOG of the target; predict, then exp() back to normal units
y_tr_log = np.log(train[TARGET].values + EPS)

results = []
# Naive baseline unchanged (already in normal units)
results.append(report("Naive baseline (vol_5)", y_te, test[BASELINE].values))

# Linear regression in log space
lin = LinearRegression().fit(X_tr, y_tr_log)
results.append(report("Linear (log-vol)", y_te, np.exp(lin.predict(X_te))))

# Random forest in log space
rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0,
                           n_jobs=-1).fit(X_tr, y_tr_log)
results.append(report("Random forest (log-vol)", y_te, np.exp(rf.predict(X_te))))

# Optional GARCH(1,1), unchanged
if RUN_GARCH:
    try:
        from arch import arch_model
        preds, actuals = [], []
        for tk, g in test.groupby("ticker"):
            hist = df[(df.ticker == tk) & (df.date < cutoff)]
            r = pd.concat([hist["ret"], g["ret"]]).values * 100
            n_train = len(hist)
            res = arch_model(r[:n_train], vol="Garch", p=1, q=1, mean="Zero").fit(disp="off")
            var5 = res.forecast(horizon=5, reindex=False).variance.values[-1].mean()
            preds.extend([np.sqrt(var5) / 100] * len(g))
            actuals.extend(g[TARGET].values)
        results.append(report("GARCH(1,1)", np.array(actuals), np.array(preds)))
    except Exception as e:
        print(f"(GARCH skipped: {e})\n")

# 3) Scoreboard
board = pd.DataFrame(results).set_index("model").round(6)
print("=== Test-set scoreboard (out-of-sample, log-vol models) ===")
print(board.to_string())

base_rmse = board.loc["Naive baseline (vol_5)", "RMSE"]
best = board["RMSE"].idxmin()
improve = (base_rmse - board.loc[best, "RMSE"]) / base_rmse * 100
print(f"\nBest model: {best}  ({improve:+.1f}% RMSE vs naive baseline)")

# 4) Save winner (note: log-space models need exp() on their output at predict time)
chosen = {"Linear (log-vol)": lin, "Random forest (log-vol)": rf}.get(best, lin)
is_log = best in ("Linear (log-vol)", "Random forest (log-vol)")
joblib.dump({"model": chosen, "features": FEATURES, "target": TARGET,
             "log_space": is_log, "cutoff_date": str(cutoff.date())}, MODEL_OUT)
print(f"Saved trained model -> {MODEL_OUT}"
      + ("  (predicts log-vol; apply np.exp to its output)" if is_log else ""))