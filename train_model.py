"""
Step 3 — Train the ML model (locally).

Loads the feature table from Step 2, splits it BY TIME (old = train, new = test,
never shuffled), trains simple models, and compares them against the naive
"tomorrow looks like today" baseline (and GARCH if the `arch` library is present).
Saves the trained model to disk.

Goal: a model that works, and a score you can defend.

Usage:
    pip install -U scikit-learn joblib
    python train_model.py
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
FEATURES    = ["ret", "vol_5", "vol_10", "vol_30"]   # inputs
TARGET      = "target_vol_fwd"                        # what we predict
BASELINE    = "vol_5"        # naive guess: next vol ≈ today's 5-day vol
TEST_FRAC   = 0.20           # last 20% of the timeline = test set
MODEL_OUT   = "volguard_model.pkl"
RUN_GARCH   = True           # set False to skip GARCH even if `arch` is installed
# -------------------------------------------------------------------------

def rmse(y, p):   # root mean squared error
    return np.sqrt(mean_squared_error(y, p))

def report(name, y, p):
    return {"model": name, "RMSE": rmse(y, p), "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p)}

# 1) Load the clean feature table
conn = sqlite3.connect(DB)
df = pd.read_sql(f"SELECT * FROM features_clean ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers, "
      f"{df['date'].min().date()} -> {df['date'].max().date()}")

# 2) TIME-BASED split — pick a cutoff date, everything before it trains,
#    everything on/after it tests. No shuffling: the model never sees the future.
cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - TEST_FRAC))]
train = df[df["date"] < cutoff]
test  = df[df["date"] >= cutoff]
print(f"Split at {cutoff.date()}  ->  train={len(train):,} rows, test={len(test):,} rows\n")

X_tr, y_tr = train[FEATURES], train[TARGET]
X_te, y_te = test[FEATURES],  test[TARGET]

results = []

# 3) Naive baseline: predict next vol = today's vol_5. No training involved.
results.append(report("Naive baseline (vol_5)", y_te, test[BASELINE]))

# 4) Simple model first: linear regression
lin = LinearRegression().fit(X_tr, y_tr)
results.append(report("Linear regression", y_te, lin.predict(X_te)))

# 5) A slightly stronger model for comparison: random forest
rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                           random_state=0, n_jobs=-1).fit(X_tr, y_tr)
results.append(report("Random forest", y_te, rf.predict(X_te)))

# 6) Optional: GARCH(1,1) per ticker (classic volatility model)
if RUN_GARCH:
    try:
        from arch import arch_model
        preds, actuals = [], []
        for tk, g in test.groupby("ticker"):
            hist = df[(df.ticker == tk) & (df.date < cutoff)]
            r = pd.concat([hist["ret"], g["ret"]]).values * 100  # arch likes %-scale
            n_train = len(hist)
            am = arch_model(r[:n_train], vol="Garch", p=1, q=1, mean="Zero")
            res = am.fit(disp="off")
            # forecast 5-day-ahead vol for each test point (expanding is ideal but slow;
            # we use the fitted model's 5-step forecast as a simple, defensible proxy)
            fc = res.forecast(horizon=5, reindex=False)
            var5 = fc.variance.values[-1].mean()          # avg daily var over next 5d
            garch_vol = np.sqrt(var5) / 100               # back to decimal scale
            preds.extend([garch_vol] * len(g))
            actuals.extend(g[TARGET].values)
        results.append(report("GARCH(1,1)", np.array(actuals), np.array(preds)))
    except Exception as e:
        print(f"(GARCH skipped: {e})\n")

# 7) Scoreboard — lower RMSE/MAE is better; higher R2 is better
board = pd.DataFrame(results).set_index("model").round(6)
print("=== Test-set scoreboard (out-of-sample) ===")
print(board.to_string())

base_rmse = board.loc["Naive baseline (vol_5)", "RMSE"]
best = board["RMSE"].idxmin()
best_rmse = board.loc[best, "RMSE"]
improve = (base_rmse - best_rmse) / base_rmse * 100
print(f"\nBest model: {best}")
print(f"  {improve:+.1f}% RMSE vs the naive baseline "
      f"({'beats' if improve > 0 else 'does NOT beat'} it)")

# 8) Save the trained model + metadata so it can be reused later
chosen = {"Linear regression": lin, "Random forest": rf}.get(best, lin)
joblib.dump({"model": chosen, "features": FEATURES, "target": TARGET,
             "cutoff_date": str(cutoff.date()), "trained_on": len(train)}, MODEL_OUT)
print(f"\nSaved trained model -> {MODEL_OUT}")