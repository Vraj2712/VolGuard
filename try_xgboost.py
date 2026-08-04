"""
Experiment — does XGBoost beat the linear model that's currently shipped?

Same discipline as validate_v3.py: chronological 60/20/20 train/val/test split,
compare feature sets (v1/v2/v3) x models on VALIDATION, lock in the winner,
score ONCE on test. XGBoost is added alongside Linear and Random Forest.

Usage:
    python try_xgboost.py
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DB = "volguard.db"
V1 = ["ret", "vol_5", "vol_10", "vol_30"]
V2 = V1 + ["range_hl", "range_5", "parkinson_5", "vol_ratio"]
V3 = V2 + ["vix", "vix_5", "vix_chg"]
SETS = {"v1 (original)": V1, "v2 (+range/vol)": V2, "v3 (+VIX)": V3}
TARGET, BASELINE = "target_vol_fwd", "vol_5"

def rmse(y, p): return np.sqrt(mean_squared_error(y, p))

conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_v3 ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])
dates = df["date"].sort_values().values
d1, d2 = dates[int(len(df) * 0.60)], dates[int(len(df) * 0.80)]
train = df[df.date < d1]
val   = df[(df.date >= d1) & (df.date < d2)]
test  = df[df.date >= d2]
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
print(f"train < {pd.Timestamp(d1).date()} <= val < {pd.Timestamp(d2).date()} <= test\n")

def make_models():
    return {
        "Linear":       LinearRegression(),
        "RF depth=4":   RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0, n_jobs=-1),
        "RF depth=8":   RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1),
        "XGB depth=3":  XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1),
        "XGB depth=5":  XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1),
    }

rows = []
for sname, feats in SETS.items():
    for mname, model in make_models().items():
        model.fit(train[feats], train[TARGET])
        rows.append({"features": sname, "model": mname,
                     "val_RMSE": rmse(val[TARGET], model.predict(val[feats]))})
board = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
print("=== Validation scoreboard (used to CHOOSE — lower is better) ===")
print(board.round(6).to_string(index=False))

win = board.iloc[0]
feats = SETS[win["features"]]
final = make_models()[win["model"]]
print(f"\nChosen: {win['model']} on {win['features']}")

trainval = pd.concat([train, val])
final.fit(trainval[feats], trainval[TARGET])
pred = final.predict(test[feats])

print("\n=== FINAL test-set score (touched once) ===")
def line(name, y, p):
    print(f"  {name:26s} RMSE={rmse(y,p):.6f}  MAE={mean_absolute_error(y,p):.6f}  R2={r2_score(y,p):+.4f}")
line("Naive baseline (vol_5)", test[TARGET], test[BASELINE])
line("Final model", test[TARGET], pred)
base = rmse(test[TARGET], test[BASELINE])
print(f"\n  Final model beats naive baseline by "
      f"{(base - rmse(test[TARGET], pred)) / base * 100:+.1f}% RMSE")

# For comparison: how does the CURRENTLY SHIPPED model (linear on v1) do on this same test set?
lin_v1 = LinearRegression().fit(trainval[V1], trainval[TARGET])
pred_v1 = lin_v1.predict(test[V1])
print("\n=== For comparison: currently shipped model (Linear on v1) ===")
line("Linear v1 (shipped)", test[TARGET], pred_v1)
