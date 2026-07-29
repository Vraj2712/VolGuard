"""
Step 5 — Validate properly and check whether the new features actually helped.

Uses a chronological THREE-way split:
    [------ train ------][-- validation --][-- test --]
       oldest 60%            middle 20%       newest 20%

- train:      models learn their parameters
- validation: we compare feature sets + model settings here (as often as we like)
- test:       touched exactly ONCE at the end for the final, honest score

It compares the original features (v1) against the enriched set (v2) so you can
see, out-of-sample, whether Step 4 was worth it.

Usage:
    python validate_and_select.py
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DB = "volguard.db"
V1 = ["ret", "vol_5", "vol_10", "vol_30"]
V2 = V1 + ["range_hl", "range_5", "parkinson_5", "vol_ratio"]
TARGET, BASELINE = "target_vol_fwd", "vol_5"
MODEL_OUT = "volguard_model_final.pkl"

def rmse(y, p): return np.sqrt(mean_squared_error(y, p))

# 1) Load the enriched table and split chronologically (no shuffle)
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_v2 ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])
dates = df["date"].sort_values().values
d1 = dates[int(len(df) * 0.60)]   # train | val boundary
d2 = dates[int(len(df) * 0.80)]   # val   | test boundary
train = df[df.date < d1]
val   = df[(df.date >= d1) & (df.date < d2)]
test  = df[df.date >= d2]
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
print(f"boundaries: train<{pd.Timestamp(d1).date()} <=val< {pd.Timestamp(d2).date()} <=test\n")

# 2) Candidates: two feature sets x a few models. Score each on VALIDATION only.
def make_models():
    return {
        "Linear":        LinearRegression(),
        "RF depth=4":    RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0, n_jobs=-1),
        "RF depth=8":    RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1),
    }

rows = []
for fset_name, feats in [("v1 (original)", V1), ("v2 (enriched)", V2)]:
    for mname, model in make_models().items():
        model.fit(train[feats], train[TARGET])
        val_rmse = rmse(val[TARGET], model.predict(val[feats]))
        rows.append({"features": fset_name, "model": mname, "val_RMSE": val_rmse})

board = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
print("=== Validation scoreboard (used to CHOOSE — lower is better) ===")
print(board.round(6).to_string(index=False))

# 3) Pick the winner by validation score, then lock it in
win = board.iloc[0]
feats = V2 if "v2" in win["features"] else V1
final_model = make_models()[win["model"]]
print(f"\nChosen: {win['model']} on {win['features']}")

# 4) Retrain the winner on train+val, then evaluate ONCE on the untouched test set
trainval = pd.concat([train, val])
final_model.fit(trainval[feats], trainval[TARGET])
pred = final_model.predict(test[feats])

print("\n=== FINAL test-set score (touched once) ===")
def line(name, y, p):
    print(f"  {name:24s} RMSE={rmse(y,p):.6f}  MAE={mean_absolute_error(y,p):.6f}  R2={r2_score(y,p):+.4f}")
line("Naive baseline (vol_5)", test[TARGET], test[BASELINE])
line(f"Final model", test[TARGET], pred)

base = rmse(test[TARGET], test[BASELINE])
imp = (base - rmse(test[TARGET], pred)) / base * 100
print(f"\n  Final model beats naive baseline by {imp:+.1f}% RMSE")

# 5) Save the final model + its metadata
joblib.dump({"model": final_model, "features": feats, "target": TARGET,
             "val_boundary": str(pd.Timestamp(d1).date()),
             "test_boundary": str(pd.Timestamp(d2).date())}, MODEL_OUT)
print(f"\nSaved final model -> {MODEL_OUT}")