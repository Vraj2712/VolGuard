"""
Step 5b — Re-run the honest validation, now including VIX (v3).

Same discipline as before: chronological train / validation / test.
Compares three feature sets so you can see whether VIX actually helped:
    v1 = original            (ret, vol_5/10/30)
    v2 = + range & volume    (range_hl, range_5, parkinson_5, vol_ratio)
    v3 = + VIX               (vix, vix_5, vix_chg)

Usage:
    python validate_v3.py
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
V3 = V2 + ["vix", "vix_5", "vix_chg"]
SETS = {"v1 (original)": V1, "v2 (+range/vol)": V2, "v3 (+VIX)": V3}
TARGET, BASELINE = "target_vol_fwd", "vol_5"
MODEL_OUT = "volguard_model_final.pkl"

def rmse(y, p): return np.sqrt(mean_squared_error(y, p))

# 1) Load features_v3 and split chronologically 60/20/20
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
        "Linear":     LinearRegression(),
        "RF depth=4": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0, n_jobs=-1),
        "RF depth=8": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1),
    }

# 2) Compare every (feature set x model) on VALIDATION only
rows = []
for sname, feats in SETS.items():
    for mname, model in make_models().items():
        model.fit(train[feats], train[TARGET])
        rows.append({"features": sname, "model": mname,
                     "val_RMSE": rmse(val[TARGET], model.predict(val[feats]))})
board = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
print("=== Validation scoreboard (used to CHOOSE — lower is better) ===")
print(board.round(6).to_string(index=False))

# 3) Lock in the winner, retrain on train+val, score ONCE on test
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

joblib.dump({"model": final, "features": feats, "target": TARGET}, MODEL_OUT)
print(f"\nSaved final model -> {MODEL_OUT}")