"""
Walk-forward validation — a more honest test than one lucky/unlucky split.

Previous scripts (validate_and_select.py, validate_v3.py, try_xgboost.py) picked
ONE chronological 60/20/20 split. That means the "beats baseline by X%" number
could just be an artifact of what happened to be in that one test window.

This script instead uses an EXPANDING walk-forward scheme:
    fold 1: oldest ~20% of days  -> seed training data only (never scored)
    fold 2: train on fold1,        test on fold2
    fold 3: train on fold1+2,      test on fold3
    fold 4: train on fold1+2+3,    test on fold4
    fold 5: train on fold1+2+3+4,  test on fold5

Every fold is trained only on data strictly BEFORE it (no look-ahead), and every
fold is out-of-sample. We report each candidate's RMSE averaged across all 4
test folds (mean +/- std) instead of a single number from a single split.

Usage:
    python walk_forward_validate.py
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DB = "volguard.db"
V1 = ["ret", "vol_5", "vol_10", "vol_30"]
V2 = V1 + ["range_hl", "range_5", "parkinson_5", "vol_ratio"]
V3 = V2 + ["vix", "vix_5", "vix_chg"]
SETS = {"v1 (original)": V1, "v2 (+range/vol)": V2, "v3 (+VIX)": V3}
TARGET, BASELINE = "target_vol_fwd", "vol_5"
N_FOLDS = 5   # 1 seed fold + 4 scored out-of-sample folds

def rmse(y, p): return np.sqrt(mean_squared_error(y, p))

def make_models():
    return {
        "Linear":      LinearRegression(),
        "RF depth=4":  RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0, n_jobs=-1),
        "RF depth=8":  RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1),
        "XGB depth=3": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1),
        "XGB depth=5": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1),
    }

# 1) Load and build N_FOLDS chronological blocks
conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM features_v3 ORDER BY date, ticker", conn)
conn.close()
df["date"] = pd.to_datetime(df["date"])
dates = np.sort(df["date"].unique())
edges = [dates[int(len(dates) * i / N_FOLDS)] for i in range(1, N_FOLDS)]
edges = [dates[0]] + edges + [dates[-1] + pd.Timedelta(days=1)]

df["fold"] = pd.cut(df["date"], bins=edges, labels=range(N_FOLDS),
                     right=False, include_lowest=True).astype(int)

print(f"{N_FOLDS} folds, boundaries:")
for i in range(N_FOLDS):
    d = df[df.fold == i]["date"]
    print(f"  fold {i}: {d.min().date()} -> {d.max().date()}  ({(df.fold==i).sum():,} rows)")
print()

# 2) For each candidate (feature set x model), walk forward and collect per-fold RMSE
rows = []
for sname, feats in SETS.items():
    for mname, _ in make_models().items():
        fold_rmses = []
        for test_fold in range(1, N_FOLDS):           # fold 0 is seed-only
            train = df[df.fold < test_fold]
            test  = df[df.fold == test_fold]
            model = make_models()[mname]
            model.fit(train[feats], train[TARGET])
            fold_rmses.append(rmse(test[TARGET], model.predict(test[feats])))
        rows.append({"features": sname, "model": mname,
                      "mean_RMSE": np.mean(fold_rmses), "std_RMSE": np.std(fold_rmses),
                      "fold_RMSEs": [round(r, 6) for r in fold_rmses]})

board = pd.DataFrame(rows).sort_values("mean_RMSE").reset_index(drop=True)
print("=== Walk-forward scoreboard (avg RMSE across 4 out-of-sample folds) ===")
print(board[["features", "model", "mean_RMSE", "std_RMSE"]].round(6).to_string(index=False))
print()
for _, r in board.iterrows():
    print(f"  {r['features']:16s} {r['model']:12s} folds={r['fold_RMSEs']}")

# 3) Naive baseline, walked forward the same way (no fitting needed, just per-fold RMSE)
base_rmses = []
for test_fold in range(1, N_FOLDS):
    test = df[df.fold == test_fold]
    base_rmses.append(rmse(test[TARGET], test[BASELINE]))
base_mean = np.mean(base_rmses)
print(f"\nNaive baseline (vol_5) walk-forward mean RMSE: {base_mean:.6f}  (per fold: {[round(r,6) for r in base_rmses]})")

# 4) Winner + honest improvement number
win = board.iloc[0]
improve = (base_mean - win["mean_RMSE"]) / base_mean * 100
print(f"\nWinner: {win['model']} on {win['features']}")
print(f"  mean RMSE {win['mean_RMSE']:.6f} +/- {win['std_RMSE']:.6f}")
print(f"  beats naive baseline by {improve:+.1f}% RMSE, averaged over {N_FOLDS-1} out-of-sample folds")

# 5) For comparison: the currently-documented single-split number
print("\n(For reference, the previously reported single-split number was +25.0% / +24.3%.)")
