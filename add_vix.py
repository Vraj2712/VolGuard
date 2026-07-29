"""
Step 4b — Add VIX (market-wide volatility index) as new features.

Unlike everything else, VIX is NOT derived from your five stocks — it's the
market's overall fear gauge, so it carries information your per-stock features
can't see. This pulls ^VIX, aligns it by date, and builds `features_v3`.

New features:
    vix      : the VIX closing level that day
    vix_5    : its 5-day average (recent market stress)
    vix_chg  : its 1-day change (is fear rising or falling?)

Usage:
    python add_vix.py
"""

import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf

DB     = "volguard.db"
OUTCSV = "features_clean_v3.csv"

# 1) Figure out the date range we need from the existing feature table
conn = sqlite3.connect(DB)
feat = pd.read_sql("SELECT * FROM features_v2", conn)
feat["date"] = pd.to_datetime(feat["date"])
start = (feat["date"].min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")  # pad for rolling
end   = (feat["date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

# 2) Pull VIX (single series). Handle both flat and MultiIndex column shapes.
print(f"Downloading ^VIX  {start} -> {end} ...")
raw = yf.download("^VIX", start=start, end=end, interval="1d",
                  auto_adjust=True, progress=False)
if raw is None or raw.empty:
    raise SystemExit("VIX download failed — check internet / yfinance is up to date.")
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

vix = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "vix"})
vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
vix = vix.sort_values("date").reset_index(drop=True)

# VIX-derived features
vix["vix_5"]   = vix["vix"].rolling(5).mean()
vix["vix_chg"] = vix["vix"].pct_change()
print(f"  got {len(vix):,} VIX days")

# 3) Merge onto the feature table by date (VIX is the same for every ticker that day)
feat["date"] = feat["date"].dt.normalize()
merged = feat.merge(vix[["date", "vix", "vix_5", "vix_chg"]], on="date", how="left")

missing = merged["vix"].isna().sum()
if missing:
    print(f"  note: {missing} rows had no VIX match; forward-filling then dropping leftovers")
    merged[["vix", "vix_5", "vix_chg"]] = merged[["vix", "vix_5", "vix_chg"]].ffill()
merged = merged.dropna(subset=["vix", "vix_5", "vix_chg"]).reset_index(drop=True)

# 4) Save features_v3
merged.to_sql("features_v3", conn, if_exists="replace", index=False)
merged.to_csv(OUTCSV, index=False)
conn.close()

print(f"\nBuilt features_v3: {len(merged):,} rows")
print("Added: vix, vix_5, vix_chg")
print("\nSample:")
with pd.option_context("display.width", 200, "display.max_columns", 25):
    print(merged[["date", "ticker", "vol_5", "vix", "vix_5", "vix_chg",
                  "target_vol_fwd"]].head(5).to_string(index=False))
print(f"\nSaved -> table 'features_v3' in {DB}  and  {OUTCSV}")