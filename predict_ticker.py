"""
VolGuard — end-to-end local prediction.

Feed in one or more tickers; it pulls recent prices, builds the exact same
features the model was trained on, loads the saved model, and prints next week's
volatility forecast. Runs entirely on your machine.

Usage:
    python predict_ticker.py AAPL
    python predict_ticker.py AAPL MSFT TSLA
    python predict_ticker.py            # defaults to AAPL

Volatility = std of daily returns over the coming 5 trading days. Shown as a
daily figure and annualised (x sqrt(252)) since that's the familiar form.
"""

import sys
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

MODEL_IN          = "volguard_model.pkl"
FALLBACK_FEATURES = ["ret", "vol_5", "vol_10", "vol_30"]


def compute_features(close: pd.Series) -> pd.DataFrame:
    """Build features EXACTLY as build_features.py did (no train/serve skew)."""
    ret = close.pct_change()
    return pd.DataFrame({
        "ret":    ret,
        "vol_5":  ret.rolling(5).std(),    # sample std (ddof=1), matches training
        "vol_10": ret.rolling(10).std(),
        "vol_30": ret.rolling(30).std(),
    })


def get_close(ticker: str) -> pd.Series:
    """Pull ~6 months of daily closes for one ticker (enough for a 30-day window)."""
    raw = yf.download(ticker, period="6mo", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].dropna()


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL"]

    bundle   = joblib.load(MODEL_IN)
    model    = bundle["model"]
    features = bundle.get("features", FALLBACK_FEATURES)
    dm       = bundle.get("documented_metrics")
    print(f"Loaded model: linear regression on {features}")
    if dm:
        print(f"Documented test RMSE {dm['test_RMSE']} "
              f"({dm['improvement_vs_baseline_%']}% better than naive baseline)")
    print()

    results = []
    for tk in tickers:
        close = get_close(tk)
        if close is None or len(close) < 31:
            print(f"  {tk}: not enough data (bad symbol, or <31 trading days) — skipped")
            continue
        feats = compute_features(close).dropna()
        if feats.empty:
            print(f"  {tk}: could not build features — skipped")
            continue
        row = feats.iloc[[-1]][features]                 # latest complete row
        pred_daily = float(model.predict(row)[0])
        results.append({
            "ticker": tk,
            "as_of": close.index[-1].date(),
            "recent_vol_5": round(float(feats["vol_5"].iloc[-1]), 5),
            "pred_daily_vol": round(pred_daily, 5),
            "pred_annual_%": round(pred_daily * np.sqrt(252) * 100, 1),
        })

    if not results:
        print("No predictions produced.")
        return

    out = pd.DataFrame(results).sort_values("pred_annual_%", ascending=False)
    print("=== Next-week volatility forecast ===")
    print(out.to_string(index=False))
    print("\n'recent_vol_5' = last week's actual daily vol, for comparison.")
    print("Higher pred_annual_% = model expects a choppier week ahead.")


if __name__ == "__main__":
    main()