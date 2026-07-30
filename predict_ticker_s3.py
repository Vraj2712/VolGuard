"""
VolGuard — end-to-end prediction, model loaded from S3.

Same as predict_ticker.py, but the trained model is pulled from your S3 bucket
instead of a local file. This is the shape Step 6 (Lambda) needs: the model
lives in the cloud, the code fetches it at runtime.

Setup:
    pip install boto3
    # bucket name comes from an env var (falls back to the default below):
    export VOLGUARD_BUCKET=volguard-vraj-2026

Usage:
    python predict_ticker_s3.py AAPL
    python predict_ticker_s3.py AAPL MSFT TSLA
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import boto3

BUCKET            = os.environ.get("VOLGUARD_BUCKET", "volguard-vraj-2026")
MODEL_KEY         = os.environ.get("VOLGUARD_MODEL_KEY", "volguard_model.pkl")
FALLBACK_FEATURES = ["ret", "vol_5", "vol_10", "vol_30"]


def load_model():
    """Fetch the model from S3; fall back to a local copy if S3 isn't reachable."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
        bundle = joblib.load(io.BytesIO(obj["Body"].read()))
        print(f"Loaded model from s3://{BUCKET}/{MODEL_KEY}")
        return bundle
    except Exception as e:
        print(f"(S3 load failed: {e}\n falling back to local volguard_model.pkl)")
        return joblib.load("volguard_model.pkl")


def compute_features(close: pd.Series) -> pd.DataFrame:
    """Build features EXACTLY as build_features.py did (no train/serve skew)."""
    ret = close.pct_change()
    return pd.DataFrame({
        "ret":    ret,
        "vol_5":  ret.rolling(5).std(),
        "vol_10": ret.rolling(10).std(),
        "vol_30": ret.rolling(30).std(),
    })


def get_close(ticker: str) -> pd.Series:
    raw = yf.download(ticker, period="6mo", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].dropna()


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL"]

    bundle   = load_model()
    model    = bundle["model"]
    features = bundle.get("features", FALLBACK_FEATURES)
    print(f"Features: {features}\n")

    results = []
    for tk in tickers:
        close = get_close(tk)
        if close is None or len(close) < 31:
            print(f"  {tk}: not enough data — skipped")
            continue
        feats = compute_features(close).dropna()
        if feats.empty:
            print(f"  {tk}: could not build features — skipped")
            continue
        pred_daily = float(model.predict(feats.iloc[[-1]][features])[0])
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
    print("=== Next-week volatility forecast (model from S3) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()