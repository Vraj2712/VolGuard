"""
Export the trained linear model to a tiny JSON file.

A linear regression is just: prediction = intercept + sum(coef_i * feature_i).
So instead of shipping scikit-learn to Lambda, we extract those numbers into
model.json — which any code can use with plain arithmetic. Then upload it to S3.

Usage:
    python export_model_json.py
    aws s3 cp model.json s3://YOUR-BUCKET/model.json
"""

import json
import joblib

MODEL_IN = "volguard_model.pkl"
JSON_OUT = "model.json"

bundle = joblib.load(MODEL_IN)
model, features = bundle["model"], bundle["features"]

payload = {
    "features": features,                       # order matters!
    "coef": [float(c) for c in model.coef_],
    "intercept": float(model.intercept_),
    "target_horizon_days": bundle.get("target_horizon_days", 5),
    "documented_metrics": bundle.get("documented_metrics", {}),
}

with open(JSON_OUT, "w") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote {JSON_OUT}")
print(json.dumps(payload, indent=2))

# quick self-check: reproduce a prediction from the JSON with pure Python
demo = {"ret": 0.01, "vol_5": 0.015, "vol_10": 0.014, "vol_30": 0.013}
x = [demo[f] for f in payload["features"]]
pred = payload["intercept"] + sum(c * xi for c, xi in zip(payload["coef"], x))
print(f"\nSelf-check prediction for {demo}:\n  {pred:.6f}  (pure-Python math works)")