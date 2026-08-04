"""
AWS Lambda handler for VolGuard.

Dependency-free: uses only boto3 (built into the Lambda runtime). Loads the
model coefficients from S3 once per warm container, then predicts with plain
arithmetic. Accepts either a direct invoke or an API Gateway/Function URL
request.

The Function URL itself has no IAM auth (authorization_type = NONE, so anyone
can reach it) — access control instead happens here, via a shared API key
checked against the x-api-key header. Set VOLGUARD_API_KEY to enable it.

Environment variables:
    VOLGUARD_BUCKET     : your S3 bucket name
    VOLGUARD_MODEL_KEY  : model.json  (default)
    VOLGUARD_API_KEY    : shared secret required in the x-api-key header

Input (JSON):  {"ret": 0.01, "vol_5": 0.015, "vol_10": 0.014, "vol_30": 0.013}
Output (JSON): {"pred_daily_vol": 0.013847, "pred_annual_pct": 22.0}
"""

import os
import json
import hmac
import boto3

BUCKET  = os.environ.get("VOLGUARD_BUCKET")
KEY     = os.environ.get("VOLGUARD_MODEL_KEY", "model.json")
API_KEY = os.environ.get("VOLGUARD_API_KEY")

_model = None  # cached across warm invocations to avoid re-downloading


def _load_model():
    global _model
    if _model is None:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        _model = json.loads(obj["Body"].read())
    return _model


def _predict(feats, model):
    x = [float(feats[f]) for f in model["features"]]     # correct feature order
    daily = model["intercept"] + sum(c * xi for c, xi in zip(model["coef"], x))
    return daily, daily * (252 ** 0.5) * 100             # daily, annualised %


def _authorized(event, body):
    if not API_KEY:          # no key configured -> auth disabled (local/dev use)
        return True
    headers = event.get("headers") or {}
    provided = next((v for k, v in headers.items() if k.lower() == "x-api-key"), None)
    if provided is None and isinstance(body, dict):
        provided = body.get("api_key")
    return provided is not None and hmac.compare_digest(str(provided), API_KEY)


def handler(event, context):
    model = _load_model()

    # Unwrap the payload — API Gateway delivers a JSON string in event["body"];
    # a direct invoke passes the dict itself.
    body = event
    if isinstance(event, dict) and "body" in event:
        raw = event["body"]
        body = json.loads(raw) if isinstance(raw, str) else (raw or {})

    if not _authorized(event, body):
        return _resp(401, {"error": "unauthorized: missing or invalid x-api-key"})

    feats = body.get("features", body)                   # allow {"features": {...}} or flat
    try:
        missing = [f for f in model["features"] if f not in feats]
        if missing:
            return _resp(400, {"error": f"missing features: {missing}"})
        daily, annual = _predict(feats, model)
        return _resp(200, {
            "pred_daily_vol": round(daily, 6),
            "pred_annual_pct": round(annual, 1),
        })
    except (TypeError, ValueError) as e:
        return _resp(400, {"error": f"bad input: {e}"})


def _resp(code, obj):
    return {"statusCode": code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(obj)}