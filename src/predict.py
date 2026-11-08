"""Batch prediction CLI.
  python -m src.predict --input new_students.csv --scenario early --output predictions.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, SCENARIOS, risk_level


def ensure_models():
    """Self-healing: if model files are missing or unreadable (fresh clone,
    deleted models/, incompatible scikit-learn version), retrain them."""
    ok = True
    for sc in SCENARIOS:
        f = MODELS_DIR / f"model_{sc}.joblib"
        try:
            joblib.load(f)
        except Exception:
            ok = False
    if not ok:
        from .train import main as train_main
        print("Models missing or unreadable - training now (about a minute)...")
        train_main()


def load_bundle(scenario):
    ensure_models()
    return joblib.load(MODELS_DIR / f"model_{scenario}.joblib")


def align_columns(df: pd.DataFrame, bundle: dict):
    """Return (aligned frame, missing columns, ignored extra columns).
    Missing columns get the training-set default instead of failing."""
    feats = bundle["features"]
    missing = [c for c in feats if c not in df.columns]
    extra = [c for c in df.columns if c not in feats]
    X = df.reindex(columns=feats)
    for c in missing:
        X[c] = bundle["defaults"][c]
    return X, missing, extra


def predict_df(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    b = load_bundle(scenario)
    X, _, _ = align_columns(df, b)
    pred = np.clip(b["model"].predict(X), 0, 20)
    lo, hi = b["interval"]
    out = df.copy()
    out["predicted_G3"] = pred.round(1)
    out["interval_low"] = np.clip(pred + lo, 0, 20).round(1)
    out["interval_high"] = np.clip(pred + hi, 0, 20).round(1)
    out["risk_level"] = [risk_level(p) for p in pred]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="early")
    ap.add_argument("--output", default="predictions.csv")
    a = ap.parse_args()
    res = predict_df(pd.read_csv(a.input), a.scenario)
    res.to_csv(a.output, index=False)
    print(res["risk_level"].value_counts().to_string())
    print(f"Saved {len(res)} predictions -> {a.output}")
