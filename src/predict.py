"""Score new applicants with the saved best model.

Usage:
    python predict.py path/to/new_data.csv [--out scored.csv]

The input CSV must contain the same raw columns as the training data
(ID/target optional). Outputs the original rows plus ``default_proba`` and a
``default_pred`` flag using the tuned decision threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data import TARGET, clean
from features import add_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "models" / "best_model.joblib"


def score(df_raw: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], bundle["threshold"]

    df = clean(df_raw)
    df = df.drop(columns=[TARGET], errors="ignore")
    X = add_features(df)

    proba = model.predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["default_proba"] = proba
    out["default_pred"] = (proba >= threshold).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV with raw applicant columns")
    ap.add_argument("--out", default="scored.csv")
    args = ap.parse_args()

    scored = score(pd.read_csv(args.input))
    scored.to_csv(args.out, index=False)
    print(f"Scored {len(scored)} rows -> {args.out}")
    print(scored[["default_proba", "default_pred"]].describe().round(4))


if __name__ == "__main__":
    main()
