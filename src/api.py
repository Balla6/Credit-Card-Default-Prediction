"""FastAPI scorer for the credit-card default model.

Run locally:
    uvicorn api:app --reload --app-dir src

Then open http://127.0.0.1:8000/docs for the interactive UI, or POST raw
applicant records to /predict.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import TARGET, clean
from features import add_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "models" / "best_model.joblib"

app = FastAPI(
    title="Credit Card Default Scorer",
    description="Predicts probability of default on next month's payment "
    "(UCI Credit Card dataset).",
    version="1.0.0",
)

_bundle: dict | None = None


def get_bundle() -> dict:
    """Lazy-load the model bundle so the app starts even before training."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python src/train.py` first.",
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Applicant(BaseModel):
    """One raw applicant record (same columns as the training data)."""

    LIMIT_BAL: float = Field(..., example=20000)
    SEX: int = Field(..., example=2)
    EDUCATION: int = Field(..., example=2)
    MARRIAGE: int = Field(..., example=1)
    AGE: int = Field(..., example=24)
    PAY_0: int = Field(..., example=2)
    PAY_2: int = Field(..., example=2)
    PAY_3: int = Field(..., example=-1)
    PAY_4: int = Field(..., example=-1)
    PAY_5: int = Field(..., example=-2)
    PAY_6: int = Field(..., example=-2)
    BILL_AMT1: float = Field(..., example=3913)
    BILL_AMT2: float = Field(..., example=3102)
    BILL_AMT3: float = Field(..., example=689)
    BILL_AMT4: float = Field(..., example=0)
    BILL_AMT5: float = Field(..., example=0)
    BILL_AMT6: float = Field(..., example=0)
    PAY_AMT1: float = Field(..., example=0)
    PAY_AMT2: float = Field(..., example=689)
    PAY_AMT3: float = Field(..., example=0)
    PAY_AMT4: float = Field(..., example=0)
    PAY_AMT5: float = Field(..., example=0)
    PAY_AMT6: float = Field(..., example=0)


class Prediction(BaseModel):
    default_proba: float
    default_pred: int
    threshold: float


class BatchRequest(BaseModel):
    records: List[Applicant]


def _score(df_raw: pd.DataFrame) -> pd.DataFrame:
    bundle = get_bundle()
    model, threshold = bundle["model"], bundle["threshold"]
    df = clean(df_raw).drop(columns=[TARGET], errors="ignore")
    proba = model.predict_proba(add_features(df))[:, 1]
    return pd.DataFrame(
        {
            "default_proba": proba,
            "default_pred": (proba >= threshold).astype(int),
            "threshold": threshold,
        }
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> Prediction:
    row = _score(pd.DataFrame([applicant.model_dump()])).iloc[0]
    return Prediction(
        default_proba=round(float(row["default_proba"]), 4),
        default_pred=int(row["default_pred"]),
        threshold=round(float(row["threshold"]), 4),
    )


@app.post("/predict_batch", response_model=List[Prediction])
def predict_batch(req: BatchRequest) -> List[Prediction]:
    df = pd.DataFrame([r.model_dump() for r in req.records])
    scored = _score(df)
    return [
        Prediction(
            default_proba=round(float(r.default_proba), 4),
            default_pred=int(r.default_pred),
            threshold=round(float(r.threshold), 4),
        )
        for r in scored.itertuples()
    ]
