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
import numpy as np
import pandas as pd
import shap
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
_explainer: shap.TreeExplainer | None = None


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


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases" or "decreases" default risk


class Explanation(BaseModel):
    default_proba: float
    default_pred: int
    threshold: float
    base_value: float
    top_features: List[FeatureContribution]


def get_explainer() -> shap.TreeExplainer:
    """Lazily build a TreeExplainer over the fitted gradient-boosted classifier."""
    global _explainer
    if _explainer is None:
        model = get_bundle()["model"]
        _explainer = shap.TreeExplainer(model.named_steps["clf"])
    return _explainer


def _prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean -> drop target -> add engineered features."""
    df = clean(df_raw).drop(columns=[TARGET], errors="ignore")
    return add_features(df)


def _score(df_raw: pd.DataFrame) -> pd.DataFrame:
    bundle = get_bundle()
    model, threshold = bundle["model"], bundle["threshold"]
    proba = model.predict_proba(_prepare(df_raw))[:, 1]
    return pd.DataFrame(
        {
            "default_proba": proba,
            "default_pred": (proba >= threshold).astype(int),
            "threshold": threshold,
        }
    )


def _explain_one(df_raw: pd.DataFrame, top_n: int) -> Explanation:
    bundle = get_bundle()
    model, threshold = bundle["model"], bundle["threshold"]
    X = _prepare(df_raw)

    # Run the preprocessing step the classifier was trained on, then SHAP.
    prep = model.named_steps["prep"]
    Xt = prep.transform(X)
    feat_names = list(prep.get_feature_names_out())

    explainer = get_explainer()
    shap_vals = explainer.shap_values(Xt)[0]  # single row, log-odds space
    base = float(np.ravel(explainer.expected_value)[0])

    order = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    contribs = [
        FeatureContribution(
            feature=feat_names[i],
            value=round(float(np.ravel(Xt[0])[i]), 4),
            shap_value=round(float(shap_vals[i]), 4),
            direction="increases" if shap_vals[i] > 0 else "decreases",
        )
        for i in order
    ]

    proba = float(model.predict_proba(X)[0, 1])
    return Explanation(
        default_proba=round(proba, 4),
        default_pred=int(proba >= threshold),
        threshold=round(float(threshold), 4),
        base_value=round(base, 4),
        top_features=contribs,
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


@app.post("/explain", response_model=Explanation)
def explain(applicant: Applicant, top_n: int = 8) -> Explanation:
    """Score an applicant and return the SHAP features driving the decision.

    Each contribution is in log-odds space: a positive ``shap_value`` pushes the
    prediction toward default, a negative one away from it, relative to the
    model's ``base_value`` (the average prediction over the training data).
    """
    return _explain_one(pd.DataFrame([applicant.model_dump()]), top_n)
