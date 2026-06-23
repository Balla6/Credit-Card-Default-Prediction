"""Train and evaluate credit-default models, then persist the best one.

Trains Logistic Regression, Random Forest, XGBoost and LightGBM, compares them
on a held-out test set (stratified), selects the best by ROC-AUC, tunes the
decision threshold for F1, and saves the fitted pipeline + metrics + plots.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data import load_clean
from features import split_xy

warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG = OUT / "figures"
MODELS = OUT / "models"
SEED = 42

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", num_tf, numeric),
        ]
    )


def make_models(X: pd.DataFrame, pos_weight: float) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("prep", build_preprocessor(X, scale=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=SEED
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(X, scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", build_preprocessor(X, scale=False)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=pos_weight,
                        eval_metric="auc",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("prep", build_preprocessor(X, scale=False)),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=600,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=SEED,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def best_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 on the given data."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(
        2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0
    )
    # thr has one fewer element than prec/rec.
    return float(thr[max(0, np.argmax(f1[:-1]))])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    X, y = split_xy(load_clean())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    results = {}
    fitted = {}
    for name, model in make_models(X_train, pos_weight).items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "f1_at_0.5": float(f1_score(y_test, (proba >= 0.5).astype(int))),
        }
        fitted[name] = model
        print(f"{name:14s}  ROC-AUC={results[name]['roc_auc']:.4f}  "
              f"PR-AUC={results[name]['pr_auc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]
    best_proba = best_model.predict_proba(X_test)[:, 1]
    thr = best_threshold(y_test, best_proba)
    y_pred = (best_proba >= thr).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "models": results,
        "best_model": best_name,
        "tuned_threshold": thr,
        "test_roc_auc": results[best_name]["roc_auc"],
        "test_pr_auc": results[best_name]["pr_auc"],
        "test_f1_tuned": float(f1_score(y_test, y_pred)),
        "test_recall_default": float(report["1"]["recall"]),
        "test_precision_default": float(report["1"]["precision"]),
        "confusion_matrix": cm.tolist(),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nBest model: {best_name} (threshold={thr:.3f})")
    print(json.dumps({k: v for k, v in metrics.items() if k != "models"}, indent=2))

    # ---- ROC curve comparison ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} ({results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png", dpi=120)
    plt.close(fig)

    # ---- confusion matrix ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No default", "Default"])
    ax.set_yticks([0, 1], ["No default", "Default"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {best_name}")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ---- feature importance (if available) ----------------------------------
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = best_model.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(clf.feature_importances_, index=names).sort_values()[-20:]
        fig, ax = plt.subplots(figsize=(7, 7))
        imp.plot.barh(ax=ax, color="#4c72b0")
        ax.set_title(f"Top feature importances — {best_name}")
        fig.tight_layout()
        fig.savefig(FIG / "feature_importance.png", dpi=120)
        plt.close(fig)

    joblib.dump(
        {"model": best_model, "threshold": thr, "features": list(X.columns)},
        MODELS / "best_model.joblib",
    )
    print(f"\nSaved best model -> {MODELS / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
