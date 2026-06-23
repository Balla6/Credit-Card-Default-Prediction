"""Exploratory data analysis: writes summary stats and figures to outputs/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data import BILL_COLS, PAY_AMT_COLS, PAY_COLS, TARGET, load_clean

OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG = OUT / "figures"

sns.set_theme(style="whitegrid")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    # ---- numeric summary ----------------------------------------------------
    summary = {
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "default_rate": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "limit_bal_median": float(df["LIMIT_BAL"].median()),
        "age_range": [int(df["AGE"].min()), int(df["AGE"].max())],
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- target balance -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET].map({0: "No default", 1: "Default"}).value_counts().plot.bar(
        ax=ax, color=["#4c72b0", "#c44e52"]
    )
    ax.set_title("Class balance")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG / "class_balance.png", dpi=120)
    plt.close(fig)

    # ---- default rate by key categoricals -----------------------------------
    for col, labels in {
        "SEX": {1: "Male", 2: "Female"},
        "EDUCATION": {1: "Grad", 2: "Univ", 3: "HS", 4: "Other"},
        "MARRIAGE": {1: "Married", 2: "Single", 3: "Other"},
    }.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        rates = df.groupby(col)[TARGET].mean()
        rates.index = [labels.get(i, str(i)) for i in rates.index]
        rates.plot.bar(ax=ax, color="#c44e52")
        ax.set_title(f"Default rate by {col}")
        ax.set_ylabel("Default rate")
        ax.axhline(df[TARGET].mean(), color="grey", ls="--", lw=1)
        fig.tight_layout()
        fig.savefig(FIG / f"default_rate_{col.lower()}.png", dpi=120)
        plt.close(fig)

    # ---- default rate by most recent repayment status -----------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("PAY_1")[TARGET].mean().plot.bar(ax=ax, color="#c44e52")
    ax.set_title("Default rate by latest repayment status (PAY_1)")
    ax.set_ylabel("Default rate")
    fig.tight_layout()
    fig.savefig(FIG / "default_rate_pay1.png", dpi=120)
    plt.close(fig)

    # ---- correlation heatmap ------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    corr_cols = ["LIMIT_BAL", "AGE", *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS, TARGET]
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(FIG / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
