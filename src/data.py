"""Data loading and cleaning for the UCI Credit Card Default dataset.

The raw file ships with a few documented quirks that we normalise here so the
rest of the pipeline can treat the columns as clean categoricals/numerics:

* ``EDUCATION`` contains undocumented codes 0, 5 and 6 -> folded into 4 ("other").
* ``MARRIAGE`` contains an undocumented code 0 -> folded into 3 ("other").
* ``PAY_0`` is renamed to ``PAY_1`` so the repayment-status columns read in order.
* ``default.payment.next.month`` is renamed to the shorter ``default``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data_raw" / "UCI_Credit_Card.csv"

TARGET = "default"

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as distributed."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with normalised categories and tidy column names."""
    df = df.copy()

    df = df.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": TARGET})
    df = df.drop(columns=["ID"], errors="ignore")

    # Fold undocumented category codes into the "other" bucket.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d[TARGET].value_counts(normalize=True).round(4).to_dict())
