"""Feature engineering for the credit-default model.

Adds behavioural features on top of the raw columns. These capture repayment
trends and credit usage that the raw monthly snapshots only encode implicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import BILL_COLS, PAY_AMT_COLS, PAY_COLS, TARGET


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1.0  # guard against divide-by-zero on zero credit limits

    # How many of the last 6 months were delinquent (PAY_* >= 1).
    df["n_delinquent_months"] = (df[PAY_COLS] >= 1).sum(axis=1)
    # Worst and average repayment status over the window.
    df["max_pay_status"] = df[PAY_COLS].max(axis=1)
    df["mean_pay_status"] = df[PAY_COLS].mean(axis=1)

    # Credit utilisation: average bill relative to the credit limit.
    df["avg_bill"] = df[BILL_COLS].mean(axis=1)
    df["utilization"] = df["avg_bill"] / (df["LIMIT_BAL"] + eps)

    # Total paid vs total billed across the window (repayment coverage).
    total_bill = df[BILL_COLS].sum(axis=1)
    total_pay = df[PAY_AMT_COLS].sum(axis=1)
    df["pay_to_bill_ratio"] = total_pay / (total_bill.abs() + eps)

    # Trend in the bill amount: latest minus oldest (positive => growing debt).
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    # Months in the window with a zero payment.
    df["n_zero_payments"] = (df[PAY_AMT_COLS] == 0).sum(axis=1)

    return df


def split_xy(df: pd.DataFrame):
    df = add_features(df)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


__all__ = ["add_features", "split_xy"]
