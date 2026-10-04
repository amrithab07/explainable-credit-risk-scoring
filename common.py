"""Shared data loading, cleaning, feature engineering and split for the credit risk project."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def load_raw():
    if os.path.exists("UCI_Credit_Card.csv"):
        df = pd.read_csv("UCI_Credit_Card.csv")
    elif os.path.exists("default of credit card clients.xls"):
        df = pd.read_excel("default of credit card clients.xls", header=1)
    else:
        raise FileNotFoundError("Put UCI_Credit_Card.csv or 'default of credit card clients.xls' next to this script.")
    df = df.rename(columns={"default.payment.next.month": "default",
                            "default payment next month": "default", "PAY_0": "PAY_1"})
    return df.drop(columns=[c for c in ["ID"] if c in df.columns])

def add_features(df):
    """Engineered, explainable features. Needs LIMIT_BAL, PAY_1..PAY_6, BILL_AMT1..6, PAY_AMT1..6."""
    df = df.copy()
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status = [f"PAY_{i}" for i in range(1, 7)]
    df["AVG_BILL"] = df[bill].mean(axis=1)
    df["AVG_PAY_AMT"] = df[pay_amt].mean(axis=1)
    df["UTILIZATION"] = (df["AVG_BILL"] / df["LIMIT_BAL"]).clip(-1, 5)
    df["PAY_RATIO"] = (df["AVG_PAY_AMT"] / df["AVG_BILL"].where(df["AVG_BILL"] > 0)).fillna(1).clip(0, 5)
    df["MONTHS_LATE"] = (df[pay_status] > 0).sum(axis=1)
    df["MAX_DELAY"] = df[pay_status].max(axis=1)
    return df

def prepare():
    """Returns X_tr, X_va, X_te, y_tr, y_va, y_te, protected (SEX/MARRIAGE, aligned to X index)."""
    df = load_raw()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df = add_features(df)
    protected = df[["SEX", "MARRIAGE"]].copy()
    X = df.drop(columns=["default", "SEX", "MARRIAGE"])
    y = df["default"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te, protected
