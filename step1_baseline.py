"""
Step 1: Credit default risk - load data, clean, split, baseline models.

Dataset: "Default of Credit Card Clients" (UCI / Kaggle), 30,000 customers, 23 features.
Put ONE of these files in the same folder as this script:
  - UCI_Credit_Card.csv                       (Kaggle version)
  - default of credit card clients.xls        (UCI version; needs: pip install xlrd)

Run:  python step1_baseline.py
Outputs: printed metrics, outputs/step1_metrics.csv, outputs/roc_curves.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, recall_score, precision_score

SEED = 42
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------- 1. load
def load():
    if os.path.exists("UCI_Credit_Card.csv"):
        df = pd.read_csv("UCI_Credit_Card.csv")
    elif os.path.exists("default of credit card clients.xls"):
        df = pd.read_excel("default of credit card clients.xls", header=1)
    else:
        raise FileNotFoundError("Put UCI_Credit_Card.csv or 'default of credit card clients.xls' next to this script.")
    df = df.rename(columns={"default.payment.next.month": "default",
                            "default payment next month": "default",
                            "PAY_0": "PAY_1"})
    return df.drop(columns=[c for c in ["ID"] if c in df.columns])

df = load()
print("Shape:", df.shape)
print("Default rate: {:.1%}".format(df["default"].mean()))
print("Missing values:", int(df.isna().sum().sum()))

# ---------------------------------------------------------------- 2. clean
# EDUCATION: 0, 5, 6 are undocumented codes -> merge into 4 (others)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
# MARRIAGE: 0 is undocumented -> merge into 3 (others)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# ---------------------------------------------------------------- 3. features
# Engineered features (simple, explainable)
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt = [f"PAY_AMT{i}" for i in range(1, 7)]
pay_status =["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

df["AVG_BILL"] = df[bill].mean(axis=1)
df["AVG_PAY_AMT"] = df[pay_amt].mean(axis=1)
df["UTILIZATION"] = (df["AVG_BILL"] / df["LIMIT_BAL"]).clip(-1, 5)
df["PAY_RATIO"] = (df["AVG_PAY_AMT"] / df["AVG_BILL"].where(df["AVG_BILL"] > 0)).fillna(1).clip(0, 5)
df["MONTHS_LATE"] = (df[pay_status] > 0).sum(axis=1)
df["MAX_DELAY"] = df[pay_status].max(axis=1)

# SEX and MARRIAGE are NOT used as model inputs; they are kept aside for the fairness audit (later step).
protected = df[["SEX", "MARRIAGE"]].copy()
X = df.drop(columns=["default", "SEX", "MARRIAGE"])
y = df["default"]

# ---------------------------------------------------------------- 4. split (stratified 60/20/20)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
print(f"Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)}")

# ---------------------------------------------------------------- 5. models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                        class_weight="balanced", random_state=SEED),
}

def ks_stat(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))

def evaluate(name, p, y_true, thr):
    pred = (p >= thr).astype(int)
    return {"model": name, "ROC_AUC": roc_auc_score(y_true, p), "PR_AUC": average_precision_score(y_true, p),
            "KS": ks_stat(y_true, p), "threshold": thr,
            "recall_default": recall_score(y_true, pred), "precision_default": precision_score(y_true, pred)}

rows, plt_data = [], {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pv = m.predict_proba(X_va)[:, 1]
    # threshold picked on VALIDATION (best KS point), then applied once to TEST
    fpr, tpr, thr = roc_curve(y_va, pv)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    pt = m.predict_proba(X_te)[:, 1]
    rows.append(evaluate(name, pt, y_te, best_thr))
    plt_data[name] = roc_curve(y_te, pt)

res = pd.DataFrame(rows).round(3)
print("\nTEST SET RESULTS")
print(res.to_string(index=False))
res.to_csv("outputs/step1_metrics.csv", index=False)

plt.figure(figsize=(5.5, 5))
for name, (f, t, _) in plt_data.items():
    plt.plot(f, t, label=name)
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
plt.legend(); plt.tight_layout(); plt.savefig("outputs/roc_curves.png", dpi=150)
print("\nSaved outputs/step1_metrics.csv and outputs/roc_curves.png")
