"""
Step 2: WoE / IV binning + logistic-regression credit scorecard.
Needs common.py and the dataset file in the same folder.
Run: python step2_scorecard.py
Outputs (in outputs/): iv_table.csv, scorecard_points.csv, score_bands.csv, score_bands.png, step2_metrics.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from common import prepare

os.makedirs("outputs", exist_ok=True)
X_tr, X_va, X_te, y_tr, y_va, y_te, protected = prepare()

# ------------------------------------------------------------ 1. binning (fit on TRAIN only)
def fit_bins(s):
    """Few distinct values -> each value is its own bin; otherwise ~10 quantile bins."""
    if s.nunique() <= 12:
        return {"kind": "cat"}
    _, edges = pd.qcut(s, q=10, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    return {"kind": "num", "edges": edges}

def apply_bins(s, spec):
    if spec["kind"] == "cat":
        return s.astype(int).astype(str)
    return pd.cut(s, spec["edges"], include_lowest=True).astype(str)

def woe_table(binned, y):
    t = pd.DataFrame({"bin": binned.values, "bad": y.values})
    g = t.groupby("bin")["bad"].agg(total="count", bad="sum")
    g["good"] = g["total"] - g["bad"]
    G, B = g["good"].sum(), g["bad"].sum()
    g["dist_good"] = (g["good"] + 0.5) / G
    g["dist_bad"] = (g["bad"] + 0.5) / B
    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])      # higher WoE = safer
    g["bad_rate"] = g["bad"] / g["total"]
    g["iv_part"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    return g

specs, tables, iv = {}, {}, {}
for col in X_tr.columns:
    specs[col] = fit_bins(X_tr[col])
    tables[col] = woe_table(apply_bins(X_tr[col], specs[col]), y_tr)
    iv[col] = tables[col]["iv_part"].sum()

iv_df = pd.Series(iv).sort_values(ascending=False).rename("IV").to_frame()
iv_df["strength"] = pd.cut(iv_df["IV"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                           labels=["useless", "weak", "medium", "strong", "very strong (check leakage)"])
iv_df.round(4).to_csv("outputs/iv_table.csv")
print("INFORMATION VALUE (top 12)\n", iv_df.head(12).round(3).to_string())

def to_woe(X):
    out = {}
    for col in X.columns:
        b = apply_bins(X[col], specs[col])
        out[col] = b.map(tables[col]["woe"]).fillna(0.0).values
    return pd.DataFrame(out, index=X.index)

W_tr, W_va, W_te = to_woe(X_tr), to_woe(X_va), to_woe(X_te)

# ------------------------------------------------------------ 2+3. feature selection + logistic regression
# AGE and EDUCATION are sensitive / proxy variables: kept OUT of the scorecard (use them in the fairness audit instead).
EXCLUDE = ["AGE", "EDUCATION"]
MAX_FEATURES, CORR_LIMIT = 12, 0.5
cands = [c for c in iv_df[iv_df["IV"] >= 0.02].index if c not in EXCLUDE]
corr = W_tr[cands].corr().abs()
keep = []
for c in cands:                                   # highest IV first; skip if too correlated with one already kept
    if all(corr.loc[c, k] < CORR_LIMIT for k in keep):
        keep.append(c)
    if len(keep) == MAX_FEATURES:
        break

while True:                                       # a safer bin must never LOSE points -> drop any wrong-sign feature
    lr = LogisticRegression(max_iter=2000).fit(W_tr[keep], y_tr)
    coefs = pd.Series(lr.coef_[0], index=keep)
    wrong = coefs[coefs > 0]
    if wrong.empty:
        break
    print("Dropping", wrong.idxmax(), "(positive coefficient)")
    keep.remove(wrong.idxmax())
print("\nSelected features:", keep)
print("Coefficients (all negative = every feature moves the score in the right direction):\n", coefs.round(3).to_string())

# ------------------------------------------------------------ 4. scorecard scaling
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20          # 600 points = 50:1 good:bad odds; +20 points doubles the odds
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)
n = len(keep)

rows = []
for col in keep:
    for b, r in tables[col].iterrows():
        pts = -(coefs[col] * r["woe"] + lr.intercept_[0] / n) * factor + offset / n
        rows.append({"feature": col, "bin": b, "count": int(r["total"]), "bad_rate": round(r["bad_rate"], 3),
                     "woe": round(r["woe"], 3), "points": round(pts, 1)})
card = pd.DataFrame(rows)
card.to_csv("outputs/scorecard_points.csv", index=False)

def score(W):
    logit_bad = lr.intercept_[0] + W[keep].values @ lr.coef_[0]
    return offset + factor * (-logit_bad)         # ln(odds_good) = -logit_bad

s_te, s_va = score(W_te), score(W_va)

# ------------------------------------------------------------ 5. evaluation (TEST)
def ks(y, s):
    f, t, _ = roc_curve(y, -s)
    return float(np.max(t - f))

auc = roc_auc_score(y_te, -s_te)
metrics = pd.DataFrame([{"model": "WoE scorecard (LogReg)", "ROC_AUC": auc, "Gini": 2 * auc - 1, "KS": ks(y_te, s_te)}]).round(3)
metrics.to_csv("outputs/step2_metrics.csv", index=False)
print("\nTEST SET RESULTS\n", metrics.to_string(index=False))
print(f"Score range on test: {s_te.min():.0f} - {s_te.max():.0f}  (mean {s_te.mean():.0f})")

# score bands: bad rate should fall steadily as the score rises
bands = pd.qcut(pd.Series(s_te, index=y_te.index), 10, duplicates="drop")
bt = pd.DataFrame({"band": bands, "bad": y_te}).groupby("band", observed=True)["bad"].agg(customers="count", bad_rate="mean")
bt["bad_rate"] = bt["bad_rate"].round(3)
bt.to_csv("outputs/score_bands.csv")
print("\nBAD RATE BY SCORE BAND (low score -> high risk)\n", bt.to_string())

plt.figure(figsize=(7, 4))
plt.bar(range(len(bt)), bt["bad_rate"] * 100, color="#4C72B0")
plt.xticks(range(len(bt)), [f"{int(b.left)}-{int(b.right)}" for b in bt.index], rotation=45, ha="right", fontsize=8)
plt.ylabel("Default rate (%)"); plt.xlabel("Score band"); plt.title("Default rate by score band (test set)")
plt.tight_layout(); plt.savefig("outputs/score_bands.png", dpi=150)
print("\nSaved files in outputs/")
