"""
Step 3: Gradient-boosting model + SHAP explanations + per-applicant reason codes.
Needs: common.py + dataset in the same folder.   pip install shap joblib
Run: python step3_shap.py
Outputs (outputs/): shap_summary_bar.png, shap_beeswarm.png, reason_codes_test.csv, step3_metrics.csv
         model/: credit_model.joblib  (model + feature list + decision threshold, used by the agent in a later step)
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score
from common import prepare, SEED

os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

X_tr, X_va, X_te, y_tr, y_va, y_te, protected = prepare()

# Same decision as the scorecard: sensitive / proxy variables stay OUT of the model (used later for the fairness audit)
EXCLUDE = ["AGE", "EDUCATION"]
features = [c for c in X_tr.columns if c not in EXCLUDE]
X_tr, X_va, X_te = X_tr[features], X_va[features], X_te[features]

# ------------------------------------------------------------ 1. model
model = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                                   min_samples_leaf=50, validation_fraction=0.1, n_iter_no_change=25,
                                   random_state=SEED)
model.fit(X_tr, y_tr)
print("Trees used:", model.n_estimators_)

pv = model.predict_proba(X_va)[:, 1]
fpr, tpr, thr = roc_curve(y_va, pv)
threshold = float(thr[np.argmax(tpr - fpr)])          # best-KS cut-off chosen on VALIDATION only

pt = model.predict_proba(X_te)[:, 1]
f, t, _ = roc_curve(y_te, pt)
pred = (pt >= threshold).astype(int)
res = pd.DataFrame([{"model": "Gradient Boosting (sklearn)", "ROC_AUC": roc_auc_score(y_te, pt), "KS": float(np.max(t - f)),
                     "threshold": threshold, "recall_default": recall_score(y_te, pred),
                     "precision_default": precision_score(y_te, pred)}]).round(3)
print("\nTEST SET RESULTS\n", res.to_string(index=False))
res.to_csv("outputs/step3_metrics.csv", index=False)

# ------------------------------------------------------------ 2. SHAP
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_te)
if isinstance(sv, list):          # older shap: [class0, class1]
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:                  # newer shap: (rows, features, classes)
    sv = sv[:, :, 1]
print("SHAP values shape:", sv.shape)       # positive SHAP = pushes the applicant TOWARD default

plt.figure()
shap.summary_plot(sv, X_te, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig("outputs/shap_summary_bar.png", dpi=150, bbox_inches="tight"); plt.close()
plt.figure()
shap.summary_plot(sv, X_te, show=False)
plt.tight_layout(); plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------ 3. reason codes
LABELS = {
    "LIMIT_BAL": "Credit limit",
    "PAY_1": "Repayment status, most recent month", "PAY_2": "Repayment status, 2 months ago",
    "PAY_3": "Repayment status, 3 months ago", "PAY_4": "Repayment status, 4 months ago",
    "PAY_5": "Repayment status, 5 months ago", "PAY_6": "Repayment status, 6 months ago",
    "AVG_BILL": "Average monthly bill", "AVG_PAY_AMT": "Average monthly payment",
    "UTILIZATION": "Credit utilisation (avg bill / limit)", "PAY_RATIO": "Payment-to-bill ratio",
    "MONTHS_LATE": "Number of months paid late (of last 6)", "MAX_DELAY": "Longest payment delay",
}
for i in range(1, 7):
    LABELS.setdefault(f"BILL_AMT{i}", f"Bill amount, {i} month(s) back")
    LABELS.setdefault(f"PAY_AMT{i}", f"Payment amount, {i} month(s) back")

def reason_codes(shap_row, value_row, cols, top_k=3):
    """Top features pushing this applicant TOWARD default (positive SHAP only)."""
    order = np.argsort(-shap_row)[:top_k]
    return [(LABELS.get(cols[j], cols[j]), value_row[j], float(shap_row[j])) for j in order if shap_row[j] > 0]

cols = list(X_te.columns)
rows = []
for k in range(len(X_te)):
    rc = reason_codes(sv[k], X_te.iloc[k].values, cols)
    rows.append({"applicant_index": X_te.index[k], "default_probability": round(float(pt[k]), 3),
                 "decision": "DECLINE" if pt[k] >= threshold else "APPROVE",
                 "actual_default": int(y_te.iloc[k]),
                 "reason_1": f"{rc[0][0]} = {rc[0][1]:.2f}" if len(rc) > 0 else "",
                 "reason_2": f"{rc[1][0]} = {rc[1][1]:.2f}" if len(rc) > 1 else "",
                 "reason_3": f"{rc[2][0]} = {rc[2][1]:.2f}" if len(rc) > 2 else ""})
out = pd.DataFrame(rows)
out.to_csv("outputs/reason_codes_test.csv", index=False)
print("\nSAMPLE DECLINED APPLICANTS")
print(out[out["decision"] == "DECLINE"].head(5).to_string(index=False))

# ------------------------------------------------------------ 4. save for the agent / app
joblib.dump({"model": model, "features": features, "threshold": threshold, "labels": LABELS}, "model/credit_model.joblib")
print("\nSaved outputs/ and model/credit_model.joblib")
