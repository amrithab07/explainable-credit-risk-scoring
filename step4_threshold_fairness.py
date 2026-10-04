"""
Step 4: (a) choose the approval cut-off by business value, (b) fairness audit.
Needs: common.py, model/credit_model.joblib (from step3), dataset in the same folder.
Run: python step4_threshold_fairness.py
Outputs (outputs/): threshold_sweep.csv, threshold_summary.csv, threshold_curves.png, fairness_audit.csv, fairness_approval.png
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import prepare

os.makedirs("outputs", exist_ok=True)

# ============================ ASSUMPTIONS (edit these and re-run) ============================
# The dataset has no profit/loss data, so we use a simple, transparent cost model per applicant:
#   approve a customer who repays   -> +1 unit of profit
#   approve a customer who defaults -> -LOSS_RATIO units (a default wipes out LOSS_RATIO good customers' profit)
#   decline                         ->  0
LOSS_RATIO_MAIN = 5
LOSS_RATIOS = [3, 5, 10]            # sensitivity analysis
# =============================================================================================

bundle = joblib.load("model/credit_model.joblib")
model, features = bundle["model"], bundle["features"]
X_tr, X_va, X_te, y_tr, y_va, y_te, protected = prepare()

p_va = model.predict_proba(X_va[features])[:, 1]
p_te = model.predict_proba(X_te[features])[:, 1]
yv, yt = y_va.values, y_te.values
thresholds = np.round(np.arange(0.05, 0.81, 0.01), 2)

def sweep(p, y, L):
    N, rows = len(y), []
    for t in thresholds:
        app = p < t                                   # approve if predicted default probability is below cut-off
        n_app = int(app.sum())
        bads = int((y[app] == 1).sum())
        goods = n_app - bads
        rows.append({"threshold": t, "approval_rate": n_app / N,
                     "default_rate_among_approved": bads / n_app if n_app else np.nan,
                     "profit_per_applicant": (goods - L * bads) / N})
    return pd.DataFrame(rows)

# ------------------------------------------------------------ (a) threshold analysis
summary, curves = [], {}
for L in LOSS_RATIOS:
    val = sweep(p_va, yv, L)
    best_t = float(val.loc[val["profit_per_applicant"].idxmax(), "threshold"])   # chosen on VALIDATION only
    te = sweep(p_te, yt, L)
    curves[L] = te
    row = te[te["threshold"] == best_t].iloc[0]
    approve_all = ((yt == 0).sum() - L * (yt == 1).sum()) / len(yt)
    summary.append({"loss_ratio": L, "best_threshold": best_t,
                    "approval_rate": round(row["approval_rate"], 3),
                    "default_rate_among_approved": round(row["default_rate_among_approved"], 3),
                    "profit_per_applicant": round(row["profit_per_applicant"], 3),
                    "profit_if_approve_everyone": round(approve_all, 3)})
summary = pd.DataFrame(summary)
print("THRESHOLD ANALYSIS (test set; cut-off picked on validation)")
print(summary.to_string(index=False))
summary.to_csv("outputs/threshold_summary.csv", index=False)
curves[LOSS_RATIO_MAIN].to_csv("outputs/threshold_sweep.csv", index=False)

T_STAR = float(summary.loc[summary["loss_ratio"] == LOSS_RATIO_MAIN, "best_threshold"].iloc[0])
print(f"\nChosen cut-off for loss ratio {LOSS_RATIO_MAIN}: approve if default probability < {T_STAR}")

main = curves[LOSS_RATIO_MAIN]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(main["threshold"], main["approval_rate"] * 100, label="Approval rate (%)")
ax[0].plot(main["threshold"], main["default_rate_among_approved"] * 100, label="Default rate among approved (%)")
ax[0].axvline(T_STAR, ls="--", color="grey"); ax[0].set_xlabel("Approval cut-off (default probability)")
ax[0].set_title("Approval rate vs. default rate"); ax[0].legend()
for L, c in curves.items():
    ax[1].plot(c["threshold"], c["profit_per_applicant"], label=f"loss ratio {L}:1")
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_xlabel("Approval cut-off (default probability)")
ax[1].set_ylabel("Profit per applicant (units)"); ax[1].set_title("Expected profit by cut-off"); ax[1].legend()
plt.tight_layout(); plt.savefig("outputs/threshold_curves.png", dpi=150)

# ------------------------------------------------------------ (b) fairness audit at the chosen cut-off
approve = p_te < T_STAR
prot = protected.loc[X_te.index]
audit_cols = {
    "SEX": prot["SEX"].map({1: "Male", 2: "Female"}),
    "MARRIAGE": prot["MARRIAGE"].map({1: "Married", 2: "Single", 3: "Other"}),
    "AGE band": pd.cut(X_te["AGE"], [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]).astype(str),
}
rows = []
for attr, g in audit_cols.items():
    g = g.values
    part = []
    for val in pd.unique(g):
        m = g == val
        part.append({"attribute": attr, "group": val, "n": int(m.sum()),
                     "actual_default_rate": yt[m].mean(),
                     "avg_predicted_default": p_te[m].mean(),
                     "approval_rate": approve[m].mean(),
                     "approval_rate_of_good_customers": approve[m & (yt == 0)].mean(),
                     "approval_rate_of_bad_customers": approve[m & (yt == 1)].mean()})
    part = pd.DataFrame(part)
    part["approval_ratio_vs_best_group"] = part["approval_rate"] / part["approval_rate"].max()
    rows.append(part)
fair = pd.concat(rows).round(3)
fair.to_csv("outputs/fairness_audit.csv", index=False)
print("\nFAIRNESS AUDIT (attributes were NOT used by the model)")
print(fair.to_string(index=False))

plt.figure(figsize=(8, 4))
labels = [f"{a}: {g}" for a, g in zip(fair["attribute"], fair["group"])]
plt.barh(labels, fair["approval_rate"] * 100, color="#4C72B0")
plt.xlabel("Approval rate (%)"); plt.title("Approval rate by group (test set)")
plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig("outputs/fairness_approval.png", dpi=150)

# save the business cut-off for the app / agent
bundle["threshold_profit"] = T_STAR
bundle["loss_ratio"] = LOSS_RATIO_MAIN
joblib.dump(bundle, "model/credit_model.joblib")
print("\nSaved outputs/ and updated model/credit_model.joblib with threshold_profit")
