"""
Step 6: Streamlit demo for the explainable credit risk project.
Run:  streamlit run streamlit_app.py
Needs everything from steps 1-5 (model/credit_model.joblib and the outputs/ folder) in this folder.
"""
import os
import pandas as pd
import streamlit as st
import step5_agent as A
from common import add_features

st.set_page_config(page_title="Explainable Credit Risk", page_icon="💳", layout="wide")


@st.cache_resource
def load_data():
    return A.load_test_applicants()


load_data()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Explanation engine")
choice = st.sidebar.selectbox("Mode", ["Template (no API key)", "Anthropic", "OpenAI-compatible"])
st.sidebar.caption("LLM modes read API keys from environment variables (ANTHROPIC_API_KEY, or OPENAI_API_KEY + "
                   "OPENAI_BASE_URL + AGENT_MODEL). Whatever the LLM writes is checked against the model's facts "
                   "and replaced by a template if it fails.")
provider = {"Template (no API key)": "template", "Anthropic": "anthropic", "OpenAI-compatible": "openai"}[choice]
try:
    llm = A.pick_llm(provider)
except Exception as e:
    st.sidebar.error(f"Could not start {choice}: {e}")
    llm = None

st.title("💳 Explainable Credit Risk Scoring")
st.caption("Credit-card default model with SHAP reason codes and a verified explanation agent. "
           "Demo on a public 2005 Taiwan dataset - not for real lending decisions.")

tab1, tab2, tab3 = st.tabs(["Decision", "Model performance", "Cut-off & fairness"])


def show_img(name):
    p = os.path.join("outputs", name)
    if os.path.exists(p):
        st.image(p)
    else:
        st.info(f"{name} not found - run the earlier steps first.")


def show_csv(name):
    p = os.path.join("outputs", name)
    if os.path.exists(p):
        st.dataframe(pd.read_csv(p))
    else:
        st.info(f"{name} not found - run the earlier steps first.")


# ---------------------------------------------------------------- tab 1: decision
with tab1:
    mode = st.radio("Applicant", ["Test-set applicant", "Custom applicant"], horizontal=True)
    applicant_id, go = None, False

    if mode == "Test-set applicant":
        ids = list(A.APPLICANTS.keys())[:300]
        applicant_id = st.selectbox("Applicant ID (held-out test set)", ids)
        go = st.button("Explain decision", type="primary")
    else:
        with st.form("custom"):
            limit = st.number_input("Credit limit (NT$)", 10000, 1000000, 100000, step=10000)
            st.markdown("**Repayment status per month, most recent first** "
                        "(-2 / -1 / 0 = no delay, 1 = one month late, 2 = two months late, ...)")
            cols = st.columns(6)
            defaults = [2, 2, 1, 0, 0, 0]
            pay = [cols[i].selectbox(f"Month {i + 1}", list(range(-2, 9)), index=defaults[i] + 2) for i in range(6)]
            with st.expander("Monthly bill amounts (NT$)"):
                bcols = st.columns(6)
                bills = [bcols[i].number_input(f"Bill {i + 1}", 0, 2000000, 40000, step=1000) for i in range(6)]
            with st.expander("Monthly payment amounts (NT$)"):
                pcols = st.columns(6)
                pays = [pcols[i].number_input(f"Paid {i + 1}", 0, 2000000, 2000, step=500) for i in range(6)]
            go = st.form_submit_button("Explain decision")
        if go:
            raw = {"LIMIT_BAL": limit}
            raw.update({f"PAY_{i + 1}": pay[i] for i in range(6)})
            raw.update({f"BILL_AMT{i + 1}": bills[i] for i in range(6)})
            raw.update({f"PAY_AMT{i + 1}": pays[i] for i in range(6)})
            A.APPLICANTS["custom"] = add_features(pd.DataFrame([raw])).iloc[0]
            applicant_id = "custom"

    if go and applicant_id is not None:
        with st.spinner("Scoring and explaining..."):
            try:
                res = A.run_agent(llm, applicant_id)
            except Exception as e:
                st.warning(f"The LLM call failed ({e}). Showing the template explanation instead.")
                res = A.run_agent(None, applicant_id)
        f = res["facts"]
        if "error" in f:
            st.error(f["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", f["decision"])
            c2.metric("Default probability", f"{f['default_probability'] * 100:.1f}%")
            c3.metric("Approval cut-off", f"{f['approval_cutoff'] * 100:.1f}%")
            (st.error if f["decision"] == "DECLINE" else st.success)(res["text"])
            if f["borderline"]:
                st.warning("Close to the cut-off: manual review recommended.")
            with st.expander("How this was produced"):
                st.write(f"Source: **{res['source']}** | verified against tool results: **{res['verified']}**")
                st.write(f"Tools called: {res['tools']}")
                if res["problems"]:
                    st.write(f"The LLM draft failed these checks: {res['problems']}")
                st.json(f)

# ---------------------------------------------------------------- tab 2: performance
with tab2:
    st.subheader("Test-set metrics")
    for name, label in [("step1_metrics.csv", "Baselines"), ("step2_metrics.csv", "WoE scorecard (6 features)"),
                        ("step3_metrics.csv", "Final gradient boosting model")]:
        st.markdown(f"**{label}**")
        show_csv(name)
    a, b = st.columns(2)
    with a:
        st.markdown("**ROC curves (baselines)**"); show_img("roc_curves.png")
        st.markdown("**SHAP: average impact**"); show_img("shap_summary_bar.png")
    with b:
        st.markdown("**Scorecard: default rate by score band**"); show_img("score_bands.png")
        st.markdown("**SHAP: direction of impact**"); show_img("shap_beeswarm.png")
    with st.expander("Scorecard points table"):
        show_csv("scorecard_points.csv")

# ---------------------------------------------------------------- tab 3: threshold + fairness
with tab3:
    st.subheader("Choosing the approval cut-off")
    st.caption("Illustrative assumption: approving a customer who repays earns +1 unit; approving one who defaults "
               "loses 3, 5 or 10 units. There is no real profit data in this dataset.")
    show_csv("threshold_summary.csv")
    show_img("threshold_curves.png")
    st.subheader("Fairness audit")
    st.caption("Sex, marital status and age were not used by the model; this checks outcomes by group. "
               "Small groups are noisy. This is a screening analysis, not a legal compliance test.")
    show_csv("fairness_audit.csv")
    show_img("fairness_approval.png")
