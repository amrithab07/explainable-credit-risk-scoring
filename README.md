# 💳 Explainable Credit Risk Scoring

An end-to-end credit-default project: a **WoE/IV logistic-regression scorecard** and a **gradient-boosting model**, **SHAP-based reason codes**, a **profit-based approval cut-off**, a **fairness audit**, and a **tool-using LLM agent** that explains each decision and is checked against the model's facts before anything is shown. A Streamlit app ties it together.

> Built on a public 2005 Taiwan credit-card dataset for learning and portfolio purposes. It is not a real lending system.

---

## 📌 Overview

Lenders need more than a probability. They need to know **why** an applicant was declined, **where** to set the approval line, and whether the model treats groups consistently. This project covers those questions step by step:

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_baseline.py` | Clean data, engineer features, stratified split, baseline models |
| 2 | `step2_scorecard.py` | WoE/IV binning and a points-based logistic-regression scorecard |
| 3 | `step3_shap.py` | Final gradient-boosting model, SHAP explanations, per-applicant reason codes |
| 4 | `step4_threshold_fairness.py` | Approval cut-off by expected profit, and a fairness audit |
| 5 | `step5_agent.py` | LLM agent that calls the model as a tool and writes a verified explanation |
| 6 | `streamlit_app.py` | Interactive demo |

---

## 📊 Dataset

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI Machine Learning Repository), 30,000 customers, 23 features, binary target = default next month. The overall default rate is **22.1%**.

- Features: credit limit, 6 months of repayment status, bill amounts and payment amounts (plus demographics).
- Split: stratified **60% train / 20% validation / 20% test** (18,000 / 6,000 / 6,000). Binning, feature selection and cut-offs are fitted on train/validation only; the test set is used for reporting.
- The data file is not included in this repo. Download it from the link above (`.xls`) or from Kaggle (`UCI_Credit_Card.csv`) and place it next to the scripts.

Citation: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2).

---

## 🧹 Features

Simple, explainable features were engineered from the raw columns:

- `AVG_BILL`, `AVG_PAY_AMT`: monthly averages
- `UTILIZATION`: average bill / credit limit
- `PAY_RATIO`: average payment / average bill
- `MONTHS_LATE`: number of the last 6 months paid late
- `MAX_DELAY`: longest payment delay

**Responsible-modelling choices:** `SEX` and `MARRIAGE` are never used as model inputs. The scorecard and the final model also exclude `AGE` and `EDUCATION`. These variables are kept aside and used only in the fairness audit.

---

## 🏆 Results (held-out test set, 6,000 customers)

| Model | ROC-AUC | KS |
|---|---|---|
| Logistic regression (baseline) | 0.761 | 0.418 |
| Gradient boosting (baseline) | 0.787 | 0.442 |
| **WoE scorecard** (6 features, no AGE/EDUCATION) | 0.768 | 0.424 |
| **Final gradient boosting** (no AGE/EDUCATION) | **0.789** | **0.444** |

Excluding age and education cost no measurable accuracy.

### Scorecard
Selected features (after IV filtering, a correlation limit, and removal of any feature with a counter-intuitive coefficient sign): `PAY_1`, `PAY_3`, `PAY_6`, `LIMIT_BAL`, `AVG_PAY_AMT`, `UTILIZATION`. Scores are scaled so that 600 points = 50:1 good-to-bad odds, and +20 points doubles the odds.

Default rate falls from **70.7%** in the lowest score band to **6.7%** in the highest, so the score ranks risk as intended. The very high information value of `PAY_1` (0.91) is expected: it is last month's repayment status, observed before the month being predicted, and is the strongest default signal in credit data, not leakage.

### Explainability
SHAP values explain the final model. For each applicant, the top factors pushing toward default are grouped into plain categories (payment delays, credit utilisation, repayment amounts, credit limit) so reason codes read like real adverse-action reasons.

<!-- Add screenshots: outputs/shap_beeswarm.png, outputs/score_bands.png, outputs/roc_curves.png -->

---

## 💰 Choosing the approval cut-off

The dataset has no profit data, so the cut-off is chosen under a simple, **assumed** cost model: approving a customer who repays earns +1 unit; approving one who defaults loses *L* units; declining earns 0. The cut-off is picked on validation data and evaluated on the test set.

| Assumed loss ratio | Cut-off | Approval rate | Default rate among approved | Profit / applicant | If everyone were approved |
|---|---|---|---|---|---|
| 3 : 1 | 0.25 | 73.5% | 11.7% | +0.39 | +0.12 |
| **5 : 1** | **0.14** | **50.0%** | **8.9%** | **+0.23** | **-0.33** |
| 10 : 1 | 0.09 | 25.4% | 6.1% | +0.08 | -1.43 |

With a 5:1 loss ratio, the model turns a loss-making book (approve everyone) into a profitable one, approving half of applicants at a default rate of 8.9% versus 22.1% overall. These are illustrative figures from an assumption, not real bank economics.

---

## ⚖️ Fairness audit

Sex, marital status and age were not used by the model; the audit checks outcomes by group at the 5:1 cut-off.

| Group | n | Approval rate | Ratio vs. best group |
|---|---|---|---|
| Male | 2,372 | 47.5% | 0.92 |
| Female | 3,628 | 51.6% | 1.00 |
| Single | 3,196 | 49.9% | 0.99 |
| Married | 2,731 | 50.4% | 1.00 |
| Other (marital) | 73 | 37.0% | 0.73 (very small group) |
| Age < 30 | 1,926 | 45.7% | 0.83 |
| Age 30-39 | 2,206 | 54.9% | 1.00 |
| Age 40-49 | 1,296 | 50.2% | 0.91 |
| Age 50+ | 572 | 45.3% | 0.83 |

- All groups clear the common 0.8 screening line except "Other" marital status, which has only 73 test customers, so that result is noise.
- Approval gaps largely follow real default-rate differences (for example, 24.1% for men vs. 20.8% for women). Predicted default rates are close to actual rates in every group; the largest gap is under-30s (23.6% predicted vs. 21.9% actual), meaning the model slightly over-penalises that group.
- Leaving sensitive variables out does not guarantee fairness, since other features can act as proxies, which is why the audit looks at outcomes.

---

## 🤖 Explanation agent

`step5_agent.py` is a tool-using LLM agent:

1. The LLM decides to call the `score_applicant` tool, which runs the model and SHAP and returns the decision, probability and grouped risk factors.
2. It writes a short explanation using only those facts.
3. A **verifier** checks the text: correct decision, only numbers that came from the tool, at least one real risk factor named, and no mention of personal characteristics. A failing draft gets one retry, then falls back to a deterministic template, so unverified text is never shown.

It works with Anthropic or any OpenAI-compatible API, or with no API key at all in template mode. It has been tested with `openai/gpt-oss-120b` through Groq.

**Example, written by the LLM (borderline decline):**

> **Decision: DECLINE.** Default probability: **15.7%**, which exceeds the approval cut-off of **14%** (borderline = true, so manual review is recommended).
>
> **Main risk factors**
> - **Repayment amounts**: pays only about 3% of the average bill each month.
> - **Credit limit**: limited to NT$ 50,000.
>
> **Favourable factor**
> - **Payment delays**: no late payments in the last 6 months.

**Example, template mode (no API key, clear decline):**

> **Decision: DECLINE.** The model estimates a 54.6% chance of default, above the 14.0% approval cut-off. Main risk factors: Payment delays (most recent month overdue by 2 month(s); paid late in 5 of the last 6 months, longest delay 2 month(s)); Repayment amounts (pays about 33% of the average bill each month). In the applicant's favour: Credit utilisation (almost no outstanding balance).

**What the verifier does not do:** it checks the decision, numbers, factor names and mentions of personal characteristics, but not qualitative wording, so an LLM draft can still add small embellishments. The prompt asks it not to, but explanations are decision-support text, not legal adverse-action notices. This is a tool-calling agent with a verification guardrail, not an autonomous multi-step planner.

---

## 📁 Project Structure

```
├── common.py                       # Shared data loading, cleaning, features, split
├── step1_baseline.py               # Baseline models
├── step2_scorecard.py              # WoE/IV scorecard
├── step3_shap.py                   # Final model + SHAP + reason codes
├── step4_threshold_fairness.py     # Cut-off analysis + fairness audit
├── step5_agent.py                  # Verified explanation agent
├── streamlit_app.py                # Demo app
├── example_applicant.json          # Sample input for the agent
├── requirements.txt
├── outputs/                        # Metrics tables and charts (generated)
└── model/                          # Saved model (generated, not committed)
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

# place the dataset file (UCI_Credit_Card.csv or "default of credit card clients.xls") in this folder
python step1_baseline.py
python step2_scorecard.py
python step3_shap.py
python step4_threshold_fairness.py

python step5_agent.py --examples --provider template      # no API key needed
streamlit run streamlit_app.py
```

Optional LLM modes (set the variables in the same terminal you run the commands from, and never commit keys):

```powershell
# Anthropic
$env:ANTHROPIC_API_KEY="your_key"

# or any OpenAI-compatible API, e.g. Groq
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_BASE_URL="https://api.groq.com/openai/v1"
$env:AGENT_MODEL="openai/gpt-oss-120b"   # list the models your key can use with: OpenAI().models.list()

python step5_agent.py --examples
```

---

## ⚠️ Limitations

- Single 2005 Taiwan snapshot; results may not transfer to other markets or periods.
- Only approved customers' outcomes are observed (no reject inference), as in most public credit datasets.
- The profit analysis rests on an assumed loss ratio, not real economics.
- The fairness audit is a screening analysis on a few attributes; some groups are small.
- The agent's explanations are limited to the model's top grouped factors, can contain minor LLM wording embellishments the verifier does not check, and are not legal adverse-action notices.

---

## 🛠️ Tech Stack

Python, pandas, scikit-learn, SHAP, matplotlib, Streamlit, LLM APIs (Anthropic or OpenAI-compatible such as Groq, optional)

---

## 👩‍💻 Author

**Baratam Amritha**
B.Tech Computer Science Engineering, VIT Chennai

- GitHub: [@amrithab07](https://github.com/amrithab07)
- LinkedIn: [amritha-baratam](https://www.linkedin.com/in/amritha-baratam)
