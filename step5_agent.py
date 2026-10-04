"""
Step 5: Tool-using LLM agent that explains credit decisions, with a verification guardrail.

Flow:  LLM  --calls-->  score_applicant / get_policy (your model + SHAP)  --facts-->  LLM writes explanation
       -> verifier checks the text against the tool facts -> one retry -> deterministic template fallback.

Needs: common.py, model/credit_model.joblib (step 3 + 4), dataset file, shap.
Optional LLM providers (pick ONE, otherwise the template mode is used - no key needed):
  Anthropic:            pip install anthropic     set ANTHROPIC_API_KEY=...        (optional AGENT_MODEL)
  OpenAI-compatible:    pip install openai        set OPENAI_API_KEY=...  OPENAI_BASE_URL=...  AGENT_MODEL=...
                        (e.g. Groq: OPENAI_BASE_URL=https://api.groq.com/openai/v1  AGENT_MODEL=llama-3.3-70b-versatile)
Run examples:
  python step5_agent.py --examples                 (3 cases -> outputs/agent_examples.md)
  python step5_agent.py --index 12997              (one applicant from the test set)
  python step5_agent.py --json example_applicant.json
  add  --provider template|anthropic|openai|auto   (default auto)
"""
import os, re, json, argparse
import numpy as np
import pandas as pd
import joblib
import shap
from common import prepare, add_features

BUNDLE = joblib.load("model/credit_model.joblib")
MODEL, FEATURES = BUNDLE["model"], BUNDLE["features"]
CUTOFF = float(BUNDLE.get("threshold_profit", BUNDLE["threshold"]))
LOSS_RATIO = BUNDLE.get("loss_ratio")
APPLICANTS = {}          # applicant_id (str) -> row of engineered features
_EXPLAINER = None

# ------------------------------------------------------------------ tools
GROUPS = {
    "Payment delays": [f"PAY_{i}" for i in range(1, 7)] + ["MONTHS_LATE", "MAX_DELAY"],
    "Credit utilisation": ["UTILIZATION", "AVG_BILL"] + [f"BILL_AMT{i}" for i in range(1, 7)],
    "Repayment amounts": ["AVG_PAY_AMT", "PAY_RATIO"] + [f"PAY_AMT{i}" for i in range(1, 7)],
    "Credit limit": ["LIMIT_BAL"],
}

def _shap_row(x):
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = shap.TreeExplainer(MODEL)
    sv = _EXPLAINER.shap_values(x)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv[0]

def _detail(group, r):
    if group == "Payment delays":
        if int(r["MONTHS_LATE"]) == 0:
            return "no late payments in the last 6 months"
        s = f"paid late in {int(r['MONTHS_LATE'])} of the last 6 months, longest delay {int(r['MAX_DELAY'])} month(s)"
        return (f"most recent month overdue by {int(r['PAY_1'])} month(s); " + s) if r["PAY_1"] > 0 else s
    if group == "Credit utilisation":
        if r["UTILIZATION"] < 0.01:
            return "almost no outstanding balance"
        return f"average bill is {r['UTILIZATION'] * 100:.0f}% of the credit limit"
    if group == "Repayment amounts":
        return f"pays about {r['PAY_RATIO'] * 100:.0f}% of the average bill each month"
    return f"credit limit is NT$ {r['LIMIT_BAL']:,.0f}"

def tool_score_applicant(applicant_id):
    key = str(applicant_id)
    if key not in APPLICANTS:
        return {"error": f"unknown applicant_id {applicant_id}"}
    r = APPLICANTS[key]
    x = r[FEATURES].astype(float).to_frame().T
    p = float(MODEL.predict_proba(x)[0, 1])
    sv = _shap_row(x)
    gs = {g: float(sum(sv[FEATURES.index(f)] for f in fs if f in FEATURES)) for g, fs in GROUPS.items()}
    risk = [g for g, v in sorted(gs.items(), key=lambda kv: -kv[1]) if v > 0.02][:3]
    prot = [g for g, v in sorted(gs.items(), key=lambda kv: kv[1]) if v < -0.02][:2]
    return {"applicant_id": key, "default_probability": round(p, 3), "approval_cutoff": round(CUTOFF, 3),
            "decision": "DECLINE" if p >= CUTOFF else "APPROVE",
            "borderline": bool(abs(p - CUTOFF) <= 0.05),
            "risk_factors": [{"factor": g, "detail": _detail(g, r)} for g in risk],
            "favourable_factors": [{"factor": g, "detail": _detail(g, r)} for g in prot]}

def tool_get_policy():
    return {"approval_rule": f"approve if predicted default probability is below {CUTOFF:.2f}",
            "cutoff_basis": f"maximises expected profit under an assumed loss ratio of {LOSS_RATIO}:1 (illustrative assumption)",
            "model": "gradient boosting trained on 30,000 credit card customers (public dataset)",
            "fairness": "protected characteristics are not used by the model",
            "note": "model output is a screening aid; borderline cases need human review"}

TOOL_FUNCS = {"score_applicant": tool_score_applicant, "get_policy": tool_get_policy}
TOOLS = [
    {"name": "score_applicant",
     "description": "Score one applicant with the credit-risk model. Returns default probability, decision, "
                    "and the main risk and favourable factors computed from SHAP values.",
     "parameters": {"type": "object", "properties": {"applicant_id": {"type": "string"}}, "required": ["applicant_id"]}},
    {"name": "get_policy",
     "description": "Return the approval policy and how the cut-off was chosen.",
     "parameters": {"type": "object", "properties": {}}},
]
SYSTEM = (
    "You explain credit-card approval decisions to a loan officer. First call score_applicant for the applicant "
    "(call get_policy only if you need the policy). Then write at most 120 words: the decision, the default "
    "probability versus the approval cut-off, the main risk factors, and one favourable factor if any. "
    "Use ONLY facts returned by the tools. Never invent numbers or factors. Never mention or speculate about "
    "gender, age, marital status, education or any other personal characteristic. If 'borderline' is true, "
    "recommend manual review; if it is false, say nothing about manual review. Do not add claims or "
    "adjectives beyond the tool results (for example avoid words like 'consistently' or 'comfortably')."
)

# ------------------------------------------------------------------ verifier + template
PROTECTED = re.compile(r"\b(gender|sex|male|female|married|marital|age|aged|race|religion|education|nationality)\b")

def verify(text, f):
    problems, low = [], (text or "").lower()
    if len(low.strip()) < 20:
        return ["answer is empty"]
    if f["decision"] == "DECLINE" and "declin" not in low:
        problems.append("state the decision DECLINE")
    if f["decision"] == "APPROVE" and "approv" not in low:
        problems.append("state the decision APPROVE")
    if PROTECTED.search(low):
        problems.append("do not mention personal characteristics")
    allowed = [f["default_probability"] * 100, f["approval_cutoff"] * 100]
    for item in f["risk_factors"] + f["favourable_factors"]:
        allowed += [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", item["detail"])]
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        if not any(abs(float(m.group(1)) - a) <= 0.6 for a in allowed):
            problems.append(f"the number {m.group(0)} is not in the tool results")
    if f["decision"] == "DECLINE" and f["risk_factors"] and not any(x["factor"].lower() in low for x in f["risk_factors"]):
        problems.append("name at least one risk factor exactly as returned by the tool")
    return problems

def template_explanation(f):
    p, c = f["default_probability"] * 100, f["approval_cutoff"] * 100
    side = "above" if f["decision"] == "DECLINE" else "below"
    out = [f"Decision: {f['decision']}. The model estimates a {p:.1f}% chance of default, {side} the {c:.1f}% approval cut-off."]
    if f["risk_factors"]:
        out.append("Main risk factors: " + "; ".join(f"{x['factor']} ({x['detail']})" for x in f["risk_factors"]) + ".")
    if f["favourable_factors"]:
        out.append("In the applicant's favour: " + "; ".join(f"{x['factor']} ({x['detail']})" for x in f["favourable_factors"]) + ".")
    if f["borderline"]:
        out.append("This case is close to the cut-off, so manual review is recommended.")
    return " ".join(out)

# ------------------------------------------------------------------ LLM adapters (neutral conversation format)
class AnthropicLLM:
    name = "anthropic"
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")
    def step(self, system, conv, tools):
        msgs = []
        for m in conv:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                blocks = ([{"type": "text", "text": m["text"]}] if m.get("text") else []) + \
                         [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in m.get("tool_calls") or []]
                if blocks:
                    msgs.append({"role": "assistant", "content": blocks})
            else:
                blk = {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(blk)
                else:
                    msgs.append({"role": "user", "content": [blk]})
        r = self.client.messages.create(model=self.model, max_tokens=600, system=system, messages=msgs,
                                        tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools])
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "args": b.input} for b in r.content if b.type == "tool_use"]
        return text, calls

class OpenAICompatLLM:
    name = "openai-compatible"
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        self.model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    def step(self, system, conv, tools):
        msgs = [{"role": "system", "content": system}]
        for m in conv:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in m["tool_calls"]]
                msgs.append(d)
            else:
                msgs.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
        r = self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=0.2, max_tokens=2000,
            tools=[{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools])
        msg = r.choices[0].message
        calls = [{"id": tc.id, "name": tc.function.name, "args": json.loads(tc.function.arguments or "{}")} for tc in (msg.tool_calls or [])]
        return msg.content or "", calls

def pick_llm(provider):
    if provider == "template":
        return None
    if provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicLLM()
    if provider == "openai" or (provider == "auto" and os.environ.get("OPENAI_API_KEY")):
        return OpenAICompatLLM()
    return None

# ------------------------------------------------------------------ the agent loop
def run_agent(llm, applicant_id, max_steps=4):
    facts, tool_log = None, []
    if llm is None:                                   # template mode: no LLM, same tool, same facts
        facts = tool_score_applicant(applicant_id)
        if "error" in facts:
            return {"text": facts["error"], "source": "error", "verified": False, "tools": [], "facts": facts, "problems": []}
        return {"text": template_explanation(facts), "source": "template", "verified": True,
                "tools": ["score_applicant"], "facts": facts, "problems": []}

    conv = [{"role": "user", "text": f"Explain the credit decision for applicant {applicant_id} to a loan officer."}]
    text = ""
    for _ in range(max_steps):
        text, calls = llm.step(SYSTEM, conv, TOOLS)
        conv.append({"role": "assistant", "text": text, "tool_calls": calls})
        if not calls:
            break
        for c in calls:
            try:
                result = TOOL_FUNCS[c["name"]](**c["args"])
            except Exception as e:                    # bad tool name / arguments are returned to the model
                result = {"error": str(e)}
            if c["name"] == "score_applicant" and "error" not in result:
                facts = result
            tool_log.append(c["name"])
            conv.append({"role": "tool", "id": c["id"], "name": c["name"], "content": json.dumps(result)})

    if facts is None:                                 # the model never scored the applicant -> do it for it
        facts = tool_score_applicant(applicant_id)
        tool_log.append("score_applicant (forced)")
    problems = verify(text, facts)
    if problems:                                      # one retry with the verifier's feedback
        conv.append({"role": "user", "text": "Your answer failed checks: " + "; ".join(problems) +
                     ". Rewrite it using only the tool results."})
        text, _ = llm.step(SYSTEM, conv, TOOLS)
        problems = verify(text, facts)
    if problems:                                      # still failing -> never show unverified text
        return {"text": template_explanation(facts), "source": "template (LLM output failed verification)",
                "verified": True, "tools": tool_log, "facts": facts, "problems": problems}
    return {"text": text.strip(), "source": llm.name, "verified": True, "tools": tool_log, "facts": facts, "problems": []}

# ------------------------------------------------------------------ CLI
def load_test_applicants():
    X_te = prepare()[2]
    for i, row in X_te.iterrows():
        APPLICANTS[str(i)] = row
    return X_te

def pick_examples(X_te):
    p = MODEL.predict_proba(X_te[FEATURES])[:, 1]
    ids = X_te.index.to_numpy()
    def first(mask):
        hit = ids[mask]
        return str(hit[0]) if len(hit) else None
    chosen = [first(p >= max(CUTOFF + 0.25, 0.5)), first(np.abs(p - CUTOFF) <= 0.03), first(p <= CUTOFF / 2)]
    return [c for c in chosen if c]

def show(applicant_id, res, md=None):
    f = res["facts"]
    block = (f"=== Applicant {applicant_id} | model p(default) = {f.get('default_probability')} | "
             f"cut-off = {f.get('approval_cutoff')} | decision = {f.get('decision')} ===\n"
             f"tools called: {res['tools']} | source: {res['source']} | verified: {res['verified']}\n"
             + (f"verifier problems: {res['problems']}\n" if res["problems"] else "") + f"\n{res['text']}\n")
    print(block)
    if md is not None:
        md.append(block.replace("===", "###") + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index"); ap.add_argument("--json"); ap.add_argument("--examples", action="store_true")
    ap.add_argument("--provider", default="auto", choices=["auto", "template", "anthropic", "openai"])
    a = ap.parse_args()
    X_te = load_test_applicants()
    llm = pick_llm(a.provider)
    print("LLM:", llm.name + " / " + llm.model if llm else "none (template mode)", "\n")
    if a.json:
        raw = pd.DataFrame([json.load(open(a.json))])
        APPLICANTS["custom"] = add_features(raw).iloc[0]
        show("custom", run_agent(llm, "custom"))
    elif a.index:
        show(a.index, run_agent(llm, a.index))
    else:
        md = ["# Agent examples\n"]
        for aid in pick_examples(X_te):
            show(aid, run_agent(llm, aid), md)
        os.makedirs("outputs", exist_ok=True)
        open("outputs/agent_examples.md", "w", encoding="utf-8").write("\n".join(md))
        print("Saved outputs/agent_examples.md")
