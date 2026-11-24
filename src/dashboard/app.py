"""Streamlit review screen for loan officers.

Reads the artifacts written by `python -m src.main`:
    models/latest/{model,preprocessor}.joblib, reports/fairness_report.json, logs/audit.jsonl

    streamlit run src/dashboard/app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.data_loader import generate_synthetic_credit_data
from src.explainability.shap_explainer import SHAPExplainer
from src.models.credit_model import CreditScoringModel

MODEL_DIR = Path("models/latest")
REPORT = Path("reports/fairness_report.json")
AUDIT_LOG = Path("logs/audit.jsonl")

st.set_page_config(page_title="XAI Credit Scoring", layout="wide")


@st.cache_resource
def load():
    model = CreditScoringModel.load(str(MODEL_DIR / "model.joblib"))
    pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
    explainer = SHAPExplainer(model=model.model, feature_names=model.feature_names)
    explainer.fit(None)
    return model, pre, explainer


@st.cache_data
def applicants(n: int = 40) -> pd.DataFrame:
    # fresh synthetic applicants (different seed from training)
    return generate_synthetic_credit_data(n_samples=n, random_state=2024).drop(columns=["credit_risk"])


if not (MODEL_DIR / "model.joblib").exists():
    st.error("No trained model found. Run `python -m src.main` first.")
    st.stop()

model, pre, explainer = load()
page = st.sidebar.radio("Page", ["Review queue", "Fairness audit", "Decision log"])

if page == "Review queue":
    df = applicants()
    X = pre.transform(df)
    df = df.assign(probability=model.predict_proba(X))
    df["decision"] = pd.cut(df["probability"], [-0.01, model.deny_threshold, model.approve_threshold, 1.01], labels=["DENIED", "REVIEW", "APPROVED"])

    st.title("Review queue")
    c1, c2, c3 = st.columns(3)
    c1.metric("Needs review", int((df["decision"] == "REVIEW").sum()))
    c2.metric("Auto-approved", int((df["decision"] == "APPROVED").sum()))
    c3.metric("Auto-denied", int((df["decision"] == "DENIED").sum()))

    only_review = st.checkbox("Show only REVIEW cases", value=True)
    view = df[df["decision"] == "REVIEW"] if only_review else df
    for idx, row in view.iterrows():
        with st.expander(f"#{idx}  p={row['probability']:.2f}  {row['decision']}"):
            left, right = st.columns([1, 2])
            left.write(row[["age", "income", "loan_amount", "loan_duration_months", "employment_status", "credit_history_status", "housing_type"]].to_frame("value"))
            exp = explainer.explain_single(X.loc[idx])
            top = pd.DataFrame(exp["top_contributions"]).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["shap_value"], y=top["feature"], orientation="h", marker_color=["#2a9d8f" if v > 0 else "#e76f51" for v in top["shap_value"]]))
            fig.update_layout(title="Top SHAP contributions (log-odds)", height=280, margin=dict(l=10, r=10, t=40, b=10))
            right.plotly_chart(fig, use_container_width=True)
            right.text(exp["explanation"])

elif page == "Fairness audit":
    st.title("Fairness audit (test set)")
    if not REPORT.exists():
        st.warning("No report yet - run `python -m src.main`.")
        st.stop()
    rep = json.loads(REPORT.read_text())
    st.metric("Gate", "PASS" if rep["passes"] else "FAIL")
    st.write({k: round(v, 3) for k, v in rep["test_metrics"].items()})
    m = pd.DataFrame(rep["metrics"])[["protected_attribute", "group_a", "group_b", "name", "value", "threshold", "passed", "severity", "note"]]
    st.dataframe(m.style.apply(lambda r: ["background-color: #fde2e1" if not r["passed"] else ""] * len(r), axis=1), use_container_width=True)

else:
    st.title("Decision log")
    if not AUDIT_LOG.exists():
        st.info("No decisions logged yet - call the API's /predict endpoint.")
    else:
        rows = [json.loads(line) for line in AUDIT_LOG.read_text().splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(rows).iloc[::-1], use_container_width=True)
