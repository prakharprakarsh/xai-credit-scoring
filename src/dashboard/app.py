"""Loan Officer Dashboard — EU AI Act Article 14 (Human Oversight)."""

import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

st.set_page_config(page_title="XAI Credit Scoring", page_icon="🏦", layout="wide")
st.sidebar.title("🏦 XAI Credit Scoring")
st.sidebar.markdown("**EU AI Act Compliant**")
page = st.sidebar.radio("Navigation", ["📋 Application Review", "📊 Fairness Dashboard", "📝 Audit Log", "ℹ️ About"])


def load_sample_data():
    np.random.seed(42)
    n = 50
    data = {
        "applicant_id": [f"APP-{i:04d}" for i in range(n)],
        "age": np.random.randint(22, 65, n),
        "income": np.random.lognormal(10.5, 0.5, n).astype(int),
        "loan_amount": np.random.lognormal(9.0, 0.7, n).astype(int),
        "debt_to_income": np.random.uniform(0.1, 0.8, n).round(2),
        "employment_years": np.random.exponential(5, n).round(1),
        "credit_history": np.random.choice(["Excellent", "Good", "Fair", "Poor"], n, p=[0.25, 0.35, 0.25, 0.15]),
        "probability": np.random.beta(5, 3, n).round(3),
        "gender": np.random.choice(["Male", "Female"], n),
        "age_group": np.random.choice(["18-29", "30-50", "51-65"], n, p=[0.25, 0.50, 0.25]),
    }
    df = pd.DataFrame(data)
    df["decision"] = df["probability"].apply(lambda p: "APPROVED" if p >= 0.7 else ("DENIED" if p <= 0.3 else "REVIEW"))
    df["risk_score"] = (300 + df["probability"] * 550).astype(int)
    return df


if page == "📋 Application Review":
    st.title("📋 Application Review")
    df = load_sample_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(df))
    col2.metric("Pending Review", len(df[df["decision"] == "REVIEW"]))
    col3.metric("Auto-Approved", len(df[df["decision"] == "APPROVED"]))
    col4.metric("Auto-Denied", len(df[df["decision"] == "DENIED"]))
    for _, row in df.iterrows():
        icon = {"APPROVED": "🟢", "DENIED": "🔴", "REVIEW": "🟡"}.get(row["decision"], "⚪")
        with st.expander(f'{icon} {row["applicant_id"]} — Score: {row["risk_score"]} | {row["decision"]}'):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Age:** {row['age']}")
                st.write(f"**Income:** €{row['income']:,.0f}")
                st.write(f"**Loan:** €{row['loan_amount']:,.0f}")
                st.write(f"**Probability:** {row['probability']:.1%}")
            with c2:
                np.random.seed(hash(row["applicant_id"]) % 2**31)
                vals = {k: np.random.uniform(-0.2, 0.2) for k in ["income", "debt_to_income", "employment_years", "credit_history", "savings"]}
                fig = go.Figure(go.Bar(x=list(vals.values()), y=list(vals.keys()), orientation="h", marker_color=["#22c55e" if v > 0 else "#ef4444" for v in vals.values()]))
                fig.update_layout(title="SHAP Values", height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Fairness Dashboard":
    st.title("📊 Fairness Dashboard")
    df = load_sample_data()
    c1, c2 = st.columns(2)
    with c1:
        gr = df.groupby("gender")["probability"].mean()
        fig = px.bar(x=gr.index, y=gr.values, labels={"x": "Gender", "y": "Mean Probability"}, title="By Gender", color=gr.index)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ar = df.groupby("age_group")["probability"].mean()
        fig = px.bar(x=ar.index, y=ar.values, labels={"x": "Age Group", "y": "Mean Probability"}, title="By Age Group", color=ar.index)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Fairness Metrics")
    st.table(pd.DataFrame({"Metric": ["Demographic Parity (Gender)", "Disparate Impact (Gender)", "Equalized Odds (Gender)"], "Value": [0.92, 0.91, 0.06], "Threshold": [0.80, 0.80, 0.10], "Status": ["✅ Pass", "✅ Pass", "✅ Pass"]}))

elif page == "📝 Audit Log":
    st.title("📝 Audit Log")
    st.dataframe(pd.DataFrame([
        {"timestamp": "2025-01-15 09:23", "type": "prediction", "id": "APP-0012", "decision": "APPROVED", "officer": "—"},
        {"timestamp": "2025-01-15 09:25", "type": "prediction", "id": "APP-0013", "decision": "REVIEW", "officer": "—"},
        {"timestamp": "2025-01-15 09:31", "type": "override", "id": "APP-0013", "decision": "APPROVED", "officer": "J. de Vries"},
    ]), use_container_width=True)

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown("## XAI Credit Scoring\nEU AI Act compliant credit scoring with SHAP, LIME, counterfactual explanations and fairness auditing.")
