#  XAI Credit Scoring — Explainable & Fair Credit Risk Assessment

> **EU AI Act Compliant** · Fairness Auditing · SHAP/LIME Explainability · Bias Mitigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Compliant-green.svg)](#eu-ai-act-compliance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

---

## 📋 Table of Contents

- [Overview](#overview)
- [EU AI Act Compliance](#eu-ai-act-compliance)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Features](#features)
- [Fairness Auditing](#fairness-auditing)
- [Explainability](#explainability)
- [API & Dashboard](#api--dashboard)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## Overview

This project implements a **transparent, auditable, and fair** credit scoring system designed for deployment in EU-regulated environments. It goes beyond standard ML pipelines by embedding **explainability** (SHAP, LIME, counterfactual explanations) and **fairness** (demographic parity, equalized odds, disparate impact analysis) directly into the model lifecycle.

**Why this matters for the Netherlands & EU:**
- The [EU AI Act](https://artificialintelligenceact.eu/) classifies credit scoring as a **high-risk AI system** (Annex III, Section 5b)
- Dutch financial institutions (under DNB/AFM supervision) must demonstrate algorithmic transparency
- This project provides a reference implementation for compliance with Articles 9, 13, 14, and 15

### Key Capabilities

- **Model Training**: Gradient Boosted Trees (XGBoost/LightGBM) with hyperparameter tuning
- **Global Explainability**: SHAP summary plots, feature importance, partial dependence
- **Local Explainability**: Per-applicant SHAP waterfall, LIME explanations, counterfactual generation
- **Fairness Auditing**: Demographic parity, equalized odds, disparate impact ratio, intersectional analysis
- **Bias Mitigation**: Pre-processing (reweighing), in-processing (fairness constraints), post-processing (threshold tuning)
- **Human-in-the-Loop**: Dashboard for loan officers to review AI decisions with explanations
- **Audit Trail**: Full logging of model decisions, explanations, and override actions
- **API**: FastAPI service with explanation endpoints for production integration

---

## EU AI Act Compliance

This project is designed as a reference implementation addressing key requirements of the EU AI Act for high-risk AI systems:

| EU AI Act Article | Requirement | Implementation |
|---|---|---|
| **Art. 9** — Risk Management | Continuous risk identification and mitigation | `src/fairness/bias_auditor.py` — automated bias detection pipeline runs on every model update |
| **Art. 10** — Data Governance | Training data quality, bias examination | `src/data/data_validator.py` — schema validation, missing value analysis, proxy variable detection |
| **Art. 13** — Transparency | Users must understand AI output | `src/explainability/` — SHAP, LIME, counterfactual explanations for every decision |
| **Art. 14** — Human Oversight | Effective human oversight mechanisms | `src/dashboard/` — loan officer review interface with override capability |
| **Art. 15** — Accuracy & Robustness | Appropriate accuracy levels, resilience | `tests/` — adversarial robustness tests, performance monitoring across subgroups |
| **Art. 17** — Quality Management | Documentation of the AI system | `docs/model_card.md` — standardized model documentation |
| **Art. 61** — Post-market Monitoring | Ongoing performance monitoring | `src/fairness/monitoring.py` — drift detection, fairness metric tracking over time |

### Compliance Checklist

- [x] Risk classification documented (High-Risk — Annex III, 5b)
- [x] Technical documentation / Model Card
- [x] Data governance procedures with bias examination
- [x] Transparency: explanations provided for each individual decision
- [x] Human oversight interface with meaningful override capability
- [x] Fairness metrics computed across protected attributes
- [x] Bias mitigation strategies implemented and documented
- [x] Logging and audit trail for all decisions
- [x] Robustness testing including adversarial scenarios
- [x] Post-deployment monitoring framework

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Raw Data → Validation → Preprocessing → Feature Engineering    │
│                    ↓ Proxy Detection                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      MODEL LAYER                                │
│  XGBoost/LightGBM → Hyperparameter Tuning → Cross-Validation   │
│                    ↓ Fairness Constraints                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  EXPLAINABILITY LAYER                            │
│  SHAP (Global+Local) │ LIME │ Counterfactuals │ Feature Import. │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    FAIRNESS LAYER                                │
│  Demographic Parity │ Equalized Odds │ Disparate Impact │ Audit │
│              ↓ Bias Mitigation (Pre/In/Post)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   SERVING LAYER                                  │
│  FastAPI Endpoints │ Loan Officer Dashboard │ Audit Logging      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xai-credit-scoring.git
cd xai-credit-scoring

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.main

# Launch the dashboard
python -m src.dashboard.app

# Start the API server
uvicorn src.api.server:app --reload
```

### Quick Demo

```python
from src.models.credit_model import CreditScoringModel
from src.explainability.shap_explainer import SHAPExplainer
from src.fairness.bias_auditor import BiasAuditor

# Train model
model = CreditScoringModel()
model.train("data/processed/train.csv")

# Get prediction with explanation
result = model.predict_with_explanation(applicant_data)
print(result.decision)        # "APPROVED" / "DENIED" / "REVIEW"
print(result.probability)     # 0.82
print(result.explanation)     # Human-readable explanation
print(result.shap_values)     # Per-feature SHAP contributions

# Run fairness audit
auditor = BiasAuditor(model, test_data)
report = auditor.full_audit(protected_attributes=["gender", "age_group", "ethnicity"])
report.save("reports/fairness_audit.html")
```

---

## Project Structure

```
xai-credit-scoring/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── model_config.yaml
│   └── fairness_config.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── preprocessor.py
│   │   └── proxy_detector.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── credit_model.py
│   │   └── hyperparameter_tuner.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── counterfactual.py
│   │   └── explanation_generator.py
│   ├── fairness/
│   │   ├── __init__.py
│   │   ├── bias_auditor.py
│   │   ├── bias_mitigator.py
│   │   ├── monitoring.py
│   │   └── report_generator.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py
│   └── dashboard/
│       ├── __init__.py
│       └── app.py
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_fairness.py
│   ├── test_explainability.py
│   ├── test_api.py
│   └── test_robustness.py
├── docs/
│   ├── model_card.md
│   └── eu_ai_act_mapping.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Fairness Auditing

The fairness module computes metrics across all protected attributes defined in `configs/fairness_config.yaml`:

```python
from src.fairness.bias_auditor import BiasAuditor

auditor = BiasAuditor(
    model=trained_model,
    test_data=test_df,
    protected_attributes=["gender", "age_group", "ethnicity"],
    favorable_outcome=1,
    reference_groups={"gender": "male", "age_group": "30-50", "ethnicity": "dutch"}
)

report = auditor.full_audit()
assert report.passes_all_thresholds(), f"Fairness violations: {report.violations}"
```

### Fairness Metrics Computed

| Metric | Formula | Threshold |
|---|---|---|
| Demographic Parity Ratio | P(Ŷ=1\|G=a) / P(Ŷ=1\|G=b) | ≥ 0.80 |
| Equalized Odds Difference | \|TPR_a - TPR_b\| + \|FPR_a - FPR_b\| | ≤ 0.10 |
| Disparate Impact Ratio | Selection Rate (unprivileged) / Selection Rate (privileged) | ≥ 0.80 |
| Predictive Parity Difference | \|PPV_a - PPV_b\| | ≤ 0.10 |
| Calibration Difference | \|E[Y\|Ŷ=p, G=a] - E[Y\|Ŷ=p, G=b]\| | ≤ 0.05 |

---

## Explainability

### Global Explanations

```python
from src.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model=trained_model)
explainer.plot_summary(test_data)
explainer.plot_feature_importance(test_data)
```

### Local Explanations (Per-Applicant)

```python
explanation = explainer.explain_single(applicant_row)
print(explanation.natural_language)
# "Your application was DENIED primarily because:
#  - Your debt-to-income ratio (0.65) is significantly above average (0.35)
#  - Your credit history length (2 years) is shorter than typical approved applicants
#  To improve: reducing your debt-to-income ratio below 0.45 would most impact your score."
```

---

## API & Dashboard

### API Endpoints

```
POST /predict              — Get credit decision + explanation
POST /predict/batch        — Batch predictions
GET  /explain/{id}         — Detailed explanation for a decision
GET  /fairness/report      — Latest fairness audit report
POST /override             — Loan officer override with reason logging
GET  /audit/log            — Decision audit trail
GET  /health               — Health check
```

### Dashboard

The Streamlit dashboard provides loan officers with:
- Individual applicant review with SHAP waterfall visualizations
- Approve/Deny/Escalate workflow with mandatory explanation review
- Fairness dashboard with real-time metrics
- Audit log viewer

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only fairness tests (CI/CD gate)
pytest tests/test_fairness.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### CI/CD Fairness Gate

The GitHub Actions workflow includes a **fairness gate** that blocks deployment if any fairness metric falls below threshold. See `.github/workflows/ci.yml`.

---

## Deployment

### Docker

```bash
docker build -t xai-credit-scoring .
docker run -p 8000:8000 -p 8501:8501 xai-credit-scoring
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure all tests pass, including fairness gates
4. Submit a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [SHAP](https://github.com/slundberg/shap) — Lundberg & Lee
- [AIF360](https://github.com/Trusted-AI/AIF360) — IBM Fairness 360
- [EU AI Act](https://artificialintelligenceact.eu/) — European Commission
