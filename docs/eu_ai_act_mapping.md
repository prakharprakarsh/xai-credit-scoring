# EU AI Act Compliance Mapping

## Risk Classification
Credit scoring = **High-Risk AI System** (Annex III, Section 5b)

## Article-by-Article Implementation

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 9 | Risk Management | `src/fairness/bias_auditor.py` |
| Art. 10 | Data Governance | `src/data/data_validator.py`, `src/data/proxy_detector.py` |
| Art. 11 | Technical Documentation | `docs/model_card.md`, this document |
| Art. 12 | Record-keeping | `src/api/server.py` audit endpoints |
| Art. 13 | Transparency | `src/explainability/` (SHAP, LIME, counterfactual) |
| Art. 14 | Human Oversight | `src/dashboard/app.py` with override capability |
| Art. 15 | Accuracy & Robustness | `tests/test_robustness.py` |
| Art. 17 | Quality Management | `.github/workflows/ci.yml` with fairness gates |
| Art. 61 | Post-market Monitoring | `src/fairness/monitoring.py` |

## Dutch Regulatory Context
- **DNB:** Algorithmic accountability aligned
- **AFM:** Fair customer treatment (Klantbelang Centraal)
- **Awgb:** Protected attributes match Dutch Equal Treatment Act
