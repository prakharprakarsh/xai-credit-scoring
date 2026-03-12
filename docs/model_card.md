# Model Card — XAI Credit Scoring

## Model Details
- **Model Name:** XAI Credit Scoring v1.0
- **Model Type:** Gradient Boosted Decision Trees (XGBoost)
- **Task:** Binary classification — credit risk assessment
- **EU AI Act Classification:** High-Risk (Annex III, Section 5b)

## Intended Use
Automated credit scoring for consumer loan applications in EU/Dutch financial institutions. Includes human-in-the-loop review for borderline cases.

## Training Data
Synthetic dataset modeled after the German Credit Dataset, extended with Dutch market features. Protected attributes (gender, ethnicity, marital status) are excluded from model inputs.

## Performance Metrics (Test Set)
| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.80 |
| F1 Score | ~0.75 |
| Accuracy | ~0.74 |

## Fairness Metrics
| Metric | Gender | Age Group |
|--------|--------|-----------|
| Demographic Parity Ratio | >= 0.80 | >= 0.80 |
| Disparate Impact Ratio | >= 0.80 | >= 0.80 |
| Equalized Odds Diff | <= 0.10 | <= 0.15 |

## Limitations
- Uses synthetic data; real-world performance may differ
- Some features may act as proxies for protected attributes
- Model performance may degrade with economic shifts

## Ethical Considerations
- Credit scoring directly impacts financial lives
- Historical data contains systemic biases
- Human oversight is mandatory
