# XAI Credit Scoring

An XGBoost credit model with per-decision SHAP explanations, a group-fairness audit that runs as a CI gate, bias mitigation, and a human-review path for borderline scores. Built to see what the EU AI Act's requirements for high-risk credit scoring (Annex III, 5(b)) look like in actual code.

[![CI](https://github.com/prakharprakarsh/xai-credit-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/prakharprakarsh/xai-credit-scoring/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> Runs on **synthetic** data (5,000 applicants, German-Credit-style columns). Two biases are injected into the labels on purpose so the audit has something to find. Nothing here should be used on real people.

## How it works

```
synthetic data ─► data checks ─► preprocess ─► reweighing ─► XGBoost
  (Art. 10)        (schema,        (protected      (gender)        │
                    proxies)        attrs dropped)                 ▼
                                               SHAP ◄──── test-set fairness audit ──► CI gate
                                                │
                              FastAPI /predict ─┴─► approve ≥ 0.70 │ review │ deny ≤ 0.30
                                     /override  ──► officer decision + reason ──► audit log
```

1. **Data checks** — schema, missing values, ranges, group sizes, and a correlation scan for features that might proxy protected attributes.
2. **Training** — `gender`, `ethnicity` and `marital_status` are not model inputs. Training rows are reweighed (Kamiran & Calders) so each (gender × outcome) cell carries equal weight.
3. **Explanations** — TreeSHAP per applicant. SHAP on XGBoost is additive in log-odds, so the explanation converts back to a probability; a test checks it matches `predict_proba`.
4. **Fairness audit** — demographic parity ratio, disparate impact, equalized odds (fairlearn definition), predictive parity and score gaps for every group vs. a reference group. Thresholds live in `configs/fairness_config.yaml`.
5. **Human oversight** — scores between 0.30 and 0.70 go to review. Officers can override any decision via the API with a mandatory reason; everything is appended to `logs/audit.jsonl`.

## Results

Test set, n = 1,000 (`python -m src.main`):

| AUC | Accuracy | F1 |
|---|---|---|
| 0.815 | 0.782 | 0.848 |

Effect of reweighing on gender (male vs female):

| | AUC | Demographic parity ratio | Equalized odds diff |
|---|---|---|---|
| No mitigation | 0.817 | 0.963 | 0.079 |
| Reweighing | 0.815 | 0.972 | **0.055** |

About 0.001 AUC for a ~30% smaller error-rate gap.

Full per-group table in [`docs/model_card.md`](docs/model_card.md).

### The 65+ group

The audit fails equalized odds for age 65+ (0.20 vs a 0.10 limit). But that group has only 21 "bad" outcomes in the test set, so one applicant moves its false-positive rate by ~0.05. I added a `min_group_size` rule: below 30 rows in either outcome class, failures are reported as warnings with a note instead of blocking the build. The alternative — what an earlier version of this repo did — was quietly loosening the threshold to 0.30 for every group, which hides real problems in large groups too.

The honest fix is more data or a bootstrap confidence interval on the metric; that's next on the list.

## Quick start

```bash
git clone https://github.com/prakharprakarsh/xai-credit-scoring.git
cd xai-credit-scoring
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.main                              # train, explain, audit, save artifacts
uvicorn src.api.server:app --reload             # API on :8000 (docs at /docs)
python -m streamlit run src/dashboard/app.py    # review dashboard on :8501
```

Example request:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "age": 34, "income": 42000, "loan_amount": 9000, "loan_duration_months": 24,
  "savings_balance": 3000, "employment_duration_years": 4, "credit_history_status": "all_paid"
}'
```

Returns the decision, probability, top three SHAP factors and a text explanation, plus a `decision_id` you can pass to `/override`.

### API

| Endpoint | |
|---|---|
| `POST /predict` | Decision + explanation (categoricals validated with `Literal` types) |
| `POST /override` | Officer override, reason ≥ 10 chars, logged |
| `GET /audit/log` | Decisions and overrides |
| `GET /health` | Includes whether a model is loaded |

### Dashboard

Three pages, all reading the real artifacts from `src.main`: a review queue with SHAP bars per applicant, the fairness audit table (violations highlighted), and the decision log.

## Tests

```bash
pytest -q        # 38 tests
```

CI runs the suite on Python 3.10–3.12, then runs `tests/test_fairness.py` as a separate step that fails on any blocking fairness violation.

## Project layout

```
src/
  data/            synthetic generator, validation, proxy scan, preprocessing
  models/          XGBoost wrapper, optional Optuna tuner
  explainability/  SHAP, LIME, counterfactuals
  fairness/        auditor, mitigation, monitoring, HTML report
  api/             FastAPI service
  dashboard/       Streamlit app
  main.py          end-to-end run
configs/fairness_config.yaml
docs/              model card, EU AI Act mapping
```

## Limitations

- Synthetic data and labels
- Fairness metrics are point estimates, no confidence intervals
- Categoricals are ordinal-encoded; fine for trees, but SHAP factors are on encoded/scaled values
- Per-group decision thresholds are computed (on validation) but not applied, since using gender at decision time is itself a legal problem in EU credit
- Monitoring only records a baseline; no scheduled drift job

What the EU AI Act mapping does and doesn't cover: [`docs/eu_ai_act_mapping.md`](docs/eu_ai_act_mapping.md).

## License

MIT — Prakhar Prakarsh
