# Model card

| | |
|---|---|
| Model | XGBoost binary classifier (`src/models/credit_model.py`) |
| Task | Credit risk: good (1) / bad (0), with a three-way decision on top: approve ≥ 0.70, deny ≤ 0.30, otherwise human review |
| Data | 5,000 synthetic applicants (`src/data/data_loader.py`), 65/15/20 train/val/test split |
| Version | 0.2.0 |

## Intended use

A worked example of explainability and fairness checks around a credit model. It is **not** trained on real lending data and should not be used to make decisions about people.

## Data

Synthetic, loosely modelled on the German Credit dataset. Labels come from a hand-written scoring rule plus noise. Two biases are injected on purpose: ~5% of women and ~3% of over-60s are flipped to "bad", so the audit has something real to detect.

Protected attributes (`gender`, `ethnicity`, `marital_status`) are removed from model inputs. `age` is an input; `age_group` is derived for auditing.

## Performance (test set, n = 1,000)

| AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| 0.815 | 0.782 | 0.848 | 0.808 | 0.893 |

## Fairness (test set, reference groups: male, age 30-50, dutch)

| Group | DPR | Equalized odds diff |
|---|---|---|
| female | 0.972 | 0.055 |
| age 18-29 | 0.923 | 0.070 |
| age 51-65 | 0.976 | 0.024 |
| age 65+ | 0.953 | **0.199** (warning, n too small) |
| non-European | 0.977 | 0.057 |
| other European | 0.970 | 0.073 |

Thresholds: DPR ≥ 0.80, equalized odds ≤ 0.10 (blocking). The 65+ group has only 21 "bad" outcomes in the test set, so its error-rate gap is reported but not used to block.

## Limitations

- Synthetic data and labels; numbers say nothing about real-world performance
- Ordinal-encoded categoricals; SHAP values are in log-odds on scaled features
- Fairness metrics are point estimates with no confidence intervals yet
- `ethnicity` is synthetic; in practice it usually isn't collected at all, which makes auditing harder
