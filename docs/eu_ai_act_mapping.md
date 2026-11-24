# EU AI Act mapping

Credit scoring of natural persons is listed as high-risk in Annex III, point 5(b). This repo is a technical prototype, not a compliance claim. The table shows which technical requirements it touches and what is missing.

| Article | What the code does | Not covered |
|---|---|---|
| Art. 10 Data governance | Schema/missing-value/range checks (`data_validator.py`), proxy correlation scan (`proxy_detector.py`), group representation stats | Data provenance, collection process, real bias examination on real data |
| Art. 12 Record-keeping | Every prediction and override is appended to `logs/audit.jsonl` | Retention, tamper-proofing |
| Art. 13 Transparency | Per-decision SHAP top factors + text explanation in the API | Instructions for use for deployers |
| Art. 14 Human oversight | Scores between 0.30 and 0.70 go to review; officers can override with a mandatory reason (`/override`) | Officer training, monitoring of override rates |
| Art. 15 Accuracy & robustness | Test-set metrics, group fairness gate in CI, perturbation tests | Adversarial robustness, cybersecurity |
| Art. 72 Post-market monitoring | Baseline snapshot in `logs/monitoring.jsonl` | Scheduled drift checks against live data |

Articles 9 (risk management) and 17 (quality management) are organisational processes and are out of scope.
