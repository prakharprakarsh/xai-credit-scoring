"""End-to-end run: data checks -> train -> explain -> fairness audit -> save.

    python -m src.main
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.data.preprocessor import CreditPreprocessor
from src.data.proxy_detector import ProxyDetector
from src.explainability.shap_explainer import SHAPExplainer
from src.fairness.bias_auditor import BiasAuditor
from src.fairness.bias_mitigator import BiasMitigator
from src.fairness.monitoring import FairnessMonitor
from src.fairness.report_generator import FairnessReportGenerator
from src.models.credit_model import CreditScoringModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROTECTED = ["gender", "age_group", "ethnicity"]


def main():
    loader = DataLoader(test_size=0.2, validation_size=0.15)
    raw_data = loader.load()

    # data checks before anything is trained
    DataValidator(protected_attributes=["gender", "age", "ethnicity"]).validate(raw_data)
    ProxyDetector(protected_attributes=["gender", "ethnicity", "marital_status"], correlation_threshold=0.30).detect(raw_data)

    splits = loader.split(raw_data, target_col="credit_risk")
    pre = CreditPreprocessor()
    X_train, feature_names = pre.fit_transform(splits["X_train"], keep_protected=False)
    X_val = pre.transform(splits["X_val"], keep_protected=False)
    X_test = pre.transform(splits["X_test"], keep_protected=False)
    train_prot = pre.transform(splits["X_train"], keep_protected=True)
    val_prot = pre.transform(splits["X_val"], keep_protected=True)
    test_prot = pre.transform(splits["X_test"], keep_protected=True)

    # reweighing (Kamiran & Calders): gender is not a model input, but it is
    # used to weight training rows so each (gender, outcome) cell counts equally
    mitigator = BiasMitigator(protected_attribute="gender", reference_group="male")
    if "gender" in train_prot.columns:
        weights = mitigator.compute_reweighing_weights(train_prot, splits["y_train"])
    else:
        weights = np.ones(len(X_train))

    model = CreditScoringModel(approve_threshold=0.70, deny_threshold=0.30)
    model.train(X_train, splits["y_train"], X_val=X_val, y_val=splits["y_val"], feature_names=feature_names, sample_weight=weights)
    test_metrics = model.evaluate(X_test, splits["y_test"])

    explainer = SHAPExplainer(model=model.model, feature_names=feature_names)
    explainer.fit(X_train)
    logger.info("Top features by mean |SHAP|:\n%s", explainer.global_feature_importance(X_test).head())
    logger.info("Example explanation:\n%s", explainer.explain_single(X_test.iloc[0])["explanation"])

    # fairness audit on the held-out test set
    attrs = [c for c in PROTECTED if c in test_prot.columns]
    audit_df = X_test.copy()
    for a in attrs:
        audit_df[a] = test_prot[a].values
    auditor = BiasAuditor(model=model, test_data=audit_df, test_labels=splits["y_test"], protected_attributes=attrs, reference_groups={"gender": "male", "age_group": "30-50"})
    report = auditor.full_audit()

    Path("reports").mkdir(exist_ok=True)
    FairnessReportGenerator().generate_html(report, test_metrics, "reports/fairness_audit.html")
    if report.passes_all_thresholds():
        logger.info("Fairness gate: PASS")
    else:
        logger.warning("Fairness gate: FAIL\n%s", "\n".join(report.violations))

    # per-group decision thresholds, tuned on validation (not test) to avoid leakage
    if "gender" in val_prot.columns:
        thresholds = mitigator.optimize_thresholds(
            probabilities=model.predict_proba(X_val), labels=splits["y_val"].values, groups=val_prot["gender"], method="equalized_odds"
        )
        logger.info("Per-group thresholds (equalized odds, tuned on val): %s", thresholds)

    out = Path("models/latest")
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "model.joblib"))
    joblib.dump(pre, out / "preprocessor.joblib")
    # metrics as JSON so the dashboard can show the real audit, not placeholders
    (Path("reports") / "fairness_report.json").write_text(
        json.dumps(
            {
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "passes": report.passes_all_thresholds(),
                "metrics": [{**asdict(m), "value": float(m.value), "passed": bool(m.passed)} for m in report.metrics],
            },
            indent=2,
        )
    )

    FairnessMonitor(log_path="logs/monitoring.jsonl").set_baseline(
        {
            "approval_rate": float(model.predict(X_test).mean()),
            "mean_probability": float(model.predict_proba(X_test).mean()),
            "metrics": test_metrics,
        }
    )

    logger.info("Test AUC %.4f | model -> models/latest/model.joblib | report -> reports/fairness_audit.html", test_metrics["auc"])
    return model, test_metrics


if __name__ == "__main__":
    main()
