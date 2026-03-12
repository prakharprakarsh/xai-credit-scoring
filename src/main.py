"""Main Pipeline — Orchestrates the complete XAI Credit Scoring workflow."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.data.preprocessor import CreditPreprocessor
from src.data.proxy_detector import ProxyDetector
from src.explainability.counterfactual import CounterfactualExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.shap_explainer import SHAPExplainer
from src.fairness.bias_auditor import BiasAuditor
from src.fairness.bias_mitigator import BiasMitigator
from src.fairness.monitoring import FairnessMonitor
from src.fairness.report_generator import FairnessReportGenerator
from src.models.credit_model import CreditScoringModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("XAI CREDIT SCORING PIPELINE")
    logger.info("EU AI Act Compliant — High-Risk AI System")
    logger.info("=" * 60)

    # Step 1: Load Data
    logger.info("\n📥 STEP 1: Loading data...")
    loader = DataLoader(test_size=0.2, validation_size=0.15)
    raw_data = loader.load()

    # Step 2: Validate Data (EU AI Act Art. 10)
    logger.info("\n🔍 STEP 2: Validating data (EU AI Act Art. 10)...")
    validator = DataValidator(protected_attributes=["gender", "age", "ethnicity"])
    validation_report = validator.validate(raw_data)

    # Step 3: Detect Proxy Variables
    logger.info("\n🔎 STEP 3: Detecting proxy variables...")
    proxy_detector = ProxyDetector(protected_attributes=["gender", "ethnicity", "marital_status"], correlation_threshold=0.30)
    proxy_report = proxy_detector.detect(raw_data)

    # Step 4: Split Data
    logger.info("\n✂️ STEP 4: Splitting data...")
    splits = loader.split(raw_data, target_col="credit_risk")

    # Step 5: Preprocess
    logger.info("\n⚙️ STEP 5: Preprocessing features...")
    preprocessor = CreditPreprocessor()
    X_train_processed, feature_names = preprocessor.fit_transform(splits["X_train"], keep_protected=False)
    X_val_processed = preprocessor.transform(splits["X_val"], keep_protected=False)
    X_test_processed = preprocessor.transform(splits["X_test"], keep_protected=False)
    X_test_with_protected = preprocessor.transform(splits["X_test"], keep_protected=True)

    # Step 6: Bias Mitigation — Reweighing
    logger.info("\n⚖️ STEP 6: Computing reweighing weights...")
    X_train_with_protected = preprocessor.transform(splits["X_train"], keep_protected=True)
    mitigator = BiasMitigator(protected_attribute="gender", reference_group="male")
    if "gender" in X_train_with_protected.columns:
        sample_weights = mitigator.compute_reweighing_weights(X_train_with_protected, splits["y_train"])
    else:
        sample_weights = np.ones(len(X_train_processed))

    # Step 7: Train Model
    logger.info("\n🤖 STEP 7: Training credit scoring model...")
    model = CreditScoringModel(approve_threshold=0.70, deny_threshold=0.30)
    train_metrics = model.train(X_train_processed, splits["y_train"], X_val=X_val_processed, y_val=splits["y_val"], feature_names=feature_names)

    # Step 8: Evaluate
    logger.info("\n📈 STEP 8: Evaluating model...")
    test_metrics = model.evaluate(X_test_processed, splits["y_test"])

    # Step 9: Explainability (EU AI Act Art. 13)
    logger.info("\n💡 STEP 9: Setting up explainability...")
    shap_explainer = SHAPExplainer(model=model.model, feature_names=feature_names)
    shap_explainer.fit(X_train_processed)
    importance = shap_explainer.global_feature_importance(X_test_processed)
    logger.info(f"Top 5 features:\n{importance.head()}")

    sample_applicant = X_test_processed.iloc[0]
    explanation = shap_explainer.explain_single(sample_applicant)
    logger.info(f"\nSample explanation:\n{explanation['explanation']}")

    # Step 10: Fairness Audit (EU AI Act Art. 9)
    logger.info("\n⚖️ STEP 10: Running fairness audit...")
    protected_attrs_available = [col for col in ["gender", "age_group", "ethnicity"] if col in X_test_with_protected.columns]

    fairness_report = None
    if protected_attrs_available:
        test_data_for_audit = X_test_processed.copy()
        for attr in protected_attrs_available:
            if attr in X_test_with_protected.columns:
                test_data_for_audit[attr] = X_test_with_protected[attr].values

        auditor = BiasAuditor(model=model, test_data=test_data_for_audit, test_labels=splits["y_test"], protected_attributes=protected_attrs_available, reference_groups={"gender": "male", "age_group": "30-50"})
        fairness_report = auditor.full_audit()

        report_gen = FairnessReportGenerator()
        Path("reports").mkdir(exist_ok=True)
        report_gen.generate_html(fairness_report, test_metrics, "reports/fairness_audit.html")

        if not fairness_report.passes_all_thresholds():
            logger.warning("⚠️ FAIRNESS VIOLATIONS DETECTED")
        else:
            logger.info("✅ All fairness thresholds passed")

    # Step 11: Threshold optimization
    logger.info("\n🔧 STEP 11: Threshold optimization...")
    if "gender" in X_test_with_protected.columns:
        probabilities = model.predict_proba(X_test_processed)
        optimized_thresholds = mitigator.optimize_thresholds(probabilities=probabilities, labels=splits["y_test"].values, groups=X_test_with_protected["gender"], method="equalized_odds")
        logger.info(f"Optimized thresholds: {optimized_thresholds}")

    # Step 12: Save Model
    logger.info("\n💾 STEP 12: Saving model...")
    Path("models/latest").mkdir(parents=True, exist_ok=True)
    model.save("models/latest/model.joblib")

    # Step 13: Monitoring
    logger.info("\n📡 STEP 13: Initializing monitoring...")
    monitor = FairnessMonitor(log_path="logs/monitoring.jsonl")
    monitor.set_baseline({"approval_rate": float(model.predict(X_test_processed).mean()), "mean_probability": float(model.predict_proba(X_test_processed).mean()), "metrics": test_metrics})

    # Done
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Fairness: {'PASS' if fairness_report and fairness_report.passes_all_thresholds() else 'CHECK REPORT'}")
    logger.info(f"Model saved to: models/latest/model.joblib")
    logger.info(f"Fairness report: reports/fairness_audit.html")
    logger.info(f"\nNext steps:")
    logger.info(f"  - Start API: uvicorn src.api.server:app --reload")
    logger.info(f"  - Start Dashboard: streamlit run src/dashboard/app.py")

    return model, test_metrics


if __name__ == "__main__":
    main()
