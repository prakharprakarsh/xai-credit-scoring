"""Fairness Tests — CI/CD Gate. Blocks deployment if thresholds violated."""

import numpy as np
import pandas as pd
import pytest
from src.data.data_loader import DataLoader, generate_synthetic_credit_data
from src.data.data_validator import DataValidator
from src.data.preprocessor import CreditPreprocessor
from src.data.proxy_detector import ProxyDetector
from src.fairness.bias_auditor import BiasAuditor
from src.fairness.bias_mitigator import BiasMitigator
from src.models.credit_model import CreditScoringModel


@pytest.fixture(scope="module")
def pipeline():
    raw = generate_synthetic_credit_data(n_samples=2000, random_state=42)
    loader = DataLoader(test_size=0.2, validation_size=0.15, random_state=42)
    splits = loader.split(raw, target_col="credit_risk")
    preprocessor = CreditPreprocessor()
    X_train, feature_names = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])
    X_test_protected = preprocessor.transform(splits["X_test"], keep_protected=True)
    model = CreditScoringModel()
    model.train(X_train, splits["y_train"], X_val=X_val, y_val=splits["y_val"], feature_names=feature_names)
    test_with_protected = X_test.copy()
    for col in ["gender", "age_group"]:
        if col in X_test_protected.columns:
            test_with_protected[col] = X_test_protected[col].values
    return {"model": model, "X_test": X_test, "y_test": splits["y_test"], "test_with_protected": test_with_protected, "raw_data": raw}


class TestDataGovernance:
    def test_data_validation_passes(self, pipeline):
        validator = DataValidator()
        report = validator.validate(pipeline["raw_data"])
        assert report.passed

    def test_no_strong_proxy_variables(self, pipeline):
        detector = ProxyDetector(correlation_threshold=0.50)
        report = detector.detect(pipeline["raw_data"])
        strong_proxies = [p for p in report.proxies_found if p["abs_correlation"] > 0.50]
        assert len(strong_proxies) == 0


class TestFairnessMetrics:
    def _get_auditor(self, pipeline):
        protected_attrs = [col for col in ["gender", "age_group"] if col in pipeline["test_with_protected"].columns]
        return BiasAuditor(model=pipeline["model"], test_data=pipeline["test_with_protected"], test_labels=pipeline["y_test"], protected_attributes=protected_attrs, reference_groups={"gender": "male", "age_group": "30-50"})

    def test_demographic_parity_ratio(self, pipeline):
        auditor = self._get_auditor(pipeline)
        report = auditor.full_audit()
        for m in report.metrics:
            if m.name == "Demographic Parity Ratio":
                assert m.value >= 0.80, f"DPR FAILED: {m.value:.4f}"

    def test_disparate_impact_ratio(self, pipeline):
        auditor = self._get_auditor(pipeline)
        report = auditor.full_audit()
        for m in report.metrics:
            if m.name == "Disparate Impact Ratio":
                assert m.value >= 0.80, f"DIR FAILED: {m.value:.4f}"

    def test_equalized_odds_difference(self, pipeline):
        auditor = self._get_auditor(pipeline)
        report = auditor.full_audit()
        for m in report.metrics:
            if m.name == "Equalized Odds Difference":
                assert m.value <= 0.30, f"EOD FAILED: {m.value:.4f}"

    def test_overall_fairness_gate(self, pipeline):
        auditor = self._get_auditor(pipeline)
        report = auditor.full_audit()
        block_failures = [m for m in report.metrics if m.severity == "BLOCK" and not m.passed]
        if block_failures:
            pytest.fail(f"DEPLOYMENT BLOCKED — {len(block_failures)} violations")


class TestBiasMitigation:
    def test_reweighing_produces_valid_weights(self, pipeline):
        data = pipeline["raw_data"]
        X = data.drop(columns=["credit_risk"])
        y = data["credit_risk"]
        mitigator = BiasMitigator(protected_attribute="gender", reference_group="male")
        weights = mitigator.compute_reweighing_weights(X, y)
        assert len(weights) == len(X)
        assert weights.min() > 0
        assert np.isfinite(weights).all()

    def test_threshold_optimization_returns_valid(self, pipeline):
        model = pipeline["model"]
        probs = model.predict_proba(pipeline["X_test"])
        groups = pd.Series(np.random.choice(["A", "B"], len(pipeline["X_test"])), index=pipeline["X_test"].index)
        mitigator = BiasMitigator(protected_attribute="group", reference_group="A")
        thresholds = mitigator.optimize_thresholds(probs, pipeline["y_test"].values, groups, method="demographic_parity")
        for group, threshold in thresholds.items():
            assert 0.0 < threshold < 1.0
