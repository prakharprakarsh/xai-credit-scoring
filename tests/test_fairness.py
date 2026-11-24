"""Fairness checks. CI fails if any BLOCK-severity metric is violated."""

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
    # same size as the default pipeline run; at 2,000 rows the gender EO gap
    # sits right on the 0.10 line and the test becomes a coin flip
    raw = generate_synthetic_credit_data(n_samples=5000, random_state=42)
    loader = DataLoader(test_size=0.2, validation_size=0.15, random_state=42)
    splits = loader.split(raw, target_col="credit_risk")
    preprocessor = CreditPreprocessor()
    X_train, feature_names = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])
    X_test_protected = preprocessor.transform(splits["X_test"], keep_protected=True)
    # train the same way src/main.py does: reweighing on gender
    X_train_protected = preprocessor.transform(splits["X_train"], keep_protected=True)
    weights = BiasMitigator("gender", "male").compute_reweighing_weights(X_train_protected, splits["y_train"])
    model = CreditScoringModel()
    model.train(X_train, splits["y_train"], X_val=X_val, y_val=splits["y_val"], feature_names=feature_names, sample_weight=weights)
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
            if m.name == "Equalized Odds Difference" and m.severity == "BLOCK":
                assert m.passed, f"EOD FAILED: {m}"

    def test_overall_fairness_gate(self, pipeline):
        auditor = self._get_auditor(pipeline)
        report = auditor.full_audit()
        block_failures = [m for m in report.metrics if m.severity == "BLOCK" and not m.passed]
        if block_failures:
            pytest.fail(f"{len(block_failures)} blocking fairness violations: {block_failures}")


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


class TestAuditorConfig:
    def test_thresholds_loaded_from_yaml(self):
        from src.fairness.bias_auditor import load_config, load_thresholds

        cfg = load_config()
        th = load_thresholds()
        assert th["equalized_odds_difference"]["value"] == cfg["thresholds"]["equalized_odds_difference"]["value"]

    def test_small_groups_do_not_block(self, pipeline):
        df = pipeline["test_with_protected"].copy()
        # tiny synthetic group: 5 rows
        df["tiny"] = "big"
        df.iloc[:5, df.columns.get_loc("tiny")] = "small"
        auditor = BiasAuditor(model=pipeline["model"], test_data=df, test_labels=pipeline["y_test"], protected_attributes=["tiny"], reference_groups={"tiny": "big"})
        report = auditor.full_audit()
        assert all(m.severity == "WARN" for m in report.metrics)
        assert all(m.note for m in report.metrics)


def test_sample_weights_change_the_model():
    raw = generate_synthetic_credit_data(n_samples=1000, random_state=0)
    splits = DataLoader(test_size=0.2, validation_size=0.15, random_state=0).split(raw, target_col="credit_risk")
    pre = CreditPreprocessor()
    X, names = pre.fit_transform(splits["X_train"])
    y = splits["y_train"]
    w = np.where(y.values == 1, 5.0, 1.0)
    a, b = CreditScoringModel(model_params={"n_estimators": 50}), CreditScoringModel(model_params={"n_estimators": 50})
    a.train(X, y, feature_names=names)
    b.train(X, y, feature_names=names, sample_weight=w)
    assert not np.allclose(a.predict_proba(X), b.predict_proba(X))
