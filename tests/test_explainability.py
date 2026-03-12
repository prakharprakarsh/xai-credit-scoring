"""Tests for Explainability — SHAP, Counterfactuals. EU AI Act Article 13."""

import numpy as np
import pytest
from src.data.data_loader import DataLoader, generate_synthetic_credit_data
from src.data.preprocessor import CreditPreprocessor
from src.explainability.counterfactual import CounterfactualExplainer
from src.explainability.shap_explainer import SHAPExplainer
from src.models.credit_model import CreditScoringModel


@pytest.fixture(scope="module")
def trained_pipeline():
    raw = generate_synthetic_credit_data(n_samples=1000, random_state=42)
    loader = DataLoader(test_size=0.2, random_state=42)
    splits = loader.split(raw, target_col="credit_risk")
    preprocessor = CreditPreprocessor()
    X_train, feature_names = preprocessor.fit_transform(splits["X_train"])
    X_test = preprocessor.transform(splits["X_test"])
    model = CreditScoringModel()
    model.train(X_train, splits["y_train"], feature_names=feature_names)
    shap_explainer = SHAPExplainer(model=model.model, feature_names=feature_names)
    shap_explainer.fit(X_train)
    return {"model": model, "shap": shap_explainer, "X_train": X_train, "X_test": X_test, "y_test": splits["y_test"], "feature_names": feature_names}


class TestSHAPExplainer:
    def test_shap_values_shape(self, trained_pipeline):
        X = trained_pipeline["X_test"].head(10)
        shap_values = trained_pipeline["shap"].compute_shap_values(X)
        assert shap_values.shape == X.shape

    def test_single_explanation_has_all_features(self, trained_pipeline):
        result = trained_pipeline["shap"].explain_single(trained_pipeline["X_test"].iloc[0])
        assert "shap_values" in result
        assert "top_contributions" in result
        assert "explanation" in result

    def test_explanation_text_is_nonempty(self, trained_pipeline):
        result = trained_pipeline["shap"].explain_single(trained_pipeline["X_test"].iloc[0])
        assert len(result["explanation"]) > 0

    def test_global_feature_importance(self, trained_pipeline):
        importance = trained_pipeline["shap"].global_feature_importance(trained_pipeline["X_test"].head(50))
        assert len(importance) == len(trained_pipeline["feature_names"])
        assert "mean_abs_shap" in importance.columns


class TestCounterfactualExplainer:
    def test_counterfactual_for_denied(self, trained_pipeline):
        model = trained_pipeline["model"]
        X_test = trained_pipeline["X_test"]
        probs = model.predict_proba(X_test)
        denied_idx = np.where(probs < 0.3)[0]
        if len(denied_idx) == 0:
            pytest.skip("No denied applicants")
        cf = CounterfactualExplainer(feature_names=trained_pipeline["feature_names"])
        result = cf.generate(X_test.iloc[denied_idx[0]], model.model)
        assert "changes_needed" in result

    def test_counterfactual_respects_immutable(self, trained_pipeline):
        cf = CounterfactualExplainer(feature_names=trained_pipeline["feature_names"], immutable_features=["age"])
        result = cf.generate(trained_pipeline["X_test"].iloc[0], trained_pipeline["model"].model)
        changed = [c["feature"] for c in result.get("changes_needed", [])]
        assert "age" not in changed
