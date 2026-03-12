"""Robustness Tests — EU AI Act Article 15."""

import numpy as np
import pandas as pd
import pytest
from src.data.data_loader import generate_synthetic_credit_data
from src.data.preprocessor import CreditPreprocessor
from src.models.credit_model import CreditScoringModel


@pytest.fixture(scope="module")
def model_and_data():
    raw = generate_synthetic_credit_data(n_samples=2000, random_state=42)
    preprocessor = CreditPreprocessor()
    target = raw["credit_risk"]
    features = raw.drop(columns=["credit_risk"])
    X, feature_names = preprocessor.fit_transform(features)
    model = CreditScoringModel()
    model.train(X, target, feature_names=feature_names)
    return {"model": model, "X": X, "y": target, "feature_names": feature_names}


class TestEdgeCases:
    def test_extreme_high_values(self, model_and_data):
        X = model_and_data["X"].iloc[0:1].copy()
        X.iloc[0] = X.iloc[0] * 100
        probs = model_and_data["model"].predict_proba(X)
        assert 0 <= probs[0] <= 1

    def test_zero_values(self, model_and_data):
        X = pd.DataFrame(np.zeros((1, len(model_and_data["feature_names"]))), columns=model_and_data["feature_names"])
        probs = model_and_data["model"].predict_proba(X)
        assert 0 <= probs[0] <= 1


class TestPredictionStability:
    def test_deterministic_predictions(self, model_and_data):
        X = model_and_data["X"].head(10)
        p1 = model_and_data["model"].predict_proba(X)
        p2 = model_and_data["model"].predict_proba(X)
        np.testing.assert_array_equal(p1, p2)

    def test_no_nan_predictions(self, model_and_data):
        probs = model_and_data["model"].predict_proba(model_and_data["X"])
        assert not np.any(np.isnan(probs))

    def test_no_inf_predictions(self, model_and_data):
        probs = model_and_data["model"].predict_proba(model_and_data["X"])
        assert not np.any(np.isinf(probs))
