"""Tests for Credit Scoring Model."""

import numpy as np
import pytest
from src.data.data_loader import DataLoader, generate_synthetic_credit_data
from src.data.preprocessor import CreditPreprocessor
from src.models.credit_model import CreditScoringModel, PredictionResult


@pytest.fixture
def sample_data():
    raw = generate_synthetic_credit_data(n_samples=1000, random_state=42)
    loader = DataLoader(test_size=0.2, validation_size=0.15, random_state=42)
    splits = loader.split(raw, target_col="credit_risk")
    preprocessor = CreditPreprocessor()
    X_train, feature_names = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])
    return {"X_train": X_train, "y_train": splits["y_train"], "X_val": X_val, "y_val": splits["y_val"], "X_test": X_test, "y_test": splits["y_test"], "feature_names": feature_names}


@pytest.fixture
def trained_model(sample_data):
    model = CreditScoringModel()
    model.train(sample_data["X_train"], sample_data["y_train"], X_val=sample_data["X_val"], y_val=sample_data["y_val"], feature_names=sample_data["feature_names"])
    return model


class TestDataGeneration:
    def test_synthetic_data_shape(self):
        data = generate_synthetic_credit_data(n_samples=500)
        assert len(data) == 500
        assert "credit_risk" in data.columns

    def test_synthetic_data_target_values(self):
        data = generate_synthetic_credit_data(n_samples=1000)
        assert set(data["credit_risk"].unique()).issubset({0, 1})

    def test_data_split_sizes(self):
        data = generate_synthetic_credit_data(n_samples=1000)
        loader = DataLoader(test_size=0.2, validation_size=0.15)
        splits = loader.split(data)
        total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        assert total == 1000


class TestCreditScoringModel:
    def test_model_trains(self, sample_data):
        model = CreditScoringModel()
        metrics = model.train(sample_data["X_train"], sample_data["y_train"], feature_names=sample_data["feature_names"])
        assert "train_auc" in metrics
        assert metrics["train_auc"] > 0.5

    def test_model_auc_above_threshold(self, trained_model, sample_data):
        metrics = trained_model.evaluate(sample_data["X_test"], sample_data["y_test"])
        assert metrics["auc"] > 0.60

    def test_predict_returns_binary(self, trained_model, sample_data):
        preds = trained_model.predict(sample_data["X_test"])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self, trained_model, sample_data):
        probs = trained_model.predict_proba(sample_data["X_test"])
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_single_returns_result(self, trained_model, sample_data):
        result = trained_model.predict_single(sample_data["X_test"].iloc[0])
        assert isinstance(result, PredictionResult)
        assert result.decision in ("APPROVED", "DENIED", "REVIEW")
        assert 300 <= result.risk_score <= 850

    def test_model_save_load(self, trained_model, sample_data, tmp_path):
        path = str(tmp_path / "model.joblib")
        trained_model.save(path)
        loaded = CreditScoringModel.load(path)
        preds_original = trained_model.predict(sample_data["X_test"])
        preds_loaded = loaded.predict(sample_data["X_test"])
        np.testing.assert_array_equal(preds_original, preds_loaded)
