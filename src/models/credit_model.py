"""Credit Scoring Model — Core training and prediction."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    decision: str
    probability: float
    risk_score: int
    explanation: str = ""
    shap_values: Optional[dict] = None
    counterfactual: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"decision": self.decision, "probability": round(self.probability, 4), "risk_score": self.risk_score, "explanation": self.explanation}


class CreditScoringModel:
    def __init__(self, approve_threshold: float = 0.70, deny_threshold: float = 0.30, model_params: Optional[dict] = None):
        self.approve_threshold = approve_threshold
        self.deny_threshold = deny_threshold
        default_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "eval_metric": "auc",
            "random_state": 42, "use_label_encoder": False,
        }
        if model_params:
            default_params.update(model_params)
        self.model = XGBClassifier(**default_params)
        self.feature_names: list = []
        self._is_fitted = False

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None) -> dict:
        self.feature_names = feature_names or list(X_train.columns)
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        logger.info(f"Training XGBoost on {len(X_train)} samples, {len(self.feature_names)} features")
        self.model.fit(X_train, y_train, **fit_params)
        self._is_fitted = True
        train_preds = self.model.predict(X_train)
        train_probs = self.model.predict_proba(X_train)[:, 1]
        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_auc": roc_auc_score(y_train, train_probs),
            "train_f1": f1_score(y_train, train_preds),
            "train_precision": precision_score(y_train, train_preds),
            "train_recall": recall_score(y_train, train_preds),
        }
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_probs = self.model.predict_proba(X_val)[:, 1]
            metrics.update({"val_accuracy": accuracy_score(y_val, val_preds), "val_auc": roc_auc_score(y_val, val_probs), "val_f1": f1_score(y_val, val_preds)})
        logger.info(f"Training complete. AUC: {metrics['train_auc']:.4f}")
        return metrics

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include="number")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include="number")
        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, applicant) -> PredictionResult:
        self._check_fitted()
        if isinstance(applicant, dict):
            applicant = pd.Series(applicant)
        X = applicant.values.reshape(1, -1)
        probability = self.model.predict_proba(X)[0, 1]
        if probability >= self.approve_threshold:
            decision = "APPROVED"
        elif probability <= self.deny_threshold:
            decision = "DENIED"
        else:
            decision = "REVIEW"
        risk_score = int(300 + probability * 550)
        return PredictionResult(decision=decision, probability=float(probability), risk_score=risk_score)

    def evaluate(self, X_test, y_test) -> dict:
        self._check_fitted()
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)
        metrics = {"accuracy": accuracy_score(y_test, preds), "auc": roc_auc_score(y_test, probs), "f1": f1_score(y_test, preds), "precision": precision_score(y_test, preds), "recall": recall_score(y_test, preds)}
        logger.info(f"Test evaluation — AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def save(self, path: str):
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CreditScoringModel":
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return instance

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
