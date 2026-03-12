"""LIME Explainer — Local Interpretable Model-Agnostic Explanations."""

import logging
from typing import Optional

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LIMEExplainer:
    def __init__(self, feature_names=None, categorical_features=None, num_features=10):
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.num_features = num_features
        self.explainer = None

    def fit(self, X_train: pd.DataFrame):
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        cat_indices = [i for i, name in enumerate(self.feature_names) if name in self.categorical_features]
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            feature_names=self.feature_names,
            categorical_features=cat_indices if cat_indices else None,
            class_names=["Bad Credit", "Good Credit"],
            mode="classification", random_state=42,
        )
        logger.info("LIME explainer initialized")

    def explain_single(self, applicant, predict_fn) -> dict:
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call fit() first.")
        if isinstance(applicant, pd.Series):
            applicant = applicant.values
        if applicant.ndim == 2:
            applicant = applicant[0]
        explanation = self.explainer.explain_instance(applicant, predict_fn, num_features=self.num_features, num_samples=5000)
        feature_weights = explanation.as_list()
        contributions = []
        for feature_desc, weight in feature_weights:
            contributions.append({"feature_description": feature_desc, "weight": float(weight), "abs_weight": abs(float(weight)), "direction": "positive" if weight > 0 else "negative"})
        contributions.sort(key=lambda x: x["abs_weight"], reverse=True)
        return {"contributions": contributions, "intercept": float(explanation.intercept[1]), "prediction_local": float(explanation.local_pred[0]), "score": float(explanation.predict_proba[1])}
