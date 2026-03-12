"""SHAP Explainer — Global and Local Explanations. EU AI Act Article 13."""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, model, feature_names: Optional[list] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self._shap_values = None

    def fit(self, X_background: pd.DataFrame):
        if self.feature_names is None:
            self.feature_names = list(X_background.columns)
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP TreeExplainer initialized")

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call fit() first.")
        self._shap_values = self.explainer.shap_values(X)
        return self._shap_values

    def explain_single(self, applicant: Union[pd.Series, pd.DataFrame]) -> dict:
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call fit() first.")
        if isinstance(applicant, pd.Series):
            applicant = applicant.to_frame().T
        shap_values = self.explainer.shap_values(applicant)[0]
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]

        contributions = []
        for i, (name, sv) in enumerate(zip(self.feature_names, shap_values)):
            contributions.append({
                "feature": name, "shap_value": float(sv), "abs_shap_value": abs(float(sv)),
                "feature_value": float(applicant.iloc[0, i]) if i < len(applicant.columns) else None,
                "direction": "increases" if sv > 0 else "decreases",
            })
        contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)

        explanation = self._generate_natural_language(contributions[:5], base_value, shap_values.sum())
        return {
            "base_value": float(base_value),
            "shap_values": {c["feature"]: c["shap_value"] for c in contributions},
            "top_contributions": contributions[:5],
            "explanation": explanation,
            "prediction_components": {"base_rate": float(base_value), "total_shap_effect": float(shap_values.sum()), "final_score": float(base_value + shap_values.sum())},
        }

    def global_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        shap_values = self.compute_shap_values(X)
        importance = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0), "mean_shap": shap_values.mean(axis=0), "std_shap": shap_values.std(axis=0)})
        importance = importance.sort_values("mean_abs_shap", ascending=False)
        importance["rank"] = range(1, len(importance) + 1)
        return importance

    def _generate_natural_language(self, top_factors, base_value, total_effect):
        score = base_value + total_effect
        if score >= 0.7:
            decision_text = "Your application is APPROVED"
        elif score <= 0.3:
            decision_text = "Your application is DENIED"
        else:
            decision_text = "Your application requires REVIEW by a loan officer"
        lines = [f"{decision_text} (score: {score:.2f}).", "", "Key factors in this decision:"]
        for i, factor in enumerate(top_factors[:3], 1):
            direction = "positively" if factor["shap_value"] > 0 else "negatively"
            feature_name = factor["feature"].replace("_", " ").title()
            lines.append(f"  {i}. {feature_name} {direction} influenced your score (impact: {factor['shap_value']:+.3f})")
        negative_factors = [f for f in top_factors if f["shap_value"] < 0]
        if negative_factors and score < 0.7:
            feature_name = negative_factors[0]["feature"].replace("_", " ").title()
            lines.append("")
            lines.append(f"To improve your score, the most impactful change would be improving your {feature_name}.")
        return "\n".join(lines)

    def get_shap_explanation_object(self, X: pd.DataFrame):
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call fit() first.")
        return self.explainer(X)
