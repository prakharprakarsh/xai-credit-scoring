"""Explanation Generator — Combines SHAP, LIME, and counterfactuals. EU AI Act Article 13."""

import logging
from typing import Optional

import pandas as pd

from src.explainability.counterfactual import CounterfactualExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    def __init__(self, shap_explainer: SHAPExplainer, lime_explainer: Optional[LIMEExplainer] = None, counterfactual_explainer: Optional[CounterfactualExplainer] = None):
        self.shap = shap_explainer
        self.lime = lime_explainer
        self.counterfactual = counterfactual_explainer

    def explain(self, applicant, model, probability, decision, include_lime=True, include_counterfactual=True) -> dict:
        explanation = {"decision": decision, "probability": probability, "risk_score": int(300 + probability * 550)}
        shap_result = self.shap.explain_single(applicant)
        explanation["shap"] = shap_result
        explanation["natural_language"] = shap_result["explanation"]
        if include_lime and self.lime is not None:
            try:
                lime_result = self.lime.explain_single(applicant.values if isinstance(applicant, pd.Series) else applicant, model.predict_proba)
                explanation["lime"] = lime_result
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        if include_counterfactual and self.counterfactual is not None and decision in ("DENIED", "REVIEW"):
            try:
                cf_result = self.counterfactual.generate(applicant, model)
                explanation["counterfactual"] = cf_result
                if cf_result.get("changes_needed"):
                    explanation["natural_language"] += "\n\n" + cf_result["explanation"]
            except Exception as e:
                logger.warning(f"Counterfactual generation failed: {e}")
        return explanation
