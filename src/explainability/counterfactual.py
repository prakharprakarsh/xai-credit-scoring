"""Counterfactual Explanations — 'What would need to change?'"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    def __init__(self, feature_names: list, immutable_features: Optional[list] = None, feature_ranges: Optional[dict] = None):
        self.feature_names = feature_names
        self.immutable_features = immutable_features or ["age", "gender", "ethnicity"]
        self.feature_ranges = feature_ranges or {}

    def generate(self, applicant, model, target_probability=0.70, max_features_to_change=3, n_steps=50) -> dict:
        current = applicant.copy()
        if isinstance(current, pd.Series):
            current_values = current.values.reshape(1, -1)
        else:
            current_values = np.array(current).reshape(1, -1)
        current_prob = model.predict_proba(current_values)[0, 1]
        if current_prob >= target_probability:
            return {"status": "already_approved", "current_probability": float(current_prob), "changes_needed": [], "message": "This applicant already meets the approval threshold."}

        mutable_indices = [i for i, name in enumerate(self.feature_names) if name not in self.immutable_features]
        feature_effects = []
        for idx in mutable_indices:
            feature_name = self.feature_names[idx]
            original_value = float(current_values[0, idx])
            best_change = None
            best_prob = current_prob
            min_val = self.feature_ranges.get(feature_name, {}).get("min", original_value * 0.1)
            max_val = self.feature_ranges.get(feature_name, {}).get("max", original_value * 3.0)
            for new_value in np.linspace(min_val, max_val, n_steps):
                modified = current_values.copy()
                modified[0, idx] = new_value
                new_prob = model.predict_proba(modified)[0, 1]
                if new_prob > best_prob:
                    best_prob = new_prob
                    best_change = {"feature": feature_name, "original_value": original_value, "suggested_value": float(new_value), "change": float(new_value - original_value), "probability_gain": float(new_prob - current_prob), "new_probability": float(new_prob)}
            if best_change:
                feature_effects.append(best_change)

        feature_effects.sort(key=lambda x: x["probability_gain"], reverse=True)
        selected_changes = feature_effects[:max_features_to_change]
        combined_values = current_values.copy()
        for change in selected_changes:
            idx = self.feature_names.index(change["feature"])
            combined_values[0, idx] = change["suggested_value"]
        combined_prob = model.predict_proba(combined_values)[0, 1]

        explanation = self._format_counterfactual(selected_changes, current_prob, combined_prob, target_probability)
        return {"status": "counterfactual_found" if combined_prob >= target_probability else "partial_improvement", "current_probability": float(current_prob), "counterfactual_probability": float(combined_prob), "target_probability": target_probability, "changes_needed": selected_changes, "reaches_target": combined_prob >= target_probability, "explanation": explanation}

    def _format_counterfactual(self, changes, current_prob, new_prob, target):
        lines = [f"Current approval probability: {current_prob:.1%}", f"Target: {target:.1%}", "", "Suggested changes:"]
        for i, change in enumerate(changes, 1):
            feature_name = change["feature"].replace("_", " ").title()
            direction = "increase" if change["change"] > 0 else "decrease"
            lines.append(f"  {i}. {direction.capitalize()} your {feature_name} from {change['original_value']:.2f} to {change['suggested_value']:.2f} (+{change['probability_gain']:.1%} improvement)")
        lines.append("")
        if new_prob >= target:
            lines.append(f"With these changes, your estimated approval probability would be {new_prob:.1%}.")
        else:
            lines.append(f"These changes would improve your probability to {new_prob:.1%}.")
        return "\n".join(lines)
