"""Bias Mitigator — Pre/In/Post-processing bias mitigation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BiasMitigator:
    def __init__(self, protected_attribute: str, reference_group: str):
        self.protected_attribute = protected_attribute
        self.reference_group = reference_group

    def compute_reweighing_weights(self, X, y):
        if self.protected_attribute not in X.columns:
            return np.ones(len(X))
        groups = X[self.protected_attribute]
        n = len(X)
        weights = np.ones(n)
        for group in groups.unique():
            for outcome in [0, 1]:
                mask_group = groups == group
                mask_outcome = y == outcome
                mask_both = mask_group & mask_outcome
                p_group = mask_group.sum() / n
                p_outcome = mask_outcome.sum() / n
                p_both = mask_both.sum() / n
                if p_both > 0:
                    expected = p_group * p_outcome
                    weights[mask_both] = expected / p_both
        weights = weights * len(weights) / weights.sum()
        logger.info(f"Reweighing weights computed. Range: [{weights.min():.3f}, {weights.max():.3f}]")
        return weights

    def optimize_thresholds(self, probabilities, labels, groups, method="equalized_odds", base_threshold=0.5):
        unique_groups = groups.unique()
        thresholds = {str(g): base_threshold for g in unique_groups}
        if method == "demographic_parity":
            target_rate = (probabilities >= 0.5).mean()
            for group in unique_groups:
                mask = groups == group
                group_probs = probabilities[mask]
                best_threshold, best_diff = 0.5, float("inf")
                for t in np.arange(0.1, 0.9, 0.01):
                    rate = (group_probs >= t).mean()
                    diff = abs(rate - target_rate)
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = t
                thresholds[str(group)] = round(best_threshold, 3)
        elif method == "equalized_odds":
            ref_mask = groups == self.reference_group
            ref_probs, ref_labels = probabilities[ref_mask], labels[ref_mask]
            ref_preds = (ref_probs >= 0.5).astype(int)
            target_tpr = ref_preds[ref_labels == 1].mean() if (ref_labels == 1).sum() > 0 else 0.5
            target_fpr = ref_preds[ref_labels == 0].mean() if (ref_labels == 0).sum() > 0 else 0.1
            thresholds[str(self.reference_group)] = 0.5
            for group in unique_groups:
                if group == self.reference_group:
                    continue
                mask = groups == group
                group_probs, group_labels = probabilities[mask], labels[mask]
                best_threshold, best_score = 0.5, float("inf")
                for t in np.arange(0.1, 0.9, 0.01):
                    preds = (group_probs >= t).astype(int)
                    tpr = preds[group_labels == 1].mean() if (group_labels == 1).sum() > 0 else 0
                    fpr = preds[group_labels == 0].mean() if (group_labels == 0).sum() > 0 else 0
                    score = abs(tpr - target_tpr) + abs(fpr - target_fpr)
                    if score < best_score:
                        best_score = score
                        best_threshold = t
                thresholds[str(group)] = round(best_threshold, 3)
        logger.info(f"Optimized thresholds ({method}): {thresholds}")
        return thresholds

    def apply_group_thresholds(self, probabilities, groups, thresholds):
        predictions = np.zeros(len(probabilities), dtype=int)
        for group, threshold in thresholds.items():
            mask = groups.astype(str) == group
            predictions[mask] = (probabilities[mask] >= threshold).astype(int)
        return predictions
