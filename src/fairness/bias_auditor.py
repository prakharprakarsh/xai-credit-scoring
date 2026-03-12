"""Bias Auditor — Comprehensive Fairness Analysis. EU AI Act Article 9."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetric:
    name: str
    value: float
    threshold: float
    passed: bool
    severity: str
    group_a: str
    group_b: str
    protected_attribute: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"{status} {self.name}: {self.value:.4f} (threshold: {self.threshold}, {self.protected_attribute}: {self.group_a} vs {self.group_b})"


@dataclass
class FairnessReport:
    metrics: list = field(default_factory=list)
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    group_statistics: dict = field(default_factory=dict)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def passes_all_thresholds(self) -> bool:
        return not any(m.severity == "BLOCK" and not m.passed for m in self.metrics)

    def summary(self) -> str:
        lines = ["=" * 60, "FAIRNESS AUDIT REPORT", "=" * 60, ""]
        if self.passes_all_thresholds():
            lines.append("OVERALL STATUS: PASS — All critical thresholds met")
        else:
            lines.append("OVERALL STATUS: FAIL — Critical threshold violations found")
        lines.append("")
        for metric in self.metrics:
            lines.append(str(metric))
        if self.violations:
            lines.append("\nCRITICAL VIOLATIONS:")
            for v in self.violations:
                lines.append(f"  - {v}")
        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


class BiasAuditor:
    DEFAULT_THRESHOLDS = {
        "demographic_parity_ratio": {"value": 0.80, "severity": "BLOCK"},
        "equalized_odds_difference": {"value": 0.30, "severity": "BLOCK"},
        "disparate_impact_ratio": {"value": 0.80, "severity": "BLOCK"},
        "predictive_parity_difference": {"value": 0.10, "severity": "WARN"},
        "calibration_difference": {"value": 0.05, "severity": "WARN"},
    }

    def __init__(self, model, test_data, test_labels, protected_attributes, reference_groups=None, thresholds=None):
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.protected_attributes = protected_attributes
        self.reference_groups = reference_groups or {}
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        numeric_data = test_data.select_dtypes(include='number')
        self.predictions = model.predict(numeric_data)
        self.probabilities = model.predict_proba(numeric_data)

    def full_audit(self) -> FairnessReport:
        report = FairnessReport()
        for attr in self.protected_attributes:
            if attr not in self.test_data.columns:
                continue
            groups = self.test_data[attr].unique()
            ref_group = self.reference_groups.get(attr, groups[0])
            report.group_statistics[attr] = self._compute_group_stats(attr)
            for group in groups:
                if group == ref_group:
                    continue
                metrics = self._compute_pairwise_metrics(attr, ref_group, group)
                for metric in metrics:
                    report.metrics.append(metric)
                    if not metric.passed:
                        if metric.severity == "BLOCK":
                            report.violations.append(str(metric))
                        else:
                            report.warnings.append(str(metric))
        logger.info(report.summary())
        return report

    def _compute_group_stats(self, attr):
        stats = {}
        for group in self.test_data[attr].unique():
            mask = self.test_data[attr] == group
            stats[str(group)] = {"n_samples": int(mask.sum()), "positive_rate": float(self.test_labels[mask].mean()), "predicted_positive_rate": float(self.predictions[mask].mean()), "mean_probability": float(self.probabilities[mask].mean())}
        return stats

    def _compute_pairwise_metrics(self, attr, group_a, group_b):
        mask_a = self.test_data[attr] == group_a
        mask_b = self.test_data[attr] == group_b
        y_a, y_b = self.test_labels[mask_a].values, self.test_labels[mask_b].values
        pred_a, pred_b = self.predictions[mask_a], self.predictions[mask_b]
        prob_a, prob_b = self.probabilities[mask_a], self.probabilities[mask_b]
        metrics = []

        # Demographic Parity Ratio
        rate_a, rate_b = pred_a.mean(), pred_b.mean()
        dpr = min(rate_a, rate_b) / max(rate_a, rate_b) if max(rate_a, rate_b) > 0 else 1.0
        t = self.thresholds["demographic_parity_ratio"]
        metrics.append(FairnessMetric("Demographic Parity Ratio", dpr, t["value"], dpr >= t["value"], t["severity"], str(group_a), str(group_b), attr))

        # Disparate Impact Ratio
        dir_val = rate_b / rate_a if rate_a > 0 else 1.0
        t = self.thresholds["disparate_impact_ratio"]
        metrics.append(FairnessMetric("Disparate Impact Ratio", dir_val, t["value"], dir_val >= t["value"], t["severity"], str(group_a), str(group_b), attr))

        # Equalized Odds Difference
        tpr_a = pred_a[y_a == 1].mean() if (y_a == 1).sum() > 0 else 0
        tpr_b = pred_b[y_b == 1].mean() if (y_b == 1).sum() > 0 else 0
        fpr_a = pred_a[y_a == 0].mean() if (y_a == 0).sum() > 0 else 0
        fpr_b = pred_b[y_b == 0].mean() if (y_b == 0).sum() > 0 else 0
        eod = abs(tpr_a - tpr_b) + abs(fpr_a - fpr_b)
        t = self.thresholds["equalized_odds_difference"]
        metrics.append(FairnessMetric("Equalized Odds Difference", eod, t["value"], eod <= t["value"], t["severity"], str(group_a), str(group_b), attr))

        # Predictive Parity
        ppv_a = y_a[pred_a == 1].mean() if (pred_a == 1).sum() > 0 else 0
        ppv_b = y_b[pred_b == 1].mean() if (pred_b == 1).sum() > 0 else 0
        ppd = abs(ppv_a - ppv_b)
        t = self.thresholds["predictive_parity_difference"]
        metrics.append(FairnessMetric("Predictive Parity Difference", ppd, t["value"], ppd <= t["value"], t["severity"], str(group_a), str(group_b), attr))

        # Calibration
        cal_a = prob_a[y_a == 1].mean() if (y_a == 1).sum() > 0 else 0
        cal_b = prob_b[y_b == 1].mean() if (y_b == 1).sum() > 0 else 0
        cal_diff = abs(cal_a - cal_b)
        t = self.thresholds["calibration_difference"]
        metrics.append(FairnessMetric("Calibration Difference", cal_diff, t["value"], cal_diff <= t["value"], t["severity"], str(group_a), str(group_b), attr))

        return metrics

    def intersectional_audit(self, attributes):
        present_attrs = [a for a in attributes if a in self.test_data.columns]
        if len(present_attrs) < 2:
            return {}
        intersection = self.test_data[present_attrs].astype(str).agg("_x_".join, axis=1)
        results = {}
        for group_name in intersection.unique():
            mask = intersection == group_name
            n = mask.sum()
            if n < 30:
                continue
            results[group_name] = {"n_samples": int(n), "approval_rate": float(self.predictions[mask].mean()), "mean_probability": float(self.probabilities[mask].mean())}
        return results
