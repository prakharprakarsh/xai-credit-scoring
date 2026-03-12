"""Post-Deployment Monitoring — EU AI Act Article 61."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FairnessMonitor:
    def __init__(self, log_path="logs/monitoring.jsonl", drift_threshold=0.05):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold
        self.baseline_metrics = None

    def set_baseline(self, metrics):
        self.baseline_metrics = metrics
        self._log_event("baseline_set", metrics)

    def check_batch(self, predictions, probabilities, protected_data, labels=None):
        report = {"timestamp": datetime.now().isoformat(), "n_predictions": len(predictions), "approval_rate": float(predictions.mean()), "mean_probability": float(probabilities.mean()), "alerts": []}
        if self.baseline_metrics:
            baseline_rate = self.baseline_metrics.get("approval_rate", 0.5)
            rate_drift = abs(predictions.mean() - baseline_rate)
            if rate_drift > self.drift_threshold:
                report["alerts"].append({"type": "approval_rate_drift", "severity": "WARNING", "message": f"Approval rate drifted by {rate_drift:.3f}"})
        group_metrics = {}
        for col in protected_data.columns:
            col_metrics = {}
            for group in protected_data[col].unique():
                mask = protected_data[col] == group
                if mask.sum() < 10:
                    continue
                col_metrics[str(group)] = {"n": int(mask.sum()), "approval_rate": float(predictions[mask].mean()), "mean_probability": float(probabilities[mask].mean())}
            group_metrics[col] = col_metrics
            rates = [m["approval_rate"] for m in col_metrics.values()]
            if len(rates) >= 2 and max(rates) > 0:
                dir_ratio = min(rates) / max(rates)
                if dir_ratio < 0.80:
                    report["alerts"].append({"type": "disparate_impact", "severity": "CRITICAL", "attribute": col, "message": f"Disparate impact for '{col}': ratio = {dir_ratio:.3f}"})
        report["group_metrics"] = group_metrics
        self._log_event("batch_check", report)
        for alert in report["alerts"]:
            logger.warning(f"MONITORING ALERT: {alert['message']}")
        return report

    def _log_event(self, event_type, data):
        entry = {"timestamp": datetime.now().isoformat(), "event_type": event_type, "data": data}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
