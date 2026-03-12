"""
Data Validator — EU AI Act Article 10 (Data Governance).
Validates training data quality, detects potential biases.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def add_error(self, message: str):
        self.errors.append(message)
        self.passed = False
        logger.error(f"VALIDATION ERROR: {message}")

    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.warning(f"VALIDATION WARNING: {message}")

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"Validation {status}: {len(self.errors)} errors, {len(self.warnings)} warnings"


class DataValidator:
    REQUIRED_COLUMNS = ["income", "loan_amount", "loan_duration_months", "employment_duration_years", "credit_history_status"]

    def __init__(self, protected_attributes: Optional[list] = None, max_missing_rate: float = 0.10, min_samples_per_group: int = 50):
        self.protected_attributes = protected_attributes or ["gender", "age", "ethnicity"]
        self.max_missing_rate = max_missing_rate
        self.min_samples_per_group = min_samples_per_group

    def validate(self, df: pd.DataFrame, target_col: str = "credit_risk") -> ValidationReport:
        report = ValidationReport()
        self._check_schema(df, report)
        self._check_missing_values(df, report)
        self._check_class_balance(df, target_col, report)
        self._check_group_representation(df, target_col, report)
        self._check_outcome_rates_by_group(df, target_col, report)
        self._check_numeric_ranges(df, report)
        self._compute_statistics(df, target_col, report)
        logger.info(report.summary())
        return report

    def _check_schema(self, df, report):
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            report.add_error(f"Missing required columns: {missing_cols}")

    def _check_missing_values(self, df, report):
        missing_rates = df.isnull().mean()
        high_missing = missing_rates[missing_rates > self.max_missing_rate]
        for col, rate in high_missing.items():
            report.add_warning(f"Column '{col}' has {rate:.1%} missing values (threshold: {self.max_missing_rate:.1%})")

    def _check_class_balance(self, df, target_col, report):
        if target_col not in df.columns:
            report.add_error(f"Target column '{target_col}' not found")
            return
        balance = df[target_col].value_counts(normalize=True)
        minority_rate = balance.min()
        if minority_rate < 0.1:
            report.add_warning(f"Severe class imbalance: minority class is {minority_rate:.1%}")
        report.statistics["class_balance"] = balance.to_dict()

    def _check_group_representation(self, df, target_col, report):
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            group_counts = df[attr].value_counts()
            small_groups = group_counts[group_counts < self.min_samples_per_group]
            for group, count in small_groups.items():
                report.add_warning(f"Protected attribute '{attr}', group '{group}' has only {count} samples (minimum: {self.min_samples_per_group})")

    def _check_outcome_rates_by_group(self, df, target_col, report):
        if target_col not in df.columns:
            return
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            group_rates = df.groupby(attr)[target_col].mean()
            max_rate = group_rates.max()
            min_rate = group_rates.min()
            if max_rate > 0 and (min_rate / max_rate) < 0.80:
                report.add_warning(f"Potential bias in training data: '{attr}' groups have outcome rates ranging from {min_rate:.1%} to {max_rate:.1%} (ratio: {min_rate / max_rate:.2f})")

    def _check_numeric_ranges(self, df, report):
        checks = {"income": (0, 10_000_000), "loan_amount": (0, 50_000_000), "age": (18, 120), "loan_duration_months": (1, 360), "debt_to_income_ratio": (0, 100)}
        for col, (min_val, max_val) in checks.items():
            if col not in df.columns:
                continue
            n_below = (df[col] < min_val).sum()
            n_above = (df[col] > max_val).sum()
            if n_below > 0:
                report.add_warning(f"Column '{col}': {n_below} values below {min_val}")
            if n_above > 0:
                report.add_warning(f"Column '{col}': {n_above} values above {max_val}")

    def _compute_statistics(self, df, target_col, report):
        report.statistics["n_samples"] = len(df)
        report.statistics["n_features"] = len(df.columns) - 1
        report.statistics["missing_rate"] = df.isnull().mean().mean()
