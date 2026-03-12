"""
Data loading and splitting utilities.
Loads the German Credit-style dataset and prepares train/test/validation splits.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def generate_synthetic_credit_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic credit scoring dataset for demonstration.
    In production, replace this with your actual data loading logic.
    """
    rng = np.random.default_rng(random_state)

    data = {
        "gender": rng.choice(["male", "female"], n_samples, p=[0.55, 0.45]),
        "age": rng.integers(18, 70, n_samples),
        "ethnicity": rng.choice(
            ["dutch", "other_european", "non_european"],
            n_samples, p=[0.70, 0.15, 0.15],
        ),
        "marital_status": rng.choice(
            ["single", "married", "divorced", "widowed"],
            n_samples, p=[0.30, 0.45, 0.20, 0.05],
        ),
        "income": rng.lognormal(mean=10.5, sigma=0.6, size=n_samples).astype(int),
        "loan_amount": rng.lognormal(mean=9.0, sigma=0.8, size=n_samples).astype(int),
        "loan_duration_months": rng.choice([6, 12, 18, 24, 36, 48, 60], n_samples),
        "existing_credits": rng.integers(0, 5, n_samples),
        "num_dependents": rng.integers(0, 5, n_samples),
        "savings_balance": rng.lognormal(mean=8.0, sigma=1.5, size=n_samples).astype(int),
        "checking_balance": rng.lognormal(mean=7.5, sigma=1.2, size=n_samples).astype(int),
        "employment_duration_years": rng.exponential(scale=5, size=n_samples).round(1),
        "residence_duration_years": rng.exponential(scale=4, size=n_samples).round(1),
        "employment_status": rng.choice(
            ["employed", "self_employed", "unemployed", "retired"],
            n_samples, p=[0.60, 0.20, 0.10, 0.10],
        ),
        "housing_type": rng.choice(
            ["own", "rent", "free"], n_samples, p=[0.45, 0.45, 0.10]
        ),
        "loan_purpose": rng.choice(
            ["car", "furniture", "education", "business", "home_renovation", "other"],
            n_samples, p=[0.25, 0.15, 0.15, 0.20, 0.15, 0.10],
        ),
        "education_level": rng.choice(
            ["secondary", "vocational", "bachelor", "master", "phd"],
            n_samples, p=[0.20, 0.25, 0.30, 0.20, 0.05],
        ),
        "credit_history_status": rng.choice(
            ["no_credits", "all_paid", "existing_paid", "delayed", "critical"],
            n_samples, p=[0.10, 0.30, 0.35, 0.15, 0.10],
        ),
    }

    df = pd.DataFrame(data)
    df["debt_to_income_ratio"] = (df["loan_amount"] / (df["income"] + 1)).round(3)

    score = (
        0.3 * np.log1p(df["income"]) / 12
        + 0.2 * (1 - df["debt_to_income_ratio"].clip(0, 2) / 2)
        + 0.15 * np.clip(df["employment_duration_years"] / 10, 0, 1)
        + 0.15 * np.where(df["credit_history_status"].isin(["all_paid", "existing_paid"]), 1, 0)
        + 0.1 * np.log1p(df["savings_balance"]) / 12
        + 0.1 * np.where(df["housing_type"] == "own", 1, 0)
    )

    score += rng.normal(0, 0.1, n_samples)
    threshold = np.percentile(score, 30)
    df["credit_risk"] = (score > threshold).astype(int)

    # Introduce subtle historical bias (to demonstrate fairness auditing)
    bias_mask = (df["gender"] == "female") & (rng.random(n_samples) < 0.05)
    df.loc[bias_mask, "credit_risk"] = 0
    age_bias_mask = (df["age"] > 60) & (rng.random(n_samples) < 0.03)
    df.loc[age_bias_mask, "credit_risk"] = 0

    logger.info(f"Generated {n_samples} samples. Default rate: {1 - df['credit_risk'].mean():.1%}")
    return df


class DataLoader:
    """Loads and splits credit scoring data."""

    def __init__(self, test_size: float = 0.2, validation_size: float = 0.15, random_state: int = 42):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        if path and Path(path).exists():
            logger.info(f"Loading data from {path}")
            return pd.read_csv(path)
        logger.info("No data file found — generating synthetic data")
        return generate_synthetic_credit_data()

    def split(self, df: pd.DataFrame, target_col: str = "credit_risk") -> dict:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state,
        )

        val_ratio = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=self.random_state,
        )

        logger.info(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}
