"""Data Preprocessor — Feature engineering and encoding."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class CreditPreprocessor:
    PROTECTED_ATTRIBUTES = ["gender", "ethnicity", "marital_status"]
    NUMERIC_FEATURES = [
        "age", "income", "loan_amount", "loan_duration_months", "existing_credits",
        "num_dependents", "savings_balance", "checking_balance",
        "employment_duration_years", "residence_duration_years", "debt_to_income_ratio",
    ]
    CATEGORICAL_FEATURES = ["employment_status", "housing_type", "loan_purpose", "education_level", "credit_history_status"]

    def __init__(self):
        self.numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        self.categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.transformer: Optional[ColumnTransformer] = None
        self._is_fitted = False
        self.feature_names: list = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
        df["monthly_payment_estimate"] = df["loan_amount"] / (df["loan_duration_months"] + 1)
        df["payment_to_income"] = df["monthly_payment_estimate"] / (df["income"] / 12 + 1)
        df["financial_stability"] = np.log1p(df["savings_balance"]) + np.log1p(df["checking_balance"]) + np.clip(df["employment_duration_years"], 0, 20) / 20
        df["age_group"] = pd.cut(df["age"], bins=[0, 29, 50, 65, 120], labels=["18-29", "30-50", "51-65", "65+"]).astype(str)
        return df

    def fit_transform(self, df: pd.DataFrame, keep_protected: bool = False) -> tuple:
        df = self._engineer_features(df)
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        numeric_cols += [c for c in ["loan_to_income", "monthly_payment_estimate", "payment_to_income", "financial_stability"] if c in df.columns]
        cat_cols = [c for c in self.CATEGORICAL_FEATURES if c in df.columns]

        self.transformer = ColumnTransformer(
            transformers=[("num", self.numeric_pipeline, numeric_cols), ("cat", self.categorical_encoder, cat_cols)],
            remainder="drop",
        )

        transformed = self.transformer.fit_transform(df)
        self.feature_names = numeric_cols + cat_cols
        self._is_fitted = True
        result = pd.DataFrame(transformed, columns=self.feature_names, index=df.index)

        if keep_protected:
            for attr in self.PROTECTED_ATTRIBUTES:
                if attr in df.columns:
                    result[attr] = df[attr].values
            if "age_group" in df.columns:
                result["age_group"] = df["age_group"].values

        logger.info(f"Preprocessed {len(result)} samples with {len(self.feature_names)} features")
        return result, self.feature_names

    def transform(self, df: pd.DataFrame, keep_protected: bool = False) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = self._engineer_features(df)
        transformed = self.transformer.transform(df)
        result = pd.DataFrame(transformed, columns=self.feature_names, index=df.index)
        if keep_protected:
            for attr in self.PROTECTED_ATTRIBUTES:
                if attr in df.columns:
                    result[attr] = df[attr].values
            if "age_group" in df.columns:
                result["age_group"] = df["age_group"].values
        return result
