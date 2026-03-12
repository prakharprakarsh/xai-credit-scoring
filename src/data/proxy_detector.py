"""Proxy Variable Detector — EU AI Act Data Governance."""

import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class ProxyReport:
    proxies_found: list = field(default_factory=list)
    correlation_matrix: dict = field(default_factory=dict)

    @property
    def has_proxies(self) -> bool:
        return len(self.proxies_found) > 0

    def summary(self) -> str:
        if not self.has_proxies:
            return "No proxy variables detected above threshold."
        lines = ["PROXY VARIABLES DETECTED:"]
        for proxy in self.proxies_found:
            lines.append(f"  - '{proxy['feature']}' correlates with '{proxy['protected_attr']}' (correlation: {proxy['correlation']:.3f})")
        return "\n".join(lines)


class ProxyDetector:
    def __init__(self, protected_attributes: list = None, correlation_threshold: float = 0.30):
        self.protected_attributes = protected_attributes or ["gender", "ethnicity", "marital_status"]
        self.correlation_threshold = correlation_threshold
        self._label_encoders: dict = {}

    def detect(self, df: pd.DataFrame, feature_columns: list = None) -> ProxyReport:
        report = ProxyReport()
        df_encoded = self._encode_for_correlation(df)
        protected_present = [a for a in self.protected_attributes if a in df_encoded.columns]

        if feature_columns is None:
            feature_columns = [c for c in df_encoded.columns if c not in self.protected_attributes and c != "credit_risk"]

        for attr in protected_present:
            for feature in feature_columns:
                if feature not in df_encoded.columns:
                    continue
                try:
                    corr = df_encoded[feature].corr(df_encoded[attr])
                    abs_corr = abs(corr)
                    if abs_corr >= self.correlation_threshold:
                        report.proxies_found.append({"feature": feature, "protected_attr": attr, "correlation": corr, "abs_correlation": abs_corr})
                except Exception:
                    continue

        report.proxies_found.sort(key=lambda x: x["abs_correlation"], reverse=True)
        logger.info(report.summary())
        return report

    def _encode_for_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
            if col not in self._label_encoders:
                self._label_encoders[col] = LabelEncoder()
                df_encoded[col] = self._label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self._label_encoders[col].transform(df_encoded[col].astype(str))
        return df_encoded
