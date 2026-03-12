"""Hyperparameter Tuner — Optuna-based optimization."""

import logging
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    def __init__(self, n_trials: int = 100, cv_folds: int = 5, timeout: int = 3600, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout = timeout
        self.random_state = random_state
        self.best_params: Optional[dict] = None
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial, X, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.random_state, "use_label_encoder": False, "eval_metric": "auc",
        }
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        auc_scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            probs = model.predict_proba(X[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y[val_idx], probs))
        return np.mean(auc_scores)

    def tune(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        logger.info(f"Starting Optuna tuning: {self.n_trials} trials")
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(lambda trial: self._objective(trial, X_np, y_np), n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = self.study.best_params
        self.best_params["random_state"] = self.random_state
        self.best_params["use_label_encoder"] = False
        self.best_params["eval_metric"] = "auc"
        logger.info(f"Tuning complete. Best AUC: {self.study.best_value:.4f}")
        return self.best_params
