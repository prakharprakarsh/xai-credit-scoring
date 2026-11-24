"""Scoring API: predictions with SHAP explanations, officer overrides, and an append-only decision log.

Loads artifacts written by `python -m src.main` from MODEL_DIR (default models/latest).
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explainability.shap_explainer import SHAPExplainer
from src.models.credit_model import CreditScoringModel

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/latest"))
AUDIT_LOG = Path(os.getenv("AUDIT_LOG", "logs/audit.jsonl"))
VERSION = "0.2.0"

_decisions: dict = {}
_audit_log: list = []
_model = None
_preprocessor = None
_shap_explainer = None


def set_model(model, preprocessor, shap_explainer):
    global _model, _preprocessor, _shap_explainer
    _model, _preprocessor, _shap_explainer = model, preprocessor, shap_explainer


def load_artifacts(model_dir: Path = MODEL_DIR) -> bool:
    model_path, pre_path = model_dir / "model.joblib", model_dir / "preprocessor.joblib"
    if not (model_path.exists() and pre_path.exists()):
        logger.warning("No model in %s - run `python -m src.main` first", model_dir)
        return False
    model = CreditScoringModel.load(str(model_path))
    pre = joblib.load(pre_path)
    explainer = SHAPExplainer(model=model.model, feature_names=model.feature_names)
    explainer.fit(None)  # TreeExplainer needs no background data
    set_model(model, pre, explainer)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="XAI Credit Scoring API", version=VERSION, lifespan=lifespan)


class ApplicantInput(BaseModel):
    age: int = Field(..., ge=18, le=120)
    income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    loan_duration_months: int = Field(..., ge=1, le=360)
    existing_credits: int = Field(0, ge=0)
    num_dependents: int = Field(0, ge=0)
    savings_balance: float = Field(0, ge=0)
    checking_balance: float = Field(0, ge=0)
    employment_duration_years: float = Field(0, ge=0)
    residence_duration_years: float = Field(0, ge=0)
    employment_status: Literal["employed", "self_employed", "unemployed", "retired"] = "employed"
    housing_type: Literal["own", "rent", "free"] = "rent"
    loan_purpose: Literal["car", "furniture", "education", "business", "home_renovation", "other"] = "other"
    education_level: Literal["secondary", "vocational", "bachelor", "master", "phd"] = "bachelor"
    credit_history_status: Literal["no_credits", "all_paid", "existing_paid", "delayed", "critical"] = "existing_paid"


class OverrideRequest(BaseModel):
    decision_id: str
    officer_id: str
    new_decision: Literal["APPROVED", "DENIED"]
    reason: str = Field(..., min_length=10)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


def _log_decision(decision_id: str, data: dict) -> None:
    entry = {"timestamp": datetime.now().isoformat(), "decision_id": decision_id, **data}
    _audit_log.append(entry)
    _decisions.setdefault(decision_id, entry)
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=_model is not None, version=VERSION, timestamp=datetime.now().isoformat())


@app.post("/predict")
async def predict(applicant: ApplicantInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = applicant.model_dump()
    # derived the same way as in the training data, so the client can't send an inconsistent value
    row["debt_to_income_ratio"] = round(row["loan_amount"] / (row["income"] + 1), 3)
    processed = _preprocessor.transform(pd.DataFrame([row]))
    result = _model.predict_single(processed.iloc[0])
    explanation = _shap_explainer.explain_single(processed.iloc[0])
    decision_id = uuid.uuid4().hex[:12]
    _log_decision(decision_id, {"type": "prediction", "decision": result.decision, "probability": result.probability, "risk_score": result.risk_score})
    return {
        "decision_id": decision_id,
        "decision": result.decision,
        "probability": result.probability,
        "risk_score": result.risk_score,
        "requires_human_review": result.decision == "REVIEW",
        "top_factors": explanation["top_contributions"][:3],
        "explanation": explanation["explanation"],
    }


@app.post("/override")
async def override_decision(override: OverrideRequest):
    if override.decision_id not in _decisions:
        raise HTTPException(status_code=404, detail="Decision not found")
    original = _decisions[override.decision_id]
    _log_decision(
        override.decision_id,
        {
            "type": "override",
            "officer_id": override.officer_id,
            "original_decision": original.get("decision"),
            "new_decision": override.new_decision,
            "reason": override.reason,
        },
    )
    return {"decision_id": override.decision_id, "original_decision": original.get("decision"), "new_decision": override.new_decision, "logged": True}


@app.get("/audit/log")
async def get_audit_log(limit: int = 100, offset: int = 0):
    return {"total": len(_audit_log), "limit": limit, "offset": offset, "entries": _audit_log[offset : offset + limit]}
