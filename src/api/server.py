"""FastAPI Server — Credit Scoring API with Explanation Endpoints. EU AI Act Article 14."""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="XAI Credit Scoring API", description="EU AI Act Compliant", version="1.0.0")


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
    employment_status: str = "employed"
    housing_type: str = "rent"
    loan_purpose: str = "other"
    education_level: str = "bachelor"
    credit_history_status: str = "existing_paid"


class OverrideRequest(BaseModel):
    decision_id: str
    officer_id: str
    new_decision: str
    reason: str = Field(..., min_length=10)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


_decisions = {}
_audit_log = []
_model = None
_preprocessor = None
_shap_explainer = None


def set_model(model, preprocessor, shap_explainer):
    global _model, _preprocessor, _shap_explainer
    _model = model
    _preprocessor = preprocessor
    _shap_explainer = shap_explainer


def _log_decision(decision_id, data):
    entry = {"timestamp": datetime.now().isoformat(), "decision_id": decision_id, **data}
    _audit_log.append(entry)
    _decisions[decision_id] = entry
    log_path = Path("logs/audit.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=_model is not None, version="1.0.0", timestamp=datetime.now().isoformat())


@app.post("/predict")
async def predict(applicant: ApplicantInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    import uuid
    import pandas as pd
    input_data = pd.DataFrame([applicant.model_dump()])
    processed = _preprocessor.transform(input_data)
    result = _model.predict_single(processed.iloc[0])
    explanation_data = _shap_explainer.explain_single(processed.iloc[0])
    decision_id = str(uuid.uuid4())[:12]
    _log_decision(decision_id, {"type": "prediction", "decision": result.decision, "probability": result.probability, "risk_score": result.risk_score})
    return {"decision": result.decision, "probability": result.probability, "risk_score": result.risk_score, "explanation": explanation_data["explanation"], "decision_id": decision_id, "timestamp": datetime.now().isoformat(), "requires_human_review": result.decision == "REVIEW"}


@app.post("/override")
async def override_decision(override: OverrideRequest):
    if override.decision_id not in _decisions:
        raise HTTPException(status_code=404, detail="Decision not found")
    original = _decisions[override.decision_id]
    _log_decision(override.decision_id, {"type": "override", "officer_id": override.officer_id, "original_decision": original.get("decision", "UNKNOWN"), "new_decision": override.new_decision, "reason": override.reason})
    return {"decision_id": override.decision_id, "original_decision": original.get("decision"), "new_decision": override.new_decision, "officer_id": override.officer_id, "timestamp": datetime.now().isoformat(), "logged": True}


@app.get("/audit/log")
async def get_audit_log(limit: int = 100, offset: int = 0):
    return {"total": len(_audit_log), "limit": limit, "offset": offset, "entries": _audit_log[offset:offset + limit]}
