"""API tests against a small model trained in the fixture."""

import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.data.data_loader import DataLoader, generate_synthetic_credit_data
from src.data.preprocessor import CreditPreprocessor
from src.explainability.shap_explainer import SHAPExplainer
from src.models.credit_model import CreditScoringModel

APPLICANT = {
    "age": 34,
    "income": 42000,
    "loan_amount": 9000,
    "loan_duration_months": 24,
    "savings_balance": 3000,
    "checking_balance": 1200,
    "employment_duration_years": 4,
    "housing_type": "rent",
    "credit_history_status": "all_paid",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    raw = generate_synthetic_credit_data(n_samples=800, random_state=1)
    splits = DataLoader(random_state=1).split(raw, target_col="credit_risk")
    pre = CreditPreprocessor()
    X, names = pre.fit_transform(splits["X_train"])
    model = CreditScoringModel(model_params={"n_estimators": 50})
    model.train(X, splits["y_train"], feature_names=names)
    explainer = SHAPExplainer(model=model.model, feature_names=names)
    explainer.fit(X)
    server.AUDIT_LOG = tmp_path_factory.mktemp("logs") / "audit.jsonl"
    server.set_model(model, pre, explainer)
    return TestClient(server.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_decision_and_explanation(client):
    r = client.post("/predict", json=APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"APPROVED", "DENIED", "REVIEW"}
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["top_factors"]) == 3


def test_unknown_category_rejected(client):
    r = client.post("/predict", json=APPLICANT | {"housing_type": "castle"})
    assert r.status_code == 422


def test_override_is_logged(client):
    decision_id = client.post("/predict", json=APPLICANT).json()["decision_id"]
    r = client.post("/override", json={"decision_id": decision_id, "officer_id": "o-17", "new_decision": "DENIED", "reason": "income could not be verified"})
    assert r.status_code == 200
    entries = client.get("/audit/log").json()["entries"]
    assert any(e["type"] == "override" and e["decision_id"] == decision_id for e in entries)
    assert server.AUDIT_LOG.exists()


def test_override_unknown_decision_404(client):
    r = client.post("/override", json={"decision_id": "nope", "officer_id": "o-1", "new_decision": "APPROVED", "reason": "long enough reason"})
    assert r.status_code == 404
