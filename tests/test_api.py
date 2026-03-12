"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

class TestAuditEndpoint:
    def test_audit_log_returns_list(self, client):
        response = client.get("/audit/log")
        assert response.status_code == 200
        assert "entries" in response.json()
