# src/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# FastAPI provides a TestClient wrapper that allows us to simulate internal
# API requests directly through Python's execution layer.
client = TestClient(app)

def test_health_endpoint_returns_200():
    """Verifies that the deployment health probe responds with an operational status code."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "database_connected" in response.json()