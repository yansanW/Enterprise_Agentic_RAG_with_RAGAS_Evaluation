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


def test_query_endpoint_payload_rejection():
    """
    Verifies that the server throws an explicit HTTP 422 validation failure 
    if a client sends a malformed or missing JSON request body.
    """
    # Sending an empty JSON object {} instead of {"question": "text"}
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422  # HTTP 422 indicates an Unprocessable Entity