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


from unittest.mock import AsyncMock, patch
from src.pipeline.schemas import GuardedAnswerSchema

def test_query_endpoint_success_path():
    """
    Integration Test: Verifies that a well-formed query request triggers 
    the pipeline correctly and returns a structured JSON answer payload.
    """
    # 1. Prepare a mock structured response matching our Pydantic schema contract
    mock_payload = GuardedAnswerSchema(
        answer="The company Alpha Corp was founded in 2026 by an AI engineer.",
        is_supported_by_context=True,
        citations=["founded in 2026 by an AI engineer"]
    )
    
    # 2. Patch both the async pipeline executor and get_session_history
    with patch("src.api.main.agent_system.aexecute_pipeline", new_callable=AsyncMock) as mock_pipeline, \
         patch("src.api.main.get_session_history") as mock_get_session_history:
        mock_pipeline.return_value = mock_payload

        # Mock session_history object with messages=[]
        class DummySessionHistory:
            messages = []
            def add_user_message(self, msg): pass
            def add_ai_message(self, msg): pass

        mock_get_session_history.return_value = DummySessionHistory()

        # 3. Simulate a valid HTTP POST request
        request_data = {
            "question": "Who founded Alpha Corp?",
            "session_id": "test_user_session_99"
        }
        response = client.post("/api/v1/query", json=request_data)

        # 4. Assert response validation metrics
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["answer"] == mock_payload.answer
        assert response_json["is_supported_by_context"] is True
        assert len(response_json["citations"]) == 1

        # Verify our app system actually triggered the pipeline with the correct inputs
        mock_pipeline.assert_called_once_with(
            query="Who founded Alpha Corp?", 
            chat_history=[]
        )

        # 1. Verify call execution count structural integrity
        mock_pipeline.assert_called_once()

        # 2. Dynamically extract the exact arguments passed to the mock call
        call_kwargs = mock_pipeline.call_args.kwargs

        # 3. Verify the query string matches perfectly
        assert call_kwargs["query"] == "Who founded Alpha Corp?"

        # 4. Verify that chat_history was passed down as a list container type
        assert isinstance(call_kwargs["chat_history"], list)