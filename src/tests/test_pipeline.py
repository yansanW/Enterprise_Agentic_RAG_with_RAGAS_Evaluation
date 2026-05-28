# src/tests/test_pipeline.py
import pytest
from src.database import initialize_vectorstore
from src.pipeline import AgenticRAGCore
from langchain_core.documents import Document


@pytest.fixture
def mock_pipeline(tmp_path, monkeypatch):
    """Fixture to instantly spin up an isolated memory database and pipeline for testing."""
    sample_docs = [
        Document(
            page_content="The company Alpha Corp was founded in 2026 by an AI engineer.",
            metadata={"source": "test.pdf", "page": 1},
        )
    ]
    db_dir = str(tmp_path / "test_pipeline_store")
    vs = initialize_vectorstore(chunks=sample_docs, persist_directory=db_dir)

    # --- MOCK COHERE ASYNC NETWORK LAYER ---
    from langchain_classic.retrievers import ContextualCompressionRetriever

    # FIX: We add 'async' here so the returned list object is correctly awaitable!
    async def mock_compress_async(*args, **kwargs):
        return sample_docs

    # Map the asynchronous mock to the background retriever lifecycle
    monkeypatch.setattr(
        ContextualCompressionRetriever, "_get_relevant_documents", mock_compress_async
    )
    monkeypatch.setattr(
        ContextualCompressionRetriever, "_aget_relevant_documents", mock_compress_async
    )
    # ---------------------------------------

    return AgenticRAGCore(vectorstore=vs)


# Mark tests with @pytest.mark.asyncio so pytest knows how to await them!
@pytest.mark.asyncio
async def test_router_selects_chat_for_greetings(mock_pipeline):
    decision = await mock_pipeline.aroute_query("Good morning! How are things?")
    assert decision == "CHAT"


@pytest.mark.asyncio
async def test_router_selects_retrieve_for_factual_queries(mock_pipeline):
    decision = await mock_pipeline.aroute_query(
        "Who founded Alpha Corp and in what year?"
    )
    assert decision == "RETRIEVE"


@pytest.mark.asyncio
async def test_end_to_end_guarded_generation(mock_pipeline):
    result = await mock_pipeline.aexecute_pipeline("Tell me about Alpha Corp.")
    assert hasattr(result, "answer")
    assert isinstance(result.is_supported_by_context, bool)
