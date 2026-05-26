# tests/test_pipeline.py
'''
Testing Connectively (Integration Testing)
an automated script that calls your ingestion engine, saves the result into 
an in-memory database, 
builds a retrieval chain, and 
evaluates the pipeline's output connectivity.
'''

import os
import pytest
from src.ingestion.pdf_parser import PDFIngestionEngine
from src.database import get_vectorstore
from src.pipeline.chains import build_agentic_rag_chain

def test_end_to_end_pipeline_connectivity():
    """
    Integration Test: Verifies that data flows completely from 
    Ingestion -> Vector Storage -> Retrieval Chain without manual UI steps.
    """
    # 1. Simulate the data ingestion step on a tiny sample document
    sample_pdf_path = "data/raw_docs/sample_test.pdf"
    
    # Guard against running tests without a valid test file
    if not os.path.exists(sample_pdf_path):
        pytest.skip("Test PDF not present, skipping integration validation.")
        
    engine = PDFIngestionEngine()
    chunks = engine.process_pdf(sample_pdf_path)
    assert len(chunks) > 0, "Ingestion engine failed to generate text chunks."
    
    # 2. Test connectivity to the database component
    vectorstore = get_vectorstore(chunks)
    
    # 3. Test connectivity to the core LangChain agentic model pipeline
    rag_chain = build_agentic_rag_chain(vectorstore)
    
    # 4. Invoke the connected system to prove the components communicate
    response = rag_chain.invoke({"input": "What is discussed on page 1?", "chat_history": []})
    
    # Validate the data matrix structure of the connected response
    assert "answer" in response
    assert "context" in response
    assert len(response["context"]) > 0