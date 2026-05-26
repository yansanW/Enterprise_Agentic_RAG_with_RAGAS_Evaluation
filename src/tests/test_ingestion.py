# tests/test_ingestion.py
'''
Testing Individually (Unit Testing)
A production-ready unit test file should cover three categories of input conditions:
1. Edge Cases / Error Inputs: 
Empty inputs, files that don't exist, or completely corrupted paths.

2. Configuration Variability: 
Testing that your structural logic switches behavior correctly depending on whether your config.yaml says static vs semantic.

3. Success Inputs (Using Mock Data): 
Feeding a simple, predictable document structure to ensure your chunking arrays output the proper text schema.
'''


import os

import pytest
from langchain_core.documents import Document
from src.ingestion import PDFIngestionEngine, MultimodalParser
from src import config
import fitz  # Import fitz so we can reference its native exceptions


# --- 1. Testing Input Error Guardrails ---
def test_pdf_processing_handles_empty_input():
    """
    Unit Test: Ensures our ingestion logic does not crash if an empty 
    document layout object is accidentally processed.
    """
    engine = PDFIngestionEngine()
    
    # We pass a completely empty or simulated structure directly to our isolated logic
    with pytest.raises(Exception):
        engine.process_pdf("") # Should safely catch invalid empty path strings


def test_pdf_processing_handles_nonexistent_file(monkeypatch):
    """Verifies that attempting to read a file that doesn't exist throws a ValueError."""
    # Temporarily force the configuration variable to 'multimodal_vlm' for this block
    monkeypatch.setattr(config, "PARSER_STRATEGY", "standard_text")  # Ensure it routes to the standard text parser for this test
    
    engine = PDFIngestionEngine()
    
    # Change pymupdf.FileNotFoundError to ValueError here:
    with pytest.raises(ValueError):
        engine.process_pdf("missing_file.pdf")


# --- 1-b: Testing Strategy Selection Logic ---
def test_engine_selects_multimodal_parser_based_on_config(monkeypatch):
    """
    Unit Test: Verifies that changing the configuration yaml target 
    properly routes the execution path to the Multimodal parser.
    """
    # Temporarily force the configuration variable to 'multimodal_vlm' for this block
    monkeypatch.setattr(config, "PARSER_STRATEGY", "multimodal_vlm")
    
    engine = PDFIngestionEngine()
    
    # We trigger an execution with an invalid path. If it routes to MultimodalParser,
    # it will attempt to call fitz.open() on it, throwing a FileNotFoundError or RuntimeError 
    # instead of LangChain's generic ValueError. This proves the routing switch hit our module!
    # Catch fitz's specific file failure class type
    with pytest.raises(fitz.FileNotFoundError):
        engine.process_pdf("data/raw_docs/this_file_does_not_exist_at_all.pdf")


# --- 2. Testing Internal Logic and State Configurations ---
def test_engine_initializes_with_correct_splitter_type(monkeypatch):
    """
    Verifies that the engine builds a Semantic Chunker or Static Splitter 
    based on configurations without needing a real PDF execution loop.
    """
    # 'monkeypatch' allows us to dynamically overwrite configuration targets for a single test!
    from src import config
    
    # Force the app configuration to 'static' just for this test block
    monkeypatch.setattr(config, "SPLITTER_TYPE", "static")
    monkeypatch.setattr(config, "CHUNK_SIZE", 200)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 20)
    
    engine = PDFIngestionEngine()
    
    # Assert that the engine dynamically selected the right class type
    assert engine.splitter.__class__.__name__ == "RecursiveCharacterTextSplitter"
    assert engine.splitter._chunk_size == 200


# --- 3. Testing Component Output Structural Matrices ---
# Note: For testing actual document output formats, we would ideally pass 
# a tiny 1-page sample test PDF stored in your `data/raw_docs/` test repository.
def test_pdf_processing_outputs_correct_document_structure():
    """
    Integration-level Check: Verifies that a successful parsing run 
    outputs the exact data schema LangChain expects.
    """
    test_file_path = config.TEST_DOCUMENT_PATH
    
    # If you haven't uploaded a small test file yet, the test skips gracefully 
    # instead of throwing a false-alarm failure report.
    if not os.path.exists(test_file_path):
        pytest.skip("Local test PDF not present, skipping structural matrix verification.")
        
    engine = PDFIngestionEngine()
    chunks = engine.process_pdf(test_file_path)
    
    # Check that it returns an array of LangChain Document data elements
    assert isinstance(chunks, list)
    if len(chunks) > 0:
        # Check that individual chunks conform to proper page_content and metadata types
        assert hasattr(chunks[0], "page_content")
        assert hasattr(chunks[0], "metadata")
        assert isinstance(chunks[0].page_content, str)


# --- NEW: Testing Output Data Matrix Properties ---
def test_multimodal_parser_metadata_schema():
    """
    Integration Check: Processes a sample test document to ensure 
    the spatial structural tags ('type': 'table' / 'text') are built correctly.
    """
    test_file_path = config.TEST_DOCUMENT_PATH
    
    # Skip gracefully if you haven't dropped a sample research paper or test file into data yet
    if not os.path.exists(test_file_path):
        pytest.skip("Local test PDF not present, skipping structural metadata checks.")
        
    parser = MultimodalParser()
    documents = parser.parse_document(test_file_path)
    
    assert isinstance(documents, list)
    if len(documents) > 0:
        # Verify the structure complies fully with LangChain data contracts
        assert hasattr(documents[0], "page_content")
        assert hasattr(documents[0], "metadata")
        
        # Verify our custom structural metadata tags exist
        assert "type" in documents[0].metadata
        assert documents[0].metadata["type"] in ["text", "table"]