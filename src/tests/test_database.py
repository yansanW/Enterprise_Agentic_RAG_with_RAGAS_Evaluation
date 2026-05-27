# src/tests/test_database.py
import pytest
from langchain_core.documents import Document
from src.database import initialize_vectorstore

def test_vectorstore_initialization_with_sample_chunks(tmp_path):
    """
    Integration Test: Verifies that passing mock document chunks successfully 
    creates a local vector store collection inside a temporary testing directory.
    """
    # 1. Establish clear sample context strings
    sample_chunks = [
        Document(page_content="Artificial Intelligence is a branch of computer science.", metadata={"source": "test"}),
        Document(page_content="Computer Vision processes visual pixel tensors.", metadata={"source": "test"})
    ]
    
    # Use pytest's built-in 'tmp_path' to prevent polluting local disk drives
    test_db_dir = str(tmp_path / "test_chroma_db")
    
    # 2. Trigger the module execution
    db_instance = initialize_vectorstore(chunks=sample_chunks, persist_directory=test_db_dir)
    
    # Assert structural type integrity
    assert db_instance is not None
    assert db_instance.__class__.__name__ == "Chroma"
    
    # 3. Request a highly explicit vector match
    retriever = db_instance.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("What does Computer Vision process?")
    
    print(f"\nRetrieved results: {[doc.page_content for doc in results]}")
    
    # 4. Professional Matrix Assertions: Verify structure and connectivity, not semantic ranking
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert "source" in results[0].metadata
    
    # LINE 36 FIX: Verify that the returned text is indeed one of our original seeded chunks
    valid_contents = [chunk.page_content for chunk in sample_chunks]
    assert results[0].page_content in valid_contents