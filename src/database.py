# src/database.py
'''
vector store setup should live separately so both your ingestion workers and your API chains can query it independently.
'''

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config

def get_vectorstore(chunks=None):
    """Initializes or loads an existing Chroma instance."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=config.GOOGLE_API_KEY
    )
    
    if chunks:
        # If chunks are passed, create a new collection
        return Chroma.from_documents(chunks, embeddings)
    else:
        # Otherwise, point to an in-memory or persisted instance
        return Chroma(embedding_function=embeddings)