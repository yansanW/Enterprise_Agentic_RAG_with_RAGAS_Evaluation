# src/database.py
"""
vector store setup should live separately so both your ingestion workers and your API chains can query it independently.
"""


from typing import Optional
import os

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
)  # High-performance SQL history tracker
from src.factory import ModelFactory
from src import config


def _get_embedding_client():
    """Internal helper to initialize the embedding model dictated by configurations."""
    return ModelFactory.get_embeddings()



def initialize_vectorstore(chunks=None, persist_directory: Optional[str] = None):
    """
    Initializes a Chroma vector database collection.

    If chunks are provided, it generates embeddings and seeds a new/existing collection.
    If chunks are None, it loads the existing vector database from the disk directory.
    """
    embeddings = _get_embedding_client()

    # Define a clean persistence storage index folder inside your data directory
    if persist_directory is None:
        persist_directory = os.path.join(config.DATA_DIR, "vectorstore")

    if chunks:
        print(
            f"📦 Seeding Chroma DB at {persist_directory} with {len(chunks)} embedded vector chunks..."
        )
        return Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=persist_directory
        )
    else:
        print(
            f"🔍 Loading existing Chroma DB storage index from {persist_directory}..."
        )
        return Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )


# --- NEW: STATEFUL CONVERSATIONAL MEMORY MANAGER ---
def get_session_history(session_id: str):
    """
    Factory function to retrieve or instantiate a unique persistent chat history
    session inside a localized SQLite relational database.
    """
    db_path = os.path.join(config.DATA_DIR, "chat_history.db")
    connection_string = f"sqlite:///{db_path}"

    # SQLChatMessageHistory automatically sets up the relational tables
    # on disk if they do not exist yet!
    return SQLChatMessageHistory(session_id=session_id, connection=connection_string)
