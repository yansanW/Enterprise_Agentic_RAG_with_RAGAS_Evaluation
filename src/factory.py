# src/factory.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

class ModelFactory:
    """
    Centralized Enterprise Factory. The single source of truth for instantiating 
    LLMs, Embeddings, and Splitters across Ingestion, Pipelines, and Testing.
    """
    
    @staticmethod
    def get_llm():
        """Instantiates the exact model provider chosen in configs."""
        if config.LLM_SOURCE == "google":
            print(f"🤖 [Factory] Initializing Cloud LLM Core: {config.GOOGLE_LLM}")
            return ChatGoogleGenerativeAI(
                model=config.GOOGLE_LLM,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=config.LLM_TEMPERATURE,
            )
        elif config.LLM_SOURCE == "ollama":
            print(f"🤖 [Factory] Initializing Local Sovereign LLM Core: {config.OLLAMA_LLM}")
            return ChatOllama(
                model=config.OLLAMA_LLM,
                base_url=config.OLLAMA_URL,
                temperature=config.LLM_TEMPERATURE,
            )
        else:
            raise ValueError(f"Unsupported LLM provider configuration: {config.LLM_SOURCE}")

    @staticmethod
    def get_embeddings():
        """Instantiates the embedding instance dynamically based on configs."""
        if config.EMBEDDING_SOURCE == "google":
            return GoogleGenerativeAIEmbeddings(
                model=config.GOOGLE_EMBEDDING, google_api_key=config.GOOGLE_API_KEY
            )
        elif config.EMBEDDING_SOURCE == "ollama":
            return OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING, base_url=config.OLLAMA_URL
            )
        else:
            raise ValueError(f"Unknown embedding source: {config.EMBEDDING_SOURCE}")

    @staticmethod
    def get_text_splitter(embeddings):
        """Selects the strategy pattern for text chunking."""

        if config.SPLITTER_TYPE == "semantic":
            print("🔧 [Factory] Initializing Dynamic Semantic Chunker...")
            return SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        elif config.SPLITTER_TYPE == "static":
            print(f"🔧 [Factory] Initializing Static Recursive Character Splitter ({config.CHUNK_SIZE} tokens)...")
            return RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
            )
        else:
            raise ValueError(f"Unknown splitter type: {config.SPLITTER_TYPE}")