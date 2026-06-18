# src/factory.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI
from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


from src import config

class ModelFactory:
    """
    Centralized Enterprise Factory Pattern.
    Unifies standard LangChain runtime components and modern Ragas 0.4+ judge factories.
    """
    
    # =========================================================================
    # SECTION 1: LANGCHAIN COMPONENTS (For core server pipeline & ingestion)
    # =========================================================================
    
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
        
    
    # =========================================================================
    # SECTION 2: MODERN NATIVE RAGAS 0.4+ CHANNELS (For Evaluators & Dataset Generators)
    # =========================================================================

    @staticmethod
    def get_ragas_llm():
        """Wraps our standard pipeline LLM cleanly for Ragas judges."""
        return LangchainLLMWrapper(ModelFactory.get_llm())
        # """Instantiates native Ragas 0.4+ Judge LLM clients."""
        # if config.LLM_SOURCE == "ollama":
        #     # Explicitly force clean base URL configuration parsing boundaries
        #     clean_url = config.OLLAMA_URL.rstrip('/')
        #     if not clean_url.endswith('/v1'):
        #         clean_url = f"{clean_url}/v1"

        #     ollama_client = OpenAI(
        #         api_key="ollama",
        #         base_url=clean_url
        #     )
        #     return llm_factory(
        #         model=config.OLLAMA_LLM, 
        #         provider="openai", 
        #         client=ollama_client
        #     )
            
        # elif config.LLM_SOURCE == "google":
        #     gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
        #     return llm_factory(
        #         model=config.GOOGLE_LLM, 
        #         provider="google", 
        #         client=gemini_client
        #     )
        # else:
        #     raise ValueError(f"Ragas LLM Factory unconfigured for: {config.LLM_SOURCE}")

    @staticmethod
    def get_ragas_embeddings():
        return LangchainEmbeddingsWrapper(ModelFactory.get_embeddings())

        # """
        # Instantiates native Ragas 0.4+ Judge Embedding Clients.
        # This provides a BaseRagasEmbeddings object to completely clear embed_query errors!
        # """
        # if config.EMBEDDING_SOURCE == "ollama":
        #     clean_url = config.OLLAMA_URL.rstrip('/')
        #     if not clean_url.endswith('/v1'):
        #         clean_url = f"{clean_url}/v1"

        #     ollama_client = OpenAI(
        #         api_key="ollama",
        #         base_url=clean_url
        #     )
        #     # Use provider="openai" to pipe native embeddings to local Ollama vector modules
        #     return embedding_factory(
        #         model=config.OLLAMA_EMBEDDING,
        #         provider="openai",
        #         client=ollama_client
        #     )
            
        # elif config.EMBEDDING_SOURCE == "google":
        #     gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
        #     return embedding_factory(
        #         model=config.GOOGLE_EMBEDDING,
        #         provider="google",
        #         client=gemini_client
        #     )
        # else:
        #     raise ValueError(f"Ragas Embedding Factory unconfigured for: {config.EMBEDDING_SOURCE}")