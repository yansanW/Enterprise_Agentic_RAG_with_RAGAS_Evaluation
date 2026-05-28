# src/ingestion/pdf_parser.py
"""
Take the PDF reading and dynamic semantic chunking logic out of the app loop and wrap it inside a clean module.

"""

from langchain_community.document_loaders import PyPDFLoader

# from langchain_pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings  # pip install langchain-ollama
from src import config


class PDFIngestionEngine:
    def __init__(self):
        # 1. Initialize the embedding instance dynamically based on your config file
        if config.EMBEDDING_SOURCE == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GOOGLE_EMBEDDING, google_api_key=config.GOOGLE_API_KEY
            )
        elif config.EMBEDDING_SOURCE == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING, base_url=config.OLLAMA_URL
            )
        else:
            raise ValueError(f"Unknown embedding source: {config.EMBEDDING_SOURCE}")

        # 2. Select the strategy pattern for text chunking
        if config.SPLITTER_TYPE == "semantic":
            print("🔧 Initializing Dynamic Semantic Chunker...")
            self.splitter = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="percentile"
            )
        elif config.SPLITTER_TYPE == "static":
            print(
                f"🔧 Initializing Static Recursive Character Splitter ({config.CHUNK_SIZE} tokens)..."
            )
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
            )
        else:
            raise ValueError(f"Unknown splitter type: {config.SPLITTER_TYPE}")

    def process_pdf(self, file_path: str):
        """Loads and processes documents dynamically based on layout complexity rules."""
        # 3. Strategy Pattern Routing for Parsing
        if config.PARSER_STRATEGY == "multimodal_vlm":
            from src.ingestion.multimodal_parser import MultimodalParser

            # Pass your configured VLM engine or client wrapper here if needed
            parser = MultimodalParser()
            documents = parser.parse_document(file_path)
        else:
            # Fallback to plain character-text reader for simple layouts
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        # 4. Group all extracted document pages uniformly into semantic chunks
        chunks = self.splitter.split_documents(documents)
        return chunks
