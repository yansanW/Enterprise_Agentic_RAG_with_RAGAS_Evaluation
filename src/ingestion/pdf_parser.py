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

from src.factory import ModelFactory
from src import config


class PDFIngestionEngine:
    def __init__(self):
        # The factory handles the logic dynamically!
        self.embeddings = ModelFactory.get_embeddings()
        self.splitter = ModelFactory.get_text_splitter(self.embeddings)
        
        
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
