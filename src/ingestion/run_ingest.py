# src/ingestion/run_ingest.py
from datetime import datetime
import os
import sys
import logging
from src.ingestion.pdf_parser import PDFIngestionEngine
from src.database import initialize_vectorstore
from src import config

# Initialize simple log routing for clarity
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline_ingestion():
    """
    Offline Database Hydration Loop. 
    Converts target documents into semantic vector math and populates the local disk vault.
    """
    target_file = config.DATA_DIR_Vectorstore_test
    
    if not os.path.exists(target_file):
        logger.error(f"❌ Target document missing at {target_file}. Please check your path configurations.")
        sys.exit(1)
        
    logger.info(f"🚀 Initializing PDF Ingestion Engine Layer...")
    ingestion_engine = PDFIngestionEngine()
    
    logger.info(f"📄 Processing layout structures and chunking target file: {target_file}...")
    # This invokes pdf_parser -> which switches to MultimodalParser if config dictates!
    # 1. Parse your chunks
    chunks = ingestion_engine.process_pdf(target_file)

    
    if not chunks:
        logger.warning("⚠️ No text or tabular chunks were extracted from the document. Aborting database storage.")
        return

    # 2. Dynamically stamp the current processing time onto every single chunk metadata layer
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for chunk in chunks:
        # Append clear audit properties directly onto the LangChain Document object
        chunk.metadata["ingested_at"] = current_time_str
        chunk.metadata["parser_strategy"] = config.PARSER_STRATEGY
        
    logger.info(f"🧬 Extracted {len(chunks)} structural components. Syncing with vector vault storage...")
    
    # Pass your split chunks directly to initialize_vectorstore() to trigger the seeding block!
    # 3. Commit to database
    vectorstore = initialize_vectorstore(chunks=chunks,
                                         persist_directory=os.path.join(config.BASE_DIR, "data/vectorstore_test"))
    
    logger.info("=======================================================")
    logger.info("✅ ENTERPRISE VECTOR VAULT HYDRATION COMPLETE on %s", current_time_str)
    logger.info(f"   Test Document Path: {target_file}")
    logger.info(f"   Storage Location: {os.path.join(config.BASE_DIR, "data/vectorstore_test")}")
    logger.info(f"   Total Active Records Added: {len(chunks)}")
    logger.info("=======================================================")

if __name__ == "__main__":
    run_pipeline_ingestion()