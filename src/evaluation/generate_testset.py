# src/research_rag/generate_testset.py
"""
Test set generator for RAGAS evaluation.
Runs ONCE to generate Q&A pairs from your documents.
Output saved to data/eval_set.json and reused by evaluate.py.

Usage:
    python -m src.research_rag.generate_testset
    or via Makefile:
    make generate-testset
"""

import json
import logging
import os
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.testset import TestsetGenerator
# from ragas.testset.evolutions import multi_context, reasoning, simple
from langchain_core.documents import Document

from src import config
# from .ingest import load_local_papers, chunk_documents
from src.database import initialize_vectorstore
from src.factory import ModelFactory
from src.ingestion import PDFIngestionEngine, MultimodalParser


logger = logging.getLogger(__name__)


def generate_testset(
    documents: list,
    output_path: Path,
    test_size: int = 25,
) -> Path:
    """
    Generate a RAGAS test set from documents and save to JSON.

    Args:
        documents:   loaded + chunked langchain Documents
        output_path: where to save eval_set.json
        test_size:   number of Q&A pairs to generate

    Returns:
        path to the saved JSON file
    """
    logger.info(f"Generating {test_size} test questions from {len(documents)} chunks...")
    logger.info("This runs once and saves to disk — grab a coffee, takes ~5 minutes.")

    # --- CLEAN UNIFIED FACTORY CALLS ---
    # No more duplicate hardcoded ChatOllama blocks!
    generator_llm = ModelFactory.get_llm()
    critic_llm = ModelFactory.get_llm()
    embeddings = ModelFactory.get_embeddings()


    generator = TestsetGenerator(
        llm=generator_llm,
        # critic_llm=critic_llm,
        embedding_model=embeddings,
    )

    # 3. Generate your dataset using langchain documents
    testset = generator.generate_with_langchain_docs(documents, testset_size=test_size)

    # # Three question types — each tests a different failure mode
    # # simple:        "What is X?"            → tests basic retrieval
    # # reasoning:     "Why did X happen?"     → tests LLM understanding
    # # multi_context: answer spans 2+ chunks  → tests cross-chunk retrieval
    # testset = generator.generate_with_langchain_docs(
    #     documents=documents,
    #     test_size=test_size,
    #     distributions={
    #         simple: 0.5,
    #         reasoning: 0.25,
    #         multi_context: 0.25,
    #     },
    # )


    # Convert to plain dict list for portability
    df = testset.to_pandas()
    records = df[["question", "ground_truth", "evolution_type"]].to_dict(orient="records")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"Saved {len(records)} Q&A pairs to {output_path}")
    logger.info("Breakdown:")
    for evo_type, count in df["evolution_type"].value_counts().items():
        logger.info(f"  {evo_type}: {count}")

    return output_path


def load_or_generate_testset(output_path: Path, test_size: int = 25) -> list[dict]:
    """
    Load existing test set if it exists, otherwise generate it.
    This is what evaluate.py calls — it never needs to know which happened.

    Args:
        output_path: path to eval_set.json
        test_size:   only used if generating fresh

    Returns:
        list of dicts with keys: question, ground_truth, evolution_type
    """
    if os.path.exists(output_path):
        logger.info(f"Found existing test set at {output_path} — loading.")
        with open(output_path) as f:
            records = json.load(f)
        logger.info(f"Loaded {len(records)} Q&A pairs.")
        return records

    # No file found — generate it
    logger.info(f"No test set found at {output_path} — generating...")

    # Get docs from existing vectorstore — not re-loading from disk
    vectorstore = initialize_vectorstore()
    docs = vectorstore.get()  # returns all stored chunks as dicts

    # Convert ChromaDB format back to LangChain Documents
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(docs["documents"], docs["metadatas"])
    ]

    print(f"DEBUG: Retrieved {len(documents)} documents from vectorstore.")

    generate_testset(
        documents=documents,
        output_path=output_path,
        test_size=test_size,
    )

    with open(output_path) as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    output_path = config.GOLDEN_DATASET_PATH

    if output_path.exists():
        overwrite = input(
            f"\nTest set already exists at {output_path}.\n"
            f"Overwrite? This will delete your existing Q&A pairs. [y/N]: "
        ).strip().lower()

        if overwrite != "y":
            logger.info("Keeping existing test set. Exiting.")
            exit(0)

        output_path.unlink()
        logger.info("Deleted existing test set. Regenerating...")

    parser = MultimodalParser()
    documents = parser.parse_document(config.TEST_DOCUMENT_PATH)
    generate_testset(
        documents=documents,
        output_path=output_path,
        test_size=25,
    )

    print(f"\n✓ Test set saved to {output_path}")
    print("  Run `make eval` to score your pipeline against it.")
