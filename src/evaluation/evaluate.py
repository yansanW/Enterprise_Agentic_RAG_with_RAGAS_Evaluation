# src/research_rag/evaluate.py
"""
RAGAS evaluation runner.
Loads test set from data/eval_set.json (generates it if missing).
Run after any pipeline change to measure improvement.

Usage:
    python -m src.research_rag.evaluate
    or via Makefile:
    make eval
"""

import json
import logging
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from .config import settings
from .ingest import load_vectorstore
from .retriever import build_retriever
from .chain import build_chain
from .generate_testset import load_or_generate_testset

logger = logging.getLogger(__name__)


def run_evaluation(output_csv: Path | None = None) -> dict:
    """
    Run RAGAS evaluation against the saved test set.

    Args:
        output_csv: optional path to save per-question results

    Returns:
        dict of mean metric scores
    """

    # ── Load test set (generates if data/eval_set.json missing) ──────────────
    test_set = load_or_generate_testset(settings.eval_set_path)
    logger.info(f"Evaluating against {len(test_set)} questions...")

    # ── Load pipeline (same as app.py uses) ──────────────────────────────────
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa_chain = build_chain(retriever)

    # ── Run each question through the pipeline ────────────────────────────────
    questions, ground_truths, answers, contexts = [], [], [], []

    for i, item in enumerate(test_set):
        logger.info(f"  [{i+1}/{len(test_set)}] {item['question'][:60]}...")

        result = qa_chain.invoke({
            "input": item["question"],
            "chat_history": [],          # fresh history per question for eval
        })

        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["context"]])

    # ── Wrap models for RAGAS internal use ───────────────────────────────────
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)
    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    eval_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # ── Run RAGAS ─────────────────────────────────────────────────────────────
    logger.info("Running RAGAS scoring...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = results.to_pandas()

    # ── Print results ─────────────────────────────────────────────────────────
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    print("\n" + "=" * 55)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 55)

    mean_scores = {}
    for metric in metrics:
        score = df[metric].mean()
        mean_scores[metric] = round(score, 3)
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "✓" if score >= 0.75 else "✗"
        print(f"  {status} {metric:<22} {bar}  {score:.3f}")

    print("\nWhat low scores mean:")
    print("  faithfulness low      → LLM hallucinating beyond chunks")
    print("  answer_relevancy low  → fix system prompt")
    print("  context_recall low    → retriever missing chunks, try reranker or larger k")
    print("  context_precision low → too much noise retrieved, reduce k or add reranker")
    print("=" * 55)

    # ── Save detailed results ─────────────────────────────────────────────────
    if output_csv is None:
        output_csv = Path("data/ragas_results.csv")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")
    print("Tip: open in a spreadsheet to see which specific questions scored low.\n")

    return mean_scores


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    run_evaluation()
