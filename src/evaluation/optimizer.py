# src/evaluation/optimizer.py
import os
import sys
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

# --- CRITICAL RUNTIME GUARD: MOCK LEGACY LANGCHAIN ROUTE ---
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    mock_vertex_mod = ModuleType("langchain_community.chat_models.vertexai")
    mock_vertex_mod.ChatVertexAI = type("ChatVertexAI", (object,), {})  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_mod

import asyncio
import json
from datasets import Dataset
from ragas import evaluate

# Import the metric classes
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.factory import ModelFactory
from src.database import initialize_vectorstore
from src.pipeline import AgenticRAGCore
from src import config


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_relevance",
    "context_precision",
    "context_recall",
)


def _json_safe(value):
    """Convert pandas/numpy values into strict JSON-compatible values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _manifest_identity():
    manifest_path = Path(config.DATA_DIR) / "openrag_slice" / "slice_manifest.json"
    if not manifest_path.exists():
        return {
            "manifest_path": str(manifest_path.relative_to(config.BASE_DIR)),
            "manifest_sha256": None,
            "dataset_revision": None,
        }

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "manifest_path": str(manifest_path.relative_to(config.BASE_DIR)),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset": manifest.get("dataset"),
        "dataset_revision": manifest.get("dataset_revision"),
        "selection": manifest.get("selection"),
        "qa_count": manifest.get("qa_count"),
    }


def _save_results(score_results, aggregate_scores, run_timestamp):
    """Persist row-level RAGAS results and provenance after a completed run."""
    result_frame = score_results.to_pandas()
    per_question = [_json_safe(row) for row in result_frame.to_dict(orient="records")]

    excluded_outputs = []
    for question_index, row in enumerate(per_question):
        for metric_name in METRIC_NAMES:
            if metric_name in row and row[metric_name] is None:
                excluded_outputs.append(
                    {
                        "question_index": question_index,
                        "metric": metric_name,
                        "reason": "RAGAS returned a missing or non-finite score; no more specific judge reason was available.",
                    }
                )

    artifact = {
        "artifact_schema_version": 1,
        "run_timestamp": run_timestamp.isoformat(),
        "dataset_slice": _manifest_identity(),
        "per_question_results": per_question,
        "aggregate_summary": [
            {"metric": name, "score": _json_safe(score)}
            for name, score in aggregate_scores.items()
        ],
        "excluded_or_malformed_judge_outputs": {
            "count": len(excluded_outputs),
            "records": excluded_outputs,
        },
    }

    output_dir = Path(config.DATA_DIR) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"openrag_slice_60_{timestamp_slug}.json"
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"💾 Full evaluation results saved to {output_path}")
    return output_path


async def run_evaluation_suite():
    """
    Offline Optimization Suite. Reads evaluation metrics dynamically from disk
    and executes RAGAS alignment matrices over the active engine pipeline.
    """
    run_timestamp = datetime.now(timezone.utc)
    print("🧪 Initializing Evaluation Pipeline Components...")

    # 1. Resolve localized dataset file path strings
    dataset_path = config.GOLDEN_DATASET_PATH

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"❌ Aborting evaluation: Target payload missing at {dataset_path}"
        )

    # 2. Extract JSON benchmarks from file system matrix
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    # Boot up pipeline engine blocks
    vectorstore = initialize_vectorstore(persist_directory=config.DATA_DIR_Vectorstore)
    agent_system = AgenticRAGCore(vectorstore=vectorstore)

    # --- UNIFIED FACTORY INJECTION ---
    # We explicitly request clean wrappers from our factory. This bypasses
    # internal class introspection bugs and prevents script fracturing!
    ragas_llm = ModelFactory.get_ragas_llm()
    ragas_embeddings = ModelFactory.get_ragas_embeddings()
    # try:
    #     ragas_llm = LangchainLLMWrapper(agent_system.llm) 
    #     ragas_embeddings = LangchainEmbeddingsWrapper(vectorstore.embeddings)
    # except Exception as e:
    #     print(f"⚠️ Model extraction warning: {e}. Falling back to clean Factory calls.")
    #     # Seamless failover backup plan
    #     ragas_llm = LangchainLLMWrapper(ModelFactory.get_llm())
    #     ragas_embeddings = LangchainEmbeddingsWrapper(ModelFactory.get_embeddings())

    queries = []
    answers = []
    contexts = []
    ground_truths = []

    print(
        f"🏃‍♂️ Executing pipeline over {len(test_questions)} dynamic evaluation queries..."
    )

    query_delay = float(os.getenv("EVALUATION_QUERY_DELAY_SECONDS", "6.5"))

    for item_index, item in enumerate(test_questions):
        if item_index and query_delay > 0:
            await asyncio.sleep(query_delay)
        q = item["question"]
        queries.append(q)
        ground_truths.append(item["ground_truth"])

        # Invoke the active async pipeline execution path
        result = await agent_system.aexecute_pipeline(query=q, chat_history=[])
        answers.append(result.answer)
        raw_context = (
            result.retrieved_context
            if result.retrieved_context
            else ["No verified context retrieved."]
        )
        contexts.append(raw_context)
        if item_index == 0:
            print("🔎 Example raw retrieved contexts passed to RAGAS:")
            print(json.dumps(raw_context, indent=2, ensure_ascii=False))

    # 3. Restructure payload matrices into a Hugging Face Dataset format
    evaluation_dict = {
        "question": queries,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(evaluation_dict)

    print("📊 Computing RAGAS Semantic Alignment Metrics...")

    # --- INITIALIZE METRIC OBJECTS EXPLICITLY ---
    # We instantiate each metric class with your local llama3 wrappers
    initialized_metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # 4. Execute mathematical grading matrix over the dataset
    score_results = evaluate(dataset=dataset, metrics=initialized_metrics)

    print("\n=======================================================")
    print("📈 FINAL SYSTEM OPTIMIZATION SCORECARD")
    print("=======================================================")
    # Using .to_pandas().to_dict(orient="records") or iterating over scores:
    # If scores is a list of dictionaries, we can consolidate or iterate them safely:
    final_scores_dict = {}
    try:
        # Convert the evaluation result directly into a clean flat dictionary
        final_scores_dict = score_results.to_pandas().mean(numeric_only=True).to_dict()

        for metric_name, score in final_scores_dict.items():
            print(f"🔹 {metric_name.upper():<20} : {score:.4f}")

    except Exception:
        # Fallback tracking if pandas operations are restricted in your environment
        print(f"📊 Raw Evaluation Scores Result Container: {score_results.scores}")
    print("=======================================================\n")
    _save_results(score_results, final_scores_dict, run_timestamp)
    return score_results


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
