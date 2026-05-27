# src/evaluation/optimizer.py
import os
import sys
from types import ModuleType

# --- CRITICAL RUNTIME GUARD: MOCK LEGACY LANGCHAIN ROUTE ---
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    mock_vertex_mod = ModuleType("langchain_community.chat_models.vertexai")
    mock_vertex_mod.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_mod

import asyncio
import json
from datasets import Dataset
from ragas import evaluate

# Import the metric classes
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.database import initialize_vectorstore
from src.pipeline.chains import AgenticRAGCore
from src import config

async def run_evaluation_suite():
    """
    Offline Optimization Suite. Reads evaluation metrics dynamically from disk 
    and executes RAGAS alignment matrices over the active engine pipeline.
    """
    print("🧪 Initializing Evaluation Pipeline Components...")
    
    # 1. Resolve localized dataset file path strings
    dataset_path = config.GOLDEN_DATASET_PATH
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Aborting evaluation: Target payload missing at {dataset_path}")
        
    # 2. Extract JSON benchmarks from file system matrix
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    
    # Boot up pipeline engine blocks
    vectorstore = initialize_vectorstore()
    agent_system = AgenticRAGCore(vectorstore=vectorstore)

    # --- EXPLICIT COMPONENT WRAPPING ---
    try:
        ragas_llm = LangchainLLMWrapper(agent_system.llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(vectorstore.embeddings)
    except Exception as e:
        print(f"⚠️ Model extraction warning: {e}. Falling back to default initialization layers.")
        ragas_llm = None
        ragas_embeddings = None
    
    queries = []
    answers = []
    contexts = []
    ground_truths = []
    
    print(f"🏃‍♂️ Executing pipeline over {len(test_questions)} dynamic evaluation queries...")
    
    for item in test_questions:
        q = item["question"]
        queries.append(q)
        ground_truths.append(item["ground_truth"])
        
        # Invoke the active async pipeline execution path
        result = await agent_system.aexecute_pipeline(query=q, chat_history=[])
        answers.append(result.answer)
        contexts.append(result.citations if result.citations else ["No verified context retrieved."])
        
    # 3. Restructure payload matrices into a Hugging Face Dataset format
    evaluation_dict = {
        "question": queries,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(evaluation_dict)
    
    print("📊 Computing RAGAS Semantic Alignment Metrics...")
    
    # --- INITIALIZE METRIC OBJECTS EXPLICITLY ---
    # We instantiate each metric class with your local llama3 wrappers
    initialized_metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm)
    ]
    
    # 4. Execute mathematical grading matrix over the dataset
    score_results = evaluate(
        dataset=dataset,
        metrics=initialized_metrics
    )
    
    print("\n=======================================================")
    print("📈 FINAL SYSTEM OPTIMIZATION SCORECARD")
    print("=======================================================")
    # Using .to_pandas().to_dict(orient="records") or iterating over scores:
    # If scores is a list of dictionaries, we can consolidate or iterate them safely:
    try:
        # Convert the evaluation result directly into a clean flat dictionary
        final_scores_dict = score_results.to_pandas().mean(numeric_only=True).to_dict()
        
        for metric_name, score in final_scores_dict.items():
            print(f"🔹 {metric_name.upper():<20} : {score:.4f}")
            
    except Exception:
        # Fallback tracking if pandas operations are restricted in your environment
        print(f"📊 Raw Evaluation Scores Result Container: {score_results.scores}")
    print("=======================================================\n")
    return score_results

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())