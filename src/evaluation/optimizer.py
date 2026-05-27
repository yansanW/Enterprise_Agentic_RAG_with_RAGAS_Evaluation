# src/evaluation/optimizer.py
import os
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from src.database import initialize_vectorstore
from src.pipeline.chains import AgenticRAGCore
import json
import config


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
    
    # 4. Execute mathematical grading matrix over the dataset
    score_results = evaluate(
        dataset=dataset,
        metrics=[config.FAITHFULNESS_METRIC, 
                 config.ANSWER_RELEVANCE_METRIC, 
                 config.CONTEXT_PRECISION_METRIC, 
                 config.CONTEXT_RECALL_METRIC]
        
    )
    
    print("\n=======================================================")
    print("📈 FINAL SYSTEM OPTIMIZATION SCORECARD")
    print("=======================================================")
    for metric_name, score in score_results.items():
        print(f"🔹 {metric_name.upper():<20} : {score:.4f}")
    print("=======================================================\n")
    
    return score_results

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())