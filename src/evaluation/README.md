# Module 4: Automated Offline Optimization & Evaluation (RAGAS)

## Overview
The Evaluation module establishes an empirical, data-driven optimization framework for the RAG engine. By moving past anecdotal "vibe checks," this module leverages **RAGAS (Retrieval Augmented Generation Assessment)** alongside an LLM-as-a-Judge architecture to mathematically score pipeline generation across independent verification vectors using a decoupled, version-controlled testing dataset.

---

## The Four Core RAGAS Metrics

To completely isolate whether an issue stems from faulty database retrieval or model generation hallucination, the optimizer scores four distinct axes:

### 1. Faithfulness (Generation Quality)
* **What it measures:** The factual integrity of the generated response.
* **Mechanism:** It maps the generated `answer` directly against the retrieved `contexts`. If the model adds external facts or assumptions not explicitly found inside the text chunks, the faithfulness score drops, signaling a model hallucination.

### 2. Answer Relevance (Generation Quality)
* **What it measures:** How directly the generated answer addresses the user's core intent.
* **Mechanism:** It evaluates the `answer` against the initial `question` string, penalizing incomplete, rambling, or evasive responses, even if they are factually correct.

### 3. Context Precision (Retrieval Quality)
* **What it measures:** The signal-to-noise ratio of your retrieved vector chunks.
* **Mechanism:** It evaluates whether the most relevant information blocks are sorted at the very top of your retrieved `contexts` array, ensuring that processing tokens are not wasted on unhelpful data segments.

### 4. Context Recall (Retrieval Quality)
* **What it measures:** The completeness of the database search operation.
* **Mechanism:** It compares the retrieved `contexts` against your ground-truth `ground_truth` entry. If a critical detail required to fully answer the question is missing from the database pull, the recall score drops, signaling that your chunking strategies or search filters need tuning.

---

## Data-Driven Decoupling (`golden_dataset.json`)

To allow non-technical project owners or developers to append test evaluation cases without editing execution paths, test boundaries are decoupled into a clean JSON array structure:

```json
[
  {
    "question": "Target query text statement.",
    "ground_truth": "The expected absolute correct target answer payload."
  }
]