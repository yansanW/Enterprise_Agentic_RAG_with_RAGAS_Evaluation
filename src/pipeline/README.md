# Module 2: Agentic Routing Core and Structured Pipeline

## Overview

The pipeline module routes queries, reformulates retrieval queries using chat history, retrieves context, and asks the configured model for a schema-conforming response. Pydantic enforces the response structure; it does not independently establish whether model-generated claims or citations are correct.

---

## Architectural Design Patterns

### 1. Agentic Router

`chains.py` uses a model to select an execution track:

- **Conversational track (`CHAT`):** Greetings and similar inputs bypass retrieval and receive the configured canned response.
- **Knowledge retrieval track (`RETRIEVE`):** Factual and document-oriented queries proceed to vector retrieval and generation.

Routing is model-generated and may vary with the configured provider and model.

### 2. Conversational Query Reformulation

For the `RETRIEVE` track, the pipeline uses message history to rewrite contextual queries into standalone retrieval queries before accessing the vector store.

### 3. Structured Output Validation

`GuardedAnswerSchema` uses Pydantic to enforce the presence and types of `answer`, `is_supported_by_context`, and `citations`. This ensures schema conformance only. The support flag and citation content remain model-generated and prompt-directed; no deterministic post-generation validator checks their factual support, citation accuracy, or hallucinations.

---

## Technical Behavior

### Asynchronous Pipeline Execution

The pipeline exposes asynchronous methods (`aroute_query`, `aexecute_pipeline`) so provider and retrieval operations can be awaited by callers. This does not imply that every underlying operation is non-blocking: Chroma and session history use disk-backed storage, and the project has no concurrency or load benchmark.

### Schema-Conforming Responses

Structured model output gives API consumers a consistent set of typed fields. It prevents malformed response shapes when validation succeeds, but it is not independent factual verification and does not establish citation correctness.

---

## Tooling

- **`src/inspect_db.py`:** Inspects the configured Chroma store and reports chunk count, unique sources, ingestion timestamps, parser strategies, and a sample snippet. Run with `python -m src.inspect_db`.

---

## Testing & Mock Isolation

Tests in `src/tests/test_pipeline.py` cover provider-backed routing and generation paths, with retrieval behavior partially isolated through mocks and a temporary vector store. These tests are marked `network` and are skipped by the default suite.

Run `python -m pytest --run-network` with the required configured live services to exercise those paths. Even with the opt-in, the tests validate their stated routing and output-shape assertions; they do not independently verify generated factual or citation content.
