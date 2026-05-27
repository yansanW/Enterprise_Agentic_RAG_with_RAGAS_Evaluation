# Module 2: Agentic Routing Core & Guardrail Pipeline

## Overview
The Pipeline module serves as the cognitive orchestration layer of the RAG engine. Instead of passing user queries blindly to vector storage indices, this component implements an intelligent triage node to determine the optimal execution track, resolves context across multi-turn chats, and enforces strict structural integrity on the generation output.

---

## Architectural Design Patterns

### 1. The Agentic Router Pattern
To optimize processing latency and save token overhead, `chains.py` implements a predictable routing gatekeeper using low-temperature models:
* **Conversational Track (`CHAT`):** Casual inputs, greetings, and platform system prompts bypass the database entirely, returning immediate lightweight responses.
* **Knowledge Retrieval Track (`RETRIEVE`):** Technical queries, data metric requests, and document lookups are dynamically routed to query the vector store index.

### 2. Conversational Query Reformulation Node
To resolve pronoun drift in multi-turn conversations (e.g., a user asking *"When was it founded?"* after discussing *"Alpha Corp"*), the pipeline intercepts the `RETRIEVE` track. It evaluates past message history strings and rewrites vague queries into standalone, detailed search queries optimized for vector database lookup precision before touching disk storage.

### 3. Deterministic Output Guardrails (Structural Modeling)
To eliminate raw text drift and mitigate model hallucinations, we decouple the generation payload from standard text streams:
* **Pydantic Schema Tracking:** We bind a strict data schema interface (`GuardedAnswerSchema`) directly to the LLM decoding layer. 
* **Factual Verification Constraints:** The model is forced to explicitly evaluate its own context coverage (`is_supported_by_context`) and isolate exact verification strings (`citations`), ensuring that the final output functions as a predictable, structured data object.

---

## Technical Justifications

### Asynchronous Pipeline Execution
The pipeline exposes asynchronous methods (`aroute_query`, `aexecute_pipeline`) to handle retrieval and generation loops natively. This ensures that I/O operations (like database lookups and model calls) do not block Python's main event thread, allowing the architecture to handle concurrent traffic seamlessly when exposed via a web API.

### Structural Schema Enforcement vs. Raw Text Streaming
In a production headless architecture, receiving conversational paragraphs like *"Based on the documents, I think..."* breaks backend parsing models. Forcing the model to speak natively in verified JSON schemas ensures absolute system stability and enables automated pipeline routing.

---

## Testing & Mock Isolation
The routing chains, query rewriting mechanics, and data schemas are continuously verified using automated unit and integration tests inside `tests/test_pipeline.py`. 

By leveraging isolated temporary directory allocations (`tmp_path`) and `monkeypatch` overrides for external APIs, the suite validates state decision trees and async flows without requiring active outbound cloud tokens or local tensor weight loading during testing cycles.