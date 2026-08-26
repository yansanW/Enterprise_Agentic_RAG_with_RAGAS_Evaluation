# Module 3: Headless FastAPI Gateway Layer

## Overview

The API module exposes the RAG pipeline through FastAPI. Its asynchronous endpoint awaits pipeline execution and maintains session history in a disk-backed SQLite database. The repository does not include load or concurrency benchmarks, and some underlying storage operations are synchronous disk I/O.

---

## Architectural Design Patterns

### 1. Headless API Gateway

The module separates user-interface concerns from backend orchestration. Its JSON REST interface can be consumed by web, mobile, or automation clients.

### 2. Session-Isolated Stateful Memory Tracking

The `/api/v1/query` route tracks conversation history by `session_id`:

- **Identification:** Each request contains a session identifier.
- **Retrieval:** The route reads matching history from the disk-backed SQLite store.
- **Persistence:** After the pipeline returns a schema-conforming generated answer, the route stores the query and output before responding.

The stored answer is model-generated. Schema validation does not independently verify its factual content or citations.

### 3. Serialization and Structural Contracts

Pydantic validates request and response structure:

- **Request contract (`QueryRequest`):** Enforces the accepted fields, types, and defaults.
- **Response contract (`GuardedAnswerSchema`):** Enforces the output fields and their types. This is structural validation rather than fact-checking.

---

## Endpoint Specifications

### System Health Probe

- **Route:** `GET /health`
- **Purpose:** Reports whether the database and pipeline objects were constructed.
- **Limitation:** Startup failures can be caught while the endpoint still returns HTTP success. Its flags do not perform an end-to-end request or verify Ollama, Cohere, embedding-provider, or generation readiness. The endpoint can therefore report success while the pipeline is inactive and should not be treated as a sufficient load-balancer readiness check.
- **Sample response:**

  ```json
  {
    "status": "healthy",
    "database_connected": true,
    "pipeline_active": true
  }
  ```

Consumers should inspect the component flags and use a separate end-to-end readiness check when deployment traffic must depend on provider availability.
