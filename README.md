# Enterprise-Agentic-RAG-with-RAGAS-Evaluation

## What This Prototype Implements

This repository is an engineering prototype for a headless RAG API. It combines PDF text extraction, table detection, persistent Chroma retrieval, intent routing, structured LLM output, and RAGAS evaluation. It is not production-ready.

### Ingestion

- `standard_text` uses `PyPDFLoader` and then the configured static or semantic splitter.
- `multimodal_vlm` uses PyMuPDF to extract page text and detect tables. The current table content is a fixed placeholder; images are not interpreted by a vision-language model.
- Chunks carry source, page, and text/table metadata. The implementation does not reconstruct or preserve the original spatial layout.

### Routing and generation

- An LLM classifies each request as `RETRIEVE` or `CHAT`.
- `RETRIEVE` queries the vector store and asks the configured LLM for a `GuardedAnswerSchema`.
- `CHAT` intentionally bypasses retrieval and returns a fixed greeting. It is not a general conversational track.
- Pydantic enforces the response shape (`answer`, `is_supported_by_context`, and `citations`). The support flag and citations are model-generated and are not independently verified or fact-checked.

### Evaluation

The RAGAS suite reads a saved golden dataset and evaluates the active configured pipeline. Its judge LLM and embeddings come from the selected Google or Ollama providers; local Ollama execution is optional, not the default guarantee. Default tests mock or skip provider-dependent paths. Passing mocked tests does not demonstrate the quality of a live local model.

## Roadmap / Known Limitations

This project keeps an enterprise RAG architecture as its direction, but the current implementation has important gaps:

- No authentication, authorization, rate limiting, telemetry, database migrations, or production operations tooling.
- No VLM integration, image understanding, chart-accuracy measurement, audio ingestion, or Whisper transcription. Those are roadmap items.
- Table detection exists, but table transcription is currently placeholder content.
- Source/page/type metadata survives ingestion; spatial relationships and visual layout do not.
- Structured output constrains shape only. Citation validation and independent fact verification remain future work.
- `CHAT` is an intentional canned-response bypass rather than open-ended conversation.
- Async interfaces are used around the pipeline, but Chroma and SQLite are disk-backed and the system has not been load-tested. No concurrency or scale benchmark is claimed.
- Provider-backed integration tests require an explicit opt-in and valid local/cloud credentials.
- The API can report healthy while the RAG pipeline is inactive; inspect the `database_connected` and `pipeline_active` fields in `/health`.

## Retrieval Pipeline

For `RETRIEVE` requests, `AgenticRAGCore` runs this sequence:

1. Chroma performs the configured base search. The checked-in configuration uses Maximum Marginal Relevance (MMR), fetching `fetch_k: 20` candidates and returning `base_k: 10` candidates.
2. `CohereRerank` scores that candidate set with `rerank-v3.5` and keeps `rerank_top_n: 3` chunks.
3. The selected LLM receives those chunks as context and returns the structured answer.

The values live under `retrieval` in `configs/config.yaml`. MMR diversifies the initial result pool; Cohere reranking is a separate hosted API call and always requires `COHERE_API_KEY` for the current pipeline. Set `search_type: similarity` to use similarity search instead of MMR. There is currently no no-reranker mode.

## Provider Matrix

LLM and embedding providers are selected independently through `provider.llm_source` and `provider.embedding_source` in `configs/config.yaml`; all four combinations are accepted by the factory.

| LLM | Embeddings | Required runtime configuration |
| --- | --- | --- |
| Google | Google | `GOOGLE_API_KEY`, plus `COHERE_API_KEY` for retrieval |
| Google | Ollama | `GOOGLE_API_KEY`, `OLLAMA_BASE_URL`, a running Ollama embedding model, and `COHERE_API_KEY` |
| Ollama | Google | `OLLAMA_BASE_URL`, a running Ollama chat model, `GOOGLE_API_KEY`, and `COHERE_API_KEY` |
| Ollama | Ollama | `OLLAMA_BASE_URL`, running Ollama chat and embedding models, and `COHERE_API_KEY` |

Model names are configured under `models` in `configs/config.yaml`: `google_llm`, `google_embedding`, `ollama_llm`, and `ollama_embedding`. Google credentials are needed whenever either selected provider is Google. Ollama defaults to `http://localhost:11434`; the configured models must already be pulled on that server. Evaluation uses the same provider selections rather than forcing a local backend.

## Project Structure
---

```
multimodal-enterprise-rag-engine/
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # GitHub Actions Automated CI/CD Pipeline
│
├── configs/
│   └── config.yaml                 # System configurations and engine profiles
│
├── data/
│   ├── raw_docs/                   # Drop ingestion PDFs here
│   └── vectorstore/                # Local ChromaDB persistent index layer
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralizes paths and system environment variables
│   ├── database.py                 # Initializes and manages Vector Store connections
│   │
│   ├── ingestion/                  # MODULE 1: Data Parsing & Structural Chunking
│   │   ├── __init__.py
│   │   ├── pdf_parser.py           # PDF loading and configurable chunking
│   │   └── multimodal_parser.py    # PyMuPDF text and table detection prototype
│   │
│   ├── pipeline/                   # MODULE 2: Retrieval & Cognitive Chains
│   │   ├── __init__.py
│   │   ├── chains.py               # Cognitive Router, Rerankers, and Core System prompts
│   │   └── schemas.py              # Pydantic models for strict JSON decoding output
│   │
│   ├── api/                        # MODULE 3: Headless Application Delivery Gateway
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI gateway application
│   │
│   ├── evaluation/                 # MODULE 4: RAGAS evaluation suite
│   │   └── optimizer.py            # Configured pipeline evaluation runner
│
│   └── tests/                      # Unit and integration tests
│       ├── test_api.py
│       ├── test_database.py
│       ├── test_evaluation.py
│       ├── test_ingestion.py
│       └── test_pipeline.py
│
├── .env.example                    # Environmental configuration template for secrets
├── Dockerfile                      # Application containerization manifest
├── pytest.ini                      # Python test configurations
└── requirements.txt                # Fixed application runtime dependency lock

```

## Quick Start & Installation

### 1. Clone and Setup Environment
```
git clone https://github.com/yansanW/Enterprise_Agentic_RAG_with_RAGAS_Evaluation.git
cd Enterprise_Agentic_RAG_with_RAGAS_Evaluation
```

### 2. Configure Local Secrets
Create a .env file in the project root directory using the template provided:
```
cp .env.example .env
```
Populate your local ```.env``` for the provider combination above. `COHERE_API_KEY` is required by every retrieval configuration. Keep `.env` out of source control.

### 3.a. Install Dependencies
Ensure your virtual environment is active (Python 3.12 recommended):
```
pip install -r requirements.txt
```

### 3.b. Containerized Deployment with Docker
Ensure Docker is installed, then build and run the API image:
```
docker build -t enterprise-agentic-rag .
docker run --rm -p 8000:8000 --env-file .env enterprise-agentic-rag
```
The repository intentionally does not include Docker Compose because the current prototype runs as a single application container. If you select Ollama, make its host endpoint reachable from the container and set `OLLAMA_BASE_URL` accordingly.

### 4. Run the Headless Server Gateway
To start the FastAPI server application locally, run:
```
uvicorn src.api.main:app --reload
```
Once initialized, navigate to ```http://127.0.0.1:8000/docs``` in your web browser to interact with the application via Swagger UI.

### 5. Run Tests
The default suite does not contact model providers:
```
python -m pytest
```
Provider-backed tests are opt-in and require deliberate test credentials/services:
```
python -m pytest --run-network
```

### 6. Execute Offline Optimization RAGAS Evaluations
To evaluate pipeline performance against the offline golden dataset questions, execute:
```
python -m src.evaluation.optimizer
```
