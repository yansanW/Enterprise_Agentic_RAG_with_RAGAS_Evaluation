# Enterprise-Agentic-RAG-with-RAGAS-Evaluation

## Architectural Features & Engineering Demonstrations
---
This repository showcases a production-ready, **headless Enterprise Agentic RAG (Retrieval-Augmented Generation) Engine** optimized for parsing layout-dense documents, persistent semantic search, and automated offline optimization.

### 1. Spatial Ingestion Engine (`src/ingestion/`)
* Replaced naive text-scraping extractions with layout-aware structural parsing via PyMuPDF (`fitz`).
* Isolates text layers and document structures, preparing visual and structural blocks for semantic chunking to preserve layout boundaries during database vectorization.

### 2. Cognitive Routing Matrix (`src/pipeline/`)
* Implements an intelligent **Cognitive Router** that evaluates incoming user query intent in real-time.
* Dynamically switches execution tracks between **`RETRIEVE`** (for fact-based database vector searches) and **`CHAT`** (for general conversational context) to protect system efficiency and minimize token waste.

### 3. Deterministic Guardrails (`src/pipeline/`)
* Enforces structural **Pydantic** schema validation directly onto LLM decoding layers, guaranteeing strict JSON output responses and protecting against context boundary leakage.

### 4. Rigorous MLOps Validation Suite (`src/evaluation/`)
* Couples an automated **RAGAS Evaluation framework** with an offline target validation dataset to completely decouple test data from pipeline execution.
* Calculates mathematical score metrics across four independent vectors: *Faithfulness*, *Answer Relevance*, *Context Precision*, and *Context Recall* utilizing a local sovereign model (**Ollama / Llama3**).

### 5. Sovereign Cloud Execution (`src/api/`)
* Delivers a headless, enterprise-grade **FastAPI** application workspace wrapped in an automated GitHub Actions CI/CD quality gate, fully configured for containerized deployment via Docker Compose.


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
│   │   ├── pdf_parser.py           # Layout-aware PDF chunking script
│   │   └── multimodal_parser.py    # Multimedia and Whisper transcription processors
│   │
│   ├── pipeline/                   # MODULE 2: Retrieval & Cognitive Chains
│   │   ├── __init__.py
│   │   ├── chains.py               # Cognitive Router, Rerankers, and Core System prompts
│   │   └── schemas.py              # Pydantic models for strict JSON decoding output
│   │
│   ├── api/                        # MODULE 3: Headless Application Delivery Gateway
│   │   ├── __init__.py
│   │   └── main.py                 # High-performance FastAPI gateway application
│   │
│   └── evaluation/                 # MODULE 4: MLOps Quality Control Suite
│       ├── golden_dataset.json     # Decoupled validation ground-truths and test cases
│       └── optimizer.py            # RAGAS metric validation calculation suite
│
│   └── tests/                    # Unit and integration tests
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
Populate your local ```.env``` with your API credentials (ensure ```.env``` is never committed to source control).

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

### 5. Execute Offline Optimization RAGAS Evaluations
To evaluate pipeline performance against the offline golden dataset questions, execute:
```
python -m src.evaluation.optimizer
```
