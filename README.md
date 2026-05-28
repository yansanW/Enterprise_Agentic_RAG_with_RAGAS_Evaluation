# Enterprise-Agentic-RAG-with-RAGAS-Evaluation

## Architectural Features & Engineering Demonstrations
---
This project demonstrates a production-ready, headless **Enterprise Multimodal RAG Engine** optimized for layout-dense documents (research papers, financial tables, media feeds).

### 1. Spatial Ingestion Engine: 
(`src/ingestion/`)
* Replaced naive left-to-right text extraction with layout-aware parsing via PyMuPDF (`fitz`). 
* Structurally isolates tables and visual grids, tagging assets with metadata matrices (`{"type": "table"}`) to preserve contextual geometric layouts.

### 2. Cross-Modal Data Tracks:
* Processes multimedia links dynamically. 
* Downloads and transcribes video streams locally using **OpenAI Whisper** with microsecond time-offset indexing.

### 3. Deterministic Guardrails:
(`src/pipeline/`)
* Implements **Pydantic** schema validation forced directly onto LLM decoding layers, guaranteeing strict JSON responses and automated fallback handling if context boundaries are leaked.

### 4. Rigorous MLOps Validation:
(`src/eval/`)
* Utilizes an automated **RAGAS Evaluation pipeline** against a curated offline Golden Dataset to analytically optimize *Context Precision* and *Faithfulness*.

### 5. Sovereign Cloud Deployment:
*  Implements a strict **Factory & Strategy Design Pattern** to seamlessly switch backends between cloud endpoints (Gemini API) and fully local tensor execution (Ollama) via `config.yaml`. 
* Delivered as a headless, containerized **FastAPI** application with **Docker Compose**.


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
├── tests/                          # Unit & Integration Validation Matrix
│   ├── test_api.py
│   ├── test_database.py
│   ├── test_evaluation.py          # Automated mock testing for evaluation integrity
│   ├── test_ingestion.py
│   └── test_pipeline.py
│
├── .env.example                    # Environmental configuration template for secrets
├── Dockerfile                      # Application containerization manifest
├── docker-compose.yml              # Multi-container orchestration config
├── pytest.ini                      # Python test configurations
└── requirements.txt                # Fixed application runtime dependency lock

```

## Quick Start & Installation

### 1. Clone and Setup Environment
```
git clone ...
cd multimodal-enterprise-rag-engine
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
This repository includes a multi-container `docker-compose` setup to orchestrate both the headless FastAPI application gateway and a persistent volume layer for your local Chroma vector store seamlessly.

#### 1. Prerequisites
Ensure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your host machine. If you plan to utilize local sovereign models (`llama3`), ensure your local [Ollama instance](https://ollama.com/) is running on your host system.

#### 2. Build and Launch the Stack
To build the application image and spin up the container network in detached (background) mode, execute:
```
docker-compose up --build -d
```

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
