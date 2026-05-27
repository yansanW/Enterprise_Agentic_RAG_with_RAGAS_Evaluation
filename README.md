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
Enterprise_Agentic_RAG_with_RAGAS_Evaluation/
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # GitHub Actions CI/CD pipeline
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralizes paths and API keys
│   ├── database.py                 # Initialises ChromaDB connection
│   │
│   ├── ingestion/                  # MODULE 1: Data Parsing & Chunking
│   │   ├── __init__.py
│   │   ├── pdf_parser.py           # Handles Semantic Chunking for PDFs
│   │   ├── multimodal_parser.py    # Handles Multimodal Extraction Pattern
│   │   └── README.md
│   │
│   ├── pipeline/              # MODULE 2: Retrieval & Generation, Core ML Logic (Pure Python)
│   │   ├── __init__.py
│   │   ├── chains.py          # Query rewriter, Cohere Rerank, Guardrails
│   │   ├── schemas.py         # Pydantic models for JSON enforcement
│   │   └── README.md
│   │
│   └── api/                   # Domain 3: Application Delivery
│   │   ├── main.py            # Headless FastAPI app gateway
│   │   └── README.md
│   │
│   └── evaluation/            # Domain 4: Evaluation with RAGAS
│   ├── optimizer.py           # Evaluation engine (already built!)
│   │
├── tests/                     # Unit & Integration tests
│   ├── test_conf.py
│   ├── test_ingestion.py
│   ├── test_databse.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── data/
│   ├── raw_docs/              # Place test PDFs here
│   └── golden_dataset.json    # Your offline evaluation Q&A pairs
│
├── README.md
├── .env.example               # Template for secrets (API keys)
├── Dockerfile                 # For containerisation
├── docker-compose.yml         # Orchstrates app + local database
├── pytest.ini
└── requirements.txt

```

### src/
Core source code.

- **ingestion/**: handles the input data boundaries (PDFs, Videos, etc.).
- **pipeline/**: handles the cognitive core (Chains, Agents, Prompts, Rerankers).
- **api/**: handles the application delivery layer (FastAPI).
- **utils/**: Helper functions


## Installation
---
```
pip install -r requirements.txt
```
---
