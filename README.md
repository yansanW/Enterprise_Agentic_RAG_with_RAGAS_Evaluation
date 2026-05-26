# Enterprise-Agentic-RAG-with-RAGAS-Evaluation

# Project Structure
---

```
Enterprise_Agentic_RAG_with_RAGAS_Evaluation/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD pipeline
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralizes paths and API keys
│   ├── database.py            # Initialises ChromaDB connection
│   │
│   ├── ingestion/             # MODULE 1: Data Parsing & Chunking
│   │   ├── __init__.py
│   │   ├── pdf_parser.py      # Handles Semantic Chunking for PDFs
│   │   └── video_parser.py    # Extracts YouTube transcripts/Whisper
│   │
│   ├── pipeline/              # MODULE 2: Retrieval & Generation, Core ML Logic (Pure Python)
│   │   ├── __init__.py
│   │   ├── agents.py          # LangGraph or routing agent logic
│   │   └── tools.py           # Retrieval, code execution tools ## to be define
│   │   ├── chains.py          # Query rewriter, Cohere Rerank, Guardrails
│   │   └── schemas.py         # Pydantic models for JSON enforcement
│   │
│   └── eval/                  # Domain 3: Evaluation with RAGAS
│   ├── eval_ragas.py          # Evaluation engine (already built!)
│   └── app_fastapi.py         # Headless API gateway (replaces Streamlit)
│   │
│   └── api/                   # Domain 4: Application Delivery
│   │   └── main.py            # Headless FastAPI app gateway
│   │
├── tests/                     # Unit & Integration tests
│   ├── test_ingestion.py
│   └── test_pipeline.py
│
├── data/
│   ├── raw_docs/              # Place test PDFs here
│   └── golden_dataset.json    # Your offline evaluation Q&A pairs
│
├── README.md
├── .env.example                  # Template for secrets (API keys)
├── Dockerfile                 # For containerisation
├── docker-compose.yml         # Orchstrates app + local database
└── requirements.txt
```
---

### src/
Core source code.

- **data/**: Data loading and preprocessing
- **models/**: Model definitions
- **training/**: Training logic
- **evaluation/**: Metrics and validation
- **utils/**: Helper functions

### configs/
Configuration files (YAML). Avoid hardcoding parameters in code.

### notebooks/
Exploration only. Do NOT put production logic here.

### main.py
Entry point of the pipeline.

---

## Philosophy

- Keep components modular
- Avoid hardcoding (use config)
- Separate responsibilities
- Make everything reusable
