# Module 1: Ingestion & Layout-Aware Parsing Engine

## Overview
The Ingestion module handles the system's data input boundaries. Rather than treating documents as flat text strings, this component is designed as an adaptive pipeline that dynamically responds to document layout complexity (text paragraphs vs. multi-column research papers and embedded grid tables).

## Architectural Design Patterns

### 1. The Strategy Pattern (Layout Dissection)
Instead of processing all files through a monolithic parser loop, we decouple file types into distinct processing strategies:
* **`pdf_parser.py` / `multimodal_parser.py`:** Integrates PyMuPDF (`fitz`) for low-latency bounding-box extraction. If a document page contains structural grid elements or visual charts, the pipeline isolates them, tagging chunks with metadata metrics (`{"type": "table"}` or `{"type": "text"}`) to prepare them for downstream Vision-Language Model (VLM) context extraction.
* **`video_parser.py`:** Extracted YouTube audio streams via `yt-dlp` and processes automatic speech recognition (ASR) locally using an execution of the **OpenAI Whisper** model, saving chunk offsets with precise visual timestamp markers.

### 2. Factory Pattern Configuration Routing
The initialization logic reads structural options directly from `configs/config.yaml`. By setting parameters like `parser_strategy: "multimodal_vlm"`, the system switches core engines at runtime without requiring any manual modification of production Python scripts.

---

## Technical Decisions & Trade-offs

### Naive Text Extraction vs. Spatial Multimodal Parsing
Standard text readers (e.g., PyPDF) extract text left-to-right, completely scrambling reading sequences in multi-column research layouts and turning tabular cells into chaotic, unsearchable tokens. 

* **The Trade-off:** Slicing page coordinates and routing blocks to visual-language encoders increases the compute overhead and token latency during the initial document upload phase. However, this trade-off is mathematically justified because it prevents downstream LLM hallucinations and significantly boosts **Context Precision** during RAGAS evaluations.

---

## Automated Verification Loop
This module is fully decoupled from presentation application layers (Streamlit / FastAPI), making it perfectly testable in complete isolation. 

Unit tests are written inside `tests/test_ingestion.py` using `pytest`. The suite implements `monkeypatch` configuration mocking and custom PyMuPDF fault-injection boundaries (`fitz.FileNotFoundError`) to ensure strict error handling before code changes are pushed to deployment registries.
```
python -m pytest src/tests/test_ingestion.py
```