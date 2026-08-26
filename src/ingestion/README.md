# Module 1: PDF Text Extraction and Table-Detection Scaffolding

## Overview

The ingestion module extracts PDF text, attaches source metadata, and splits the extracted content before vector indexing. `pdf_parser.py` selects either `PyPDFLoader` or `MultimodalParser` according to configuration. The latter adds table-detection scaffolding through PyMuPDF, but it does not provide vision-language processing or visual document understanding.

```text
Raw PDFs ➔ Configured text/table-detection parser ➔ Chunk splitter ➔ Database indexer
```

Neither parser preserves the PDF's spatial layout, columns, diagrams, charts, or embedded-image semantics. Images are not interpreted.

---

## Parsing Behavior

### 1. Standard Text Extraction (`pdf_parser.py`)

The `standard_text` strategy uses `PyPDFLoader` to read the text layer exposed by a PDF. The resulting documents are passed to the configured static or semantic splitter. Source and page metadata supplied by the loader remain available on chunks, but spatial relationships from the original page are not reconstructed or preserved.

### 2. Table-Detection Scaffolding (`multimodal_parser.py`)

The `multimodal_vlm` strategy uses PyMuPDF to extract page text and detect tables. Pages containing detected tables are rendered as images as scaffolding for future multimodal processing, but the current implementation does not send those images to a vision-language model.

Detected tables are represented by fixed placeholder Markdown table content. This is not table transcription or extraction, and it does not retain the table's real values. Charts, diagrams, and other images are not summarized or interpreted.

### 3. Configurable Chunking

The static strategy uses `RecursiveCharacterTextSplitter` with configurable chunk size and overlap. Overlap can reduce context loss at chunk boundaries, but it does not guarantee that every sentence or fact remains intact. The semantic strategy uses `SemanticChunker`, whose boundaries behave differently and likewise provide no zero-loss guarantee.

---

## Component Layout & Responsibilities

- **`pdf_parser.py`:** Selects the configured parsing strategy and applies static or semantic chunking to extracted documents.
- **`multimodal_parser.py`:** Extracts page text, detects tables, and emits text/table-typed documents; detected table content is currently a placeholder.
- **`run_ingest.py`:** Runs ingestion into the test vector store and adds `ingested_at` and `parser_strategy` audit metadata to every chunk. Run with `python -m src.ingestion.run_ingest`.

---

## Testing & Pipeline Verification

By default, `src/tests/test_ingestion.py` checks invalid-input handling, parser-strategy routing, static splitter configuration, and the output document shape when the configured local fixture exists. Fixture-dependent parsing checks skip when the test PDF is unavailable.

The live multimodal metadata check is marked `network` and skips unless tests are run with `python -m pytest --run-network`; it can also skip when its document fixture is missing. Consequently, the default suite does not guarantee that document-dependent or live parsing checks execute on every run.
