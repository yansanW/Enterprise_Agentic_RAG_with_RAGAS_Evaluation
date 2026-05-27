# Module 1: Layout-Aware & Multimodal Document Ingestion Pipeline

## Overview
The Ingestion module handles the parsing, structural breakdown, and normalization of raw incoming documents before they are transformed into mathematical vectors. Instead of treating text as a flat, unformatted character stream, this pipeline preserves layout structures and processes rich visual assets (such as diagrams, charts, and embedded images) to protect critical semantic context for downstream generation.

```
Raw Documents ➔ Specialized Parsers (pdf / multimodal) ➔ Database Indexer (database.py)
```
---

## Architectural Design Patterns

### 1. Layout-Aware Parsing Pattern
When processing dense technical papers or corporate records, naive text scrapers flatten structural layout markers, turning complex table grids and columns into unreadable walls of text. This module isolates the parsing loops via dedicated extraction loaders (`pdf_parser.py`) to preserve structural boundaries:
* **Structural Preservation:** Headers, footers, and side-by-side data columns maintain spatial proximity logic.
* **Metadata Attachment:** Original source attributes, file names, and specific document page indices are appended to every chunk object automatically, facilitating upstream verification and citation paths.

### 2. Multimodal Extraction Pattern (`multimodal_parser.py`)
To process enterprise knowledge bases that rely heavily on visual figures, financial charts, and technical schemas, the ingestion layer includes a dedicated multimodal parser. 
* **Visual Representation Processing:** Instead of ignoring embedded images or rendering them as unreadable text blocks, this component processes images or layouts using vision-capable model nodes.
* **Contextual Image Summarization:** Visual elements are converted into descriptive textual summaries or structured data blocks that retain 100% of the chart's original data properties, allowing the vector database to cleanly index figures for semantic text searching.

### 3. Overlapping Sliders (Recursive Chunking)
To prevent critical facts or technical details from being cut in half at a hard text-limit boundary, the pipeline utilizes an overlapping sliding window strategy:
* **Chunk Size:** Configured to balance semantic density and processing efficiency.
* **Chunk Overlap:** Maintains text boundaries between adjacent chunks, guaranteeing that sentences spanning across chunk splits retain their context in at least one vector window.

---

## Component Layout & Responsibilities

* **`pdf_parser.py`:** Standard structural loader that reads raw disk binaries, extracts page text layout grids, and structures them into clean text chunk arrays.
* **`multimodal_parser.py`:** Advanced parsing asset designed to isolate, extract, and interpret visual diagrams, embedded images, and data figures using multimodal processing engines.

---

## Testing & Pipeline Verification
The extraction accuracy, multimodal parsing stability, and chunk sizing constraints are validated continuously using automated mocks inside `src/tests/test_ingestion.py`. 

The test cases isolate local file system storage boundaries using isolated temporary directories (`tmp_path`), ensuring that your code can cleanly parse sample inputs and enforce precise split structures across all continuous integration environments without depending on external asset paths or hitting active cloud endpoints.