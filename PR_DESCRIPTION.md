# README audit remediation

## Bugs fixed

- Removed the import-time Ollama request and standardized local-provider configuration on `OLLAMA_BASE_URL`.
- Added early pytest safety configuration plus an explicit `--run-network` opt-in.
- Changed the container command to serve FastAPI with Uvicorn and made degraded `/health` startup state safe.

## Stale docs fixed

- Corrected the clone URL/directory and test paths.
- Removed nonexistent Docker Compose instructions in favor of the single-container workflow.

## Claims corrected

- Reframed the project as a prototype and documented current ingestion, canned `CHAT`, structured-output, evaluation, concurrency, and production limitations.
- Moved VLM, layout reconstruction, audio/Whisper, verification, and scale ambitions to the roadmap.

## Missing docs added

- Documented Chroma MMR retrieval followed by Cohere reranking, including settings and credentials.
- Added the Google/Ollama LLM and embedding provider matrix and test-network opt-in.

## Verification

- A clean `git archive HEAD` export contains the documented paths and passes Python bytecode compilation.
- Docker was successfully built and run, and `/health` returned `status: healthy`, `database_connected: true`, and `pipeline_active: true`.

## Open RAG raw-context correction

The 60-item Open RAG evaluation now passes the reranked documents raw `page_content` to RAGAS instead of model-generated citations. The first runtime sample was visually confirmed as full source-PDF paragraphs.

| Metric | Citation contexts (buggy) | Raw retrieved contexts (fixed) |
| --- | ---: | ---: |
| Faithfulness | 0.3063 | 0.7516 |
| Answer Relevancy | 0.6795 | 0.6795 |
| Context Precision | 0.1167 | 0.8192 |
| Context Recall | 0.4269 | 0.8220 |

The corrected run had two malformed local-judge outputs out of 240 metric jobs; RAGAS omitted those values from aggregation. The context metrics are now based on retriever output rather than citation strings. Faithfulness also changed because RAGAS evaluates answer claims against contexts; Answer Relevancy remained unchanged because it does not depend on contexts.
