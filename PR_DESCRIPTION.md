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
- The host denied access to `/var/run/docker.sock`, so Docker build/run and the `/health` curl could not be completed in this environment.
