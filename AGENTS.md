# AGENTS.md

## Project overview

This repository is the Personal Document Finder application: a FastAPI-based document intelligence platform for ingestion, OCR, classification, semantic search, graph linking, and agentic querying. See the project overview in [README.md](README.md) and the deeper design notes in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Essential commands

- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Run the API locally:
  ```bash
  uvicorn app.main:app --reload
  ```
- Health check:
  ```bash
  curl http://localhost:8001/health
  ```
- Run the test suite:
  ```bash
  pytest tests/ -v
  ```
- Run the Streamlit UI:
  ```bash
  streamlit run ui/app.py --server.port 8501
  ```
- Run the MCP server:
  ```bash
  python -m app.mcp.server
  ```
- Run evaluation tasks:
  ```bash
  python -m app.evaluation
  ```

## Codebase layout

- [app/main.py](app/main.py): FastAPI application startup, middleware, router registration, and health endpoint.
- [app/config/settings.py](app/config/settings.py): central environment/config settings loaded from `.env`.
- [app/api/](app/api/): REST API routes and request/response schemas.
- [app/services/](app/services/): business/service layer for document, alert, comparison, insight, and timeline logic.
- [app/rag/](app/rag/): retrieval, re-ranking, citations, memory, and agent orchestration.
- [app/knowledge_graph/](app/knowledge_graph/): graph entities and relationships.
- [app/ocr/](app/ocr/): OCR engine implementations and pipeline.
- [app/embeddings/](app/embeddings/): chunking and embedding utilities.
- [app/vectorstore/](app/vectorstore/): vector database wrapper.
- [app/agent/](app/agent/): planner and tool orchestration for agentic workflows.
- [tests/](tests/): pytest-based regression and phase validation tests.

## Conventions for agents

- Use the existing layered structure: routes -> services -> platform-specific components.
- Keep environment configuration centralized in [app/config/settings.py](app/config/settings.py) instead of hardcoding provider or storage values.
- Prefer adding or updating tests for changed behavior in [tests/](tests/).
- Keep documentation in the existing docs set and link to it rather than duplicating it in code comments.
- Follow the project’s FastAPI/Pydantic patterns and keep API schemas aligned with route handlers.
- Respect the modularization by feature area; do not mix OCR, graph, and retrieval concerns in a single module unless the existing pattern already does so.
- For API or UI changes, validate the relevant behavior with the closest existing pytest target and avoid broad refactors without a clear cause.
- Treat the docs in [docs/](docs/) as the authoritative source for architecture and product behavior; link to them instead of restating design details in code comments or PR descriptions.
- Keep the FastAPI app and Streamlit UI aligned on the same ports and auth patterns: the app runs on port 8001 and the UI expects this backend at `http://localhost:8001`.

## Operational guidance for AI coding agents

- Start from the existing feature boundaries: document ingestion, OCR extraction, embeddings/search, graph logic, and agent orchestration are each intentionally separated in the app package.
- When a change touches multiple layers, keep the change small and preserve the integration points already defined by the route/schema/service pattern.
- Prefer targeted tests over ad hoc manual checks when a regression is in scope; the suite under [tests/](tests/) is the main project validation signal.
- Before adding new dependencies or environment flags, check whether the project already has a supported config path or module abstraction for that concern.

## References

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/API.md](docs/API.md)
- [docs/RAG.md](docs/RAG.md)
- [docs/MCP.md](docs/MCP.md)
- [docs/KNOWLEDGE_GRAPH.md](docs/KNOWLEDGE_GRAPH.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
