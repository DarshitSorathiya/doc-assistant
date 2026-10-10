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
  curl http://localhost:8000/health
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

## References

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/API.md](docs/API.md)
- [docs/RAG.md](docs/RAG.md)
- [docs/MCP.md](docs/MCP.md)
- [docs/KNOWLEDGE_GRAPH.md](docs/KNOWLEDGE_GRAPH.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
