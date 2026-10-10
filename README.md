# Personal Document Finder

AI-powered Personal Document Intelligence platform. Upload, OCR, classify, search, and query your personal documents with agentic RAG, knowledge-graph linking, and proactive alerts.

## Quick Start (Development)

### 1. Prerequisites
- Python 3.11+
- (Optional) Tesseract OCR installed on your system
- (Optional) Docker & Docker Compose

### 2. Setup

```bash
# Clone / navigate to the project
cd document-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

Visit [http://localhost:8001/health](http://localhost:8001/health) — you should see:

```json
{
  "status": "healthy",
  "service": "Personal Document Finder",
  "version": "0.1.0"
}
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
document-assistant/
├── app/
│   ├── api/{routes/, schemas/}        # REST API endpoints & Pydantic models
│   ├── agent/{agent, planner, tools}  # Agentic RAG orchestration
│   ├── auth/{jwt, security}           # JWT auth & RBAC
│   ├── config/                        # Pydantic Settings (.env)
│   ├── database/                      # SQLAlchemy engine & sessions
│   ├── embeddings/                    # Sentence-transformers wrapper
│   ├── extraction/{classification, entities, metadata}
│   ├── knowledge_graph/{entities, relationships, graph}
│   ├── mcp/{server, tools, resources} # MCP protocol server
│   ├── ocr/{paddleocr, tesseract}     # OCR engines
│   ├── rag/{retriever, hybrid_search, reranker, chain, citations}
│   ├── services/{document, summary, comparison, timeline, alert}
│   ├── evaluation/{retrieval, generation}
│   ├── models/                        # SQLAlchemy ORM models
│   ├── vectorstore/                   # ChromaDB wrapper
│   └── utils/
├── ui/app.py                          # Streamlit UI
├── tests/
├── docs/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Build Phases

| Phase | Description | Status |
|-------|------------|--------|
| 1 | Repository Scaffold & Config | ✅ Complete |
| 2 | Database Models & Auth (JWT + RBAC) | ✅ Complete |
| 3 | Document Upload, Storage & Metadata | ✅ Complete |
| 4 | OCR & Text Extraction Pipeline | ✅ Complete |
| 5 | Classification & Entity Extraction | ✅ Complete |
| 6 | Embeddings, Vector Store & Chunking | ✅ Complete |
| 7 | Hybrid Search (Vector + BM25) | ✅ Complete |
| 8 | RAG Chain, Citations & Memory | ✅ Complete |
| 9 | Agentic RAG & Tool Calling | ✅ Complete |
| 10 | Document Services | ✅ Complete |
| 11 | Knowledge Graph & Alerts | ✅ Complete |
| 12 | MCP Server & Audit Logging | ✅ Complete |
| 13 | Streamlit UI | ✅ Complete |
| 14 | RAG Evaluation & Full Test Suite | ✅ Complete |
| 15 | Docs, Security & Deployment | ✅ Complete |

## Running the Streamlit UI

```bash
streamlit run ui/app.py --server.port 8501
```

## Running the MCP Server

```bash
python -m app.mcp.server
```

## Running RAG Evaluation

```bash
python -m app.evaluation
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and component overview
- [API Reference](docs/API.md) — Complete REST API documentation
- [RAG Pipeline](docs/RAG.md) — Retrieval and generation pipeline details
- [MCP Server](docs/MCP.md) — MCP protocol server documentation
- [Knowledge Graph](docs/KNOWLEDGE_GRAPH.md) — Knowledge graph design
- [Deployment Guide](docs/DEPLOYMENT.md) — Setup, Docker, and production checklist

## License

Private — all rights reserved.

