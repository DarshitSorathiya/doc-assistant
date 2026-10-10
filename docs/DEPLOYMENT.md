# Deployment Guide — Personal Document Finder

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- pip / virtualenv

### 1. Setup
```bash
cd document-assistant
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration
```bash
cp .env.example .env
# Edit .env with your settings:
#   - Set OPENAI_API_KEY if using OpenAI
#   - Set JWT_SECRET_KEY to a random string
#   - Set SECRET_KEY to a random string
```

### 3. Run the API Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Verify: `curl http://localhost:8000/health`

### 4. Run the Streamlit UI
```bash
streamlit run ui/app.py --server.port 8501
```
Open: http://localhost:8501

### 5. Run the MCP Server
```bash
python -m app.mcp.server
```

### 6. Run Tests
```bash
pytest -v -p no:asyncio
```

### 7. Run RAG Evaluation
```bash
python -m app.evaluation
```

---

## Docker Deployment

### Using Docker Compose
```bash
docker-compose up --build
```

This starts:
- **API server** on port 8000
- **PostgreSQL** on port 5432 (internal)
- **Ollama** (optional) on port 11434

### Environment Variables for Production
Set these in your `.env` or via Docker environment:

| Variable | Production Value | Notes |
|----------|-----------------|-------|
| `JWT_SECRET_KEY` | Random 64-char string | **MUST** change from default |
| `SECRET_KEY` | Random 64-char string | **MUST** change from default |
| `DATABASE_URL` | `postgresql://...` | Use PostgreSQL in production |
| `DEBUG` | `false` | Disable SQL logging & wide CORS |
| `OPENAI_API_KEY` | Your API key | Required if `LLM_PROVIDER=openai` |
| `MAX_UPLOAD_SIZE_MB` | `50` | Adjust as needed |

---

## Production Checklist

### Security
- [ ] Change `JWT_SECRET_KEY` and `SECRET_KEY` to strong random values
- [ ] Set `DEBUG=false`
- [ ] Configure CORS `allow_origins` to your specific domain(s)
- [ ] Enable HTTPS (reverse proxy with nginx/traefik)
- [ ] Set strict file permissions on `storage/` directory
- [ ] Review rate limiting and request size limits
- [ ] Verify all routes enforce user-level access control
- [ ] Confirm audit logging is active and capturing all actions

### Database
- [ ] Migrate from SQLite to PostgreSQL for production
- [ ] Set up database backups
- [ ] Run Alembic migrations: `alembic upgrade head`

### Storage
- [ ] Configure external storage (S3, GCS) for document files if needed
- [ ] Set appropriate `MAX_UPLOAD_SIZE_MB`
- [ ] Ensure `storage/` directory has proper permissions

### Monitoring
- [ ] Set up application logging to a centralized service
- [ ] Monitor `/health` endpoint
- [ ] Set up alerting on error rates

---

## Streamlit UI Smoke Test Checklist

Since Streamlit isn't unit-testable the same way as the API, verify manually:

| Screen | Test Steps | Expected Result |
|--------|-----------|-----------------|
| Login | Enter valid credentials, click Login | Redirected to Documents page |
| Register | Create new account | Success message, can login |
| Documents | Upload a PDF file | File appears in list with "processing" → "ready" status |
| Documents | View uploaded documents | Category badges, status badges, page count shown |
| Chat Q&A | Ask "What documents do I have?" | Lists uploaded documents with sources |
| Chat Q&A | Ask follow-up question | Resolves context from prior turn |
| Search | Run hybrid search for a keyword | Results with scores and text previews |
| Insights | View alerts dashboard | Shows expiry/missing doc/duplicate alerts |
| Timeline | Generate timeline | Chronologically sorted date events |
| Knowledge Graph | View graph | Nodes and edges for entities/documents |
| Compare | Select two documents, compare | Structured diff with added/removed info |
| Logout | Click logout | Returns to login screen |

---

## Architecture Reference

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full system design.
See [`API.md`](API.md) for complete API reference.
See [`MCP.md`](MCP.md) for MCP server documentation.
See [`RAG.md`](RAG.md) for RAG pipeline details.
See [`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md) for knowledge graph documentation.
