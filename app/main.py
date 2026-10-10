# ============================================================
# Personal Document Finder — FastAPI Application Entry Point
# ============================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.database import create_tables
from app.api.routes.auth import router as auth_router
from app.api.routes.documents import router as documents_router
from app.api.routes.search import router as search_router
from app.api.routes.query import router as query_router
from app.api.routes.services import router as services_router
from app.api.routes.graph import router as graph_router
from app.api.routes.alerts import router as alerts_router
from app.api.routes.audit import router as audit_router


# ------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan manager.

    Startup:
      - Log configuration summary (without secrets).
      - Create database tables.
    Shutdown:
      - Clean up resources.
    """
    settings = get_settings()
    print("=" * 60)
    print("  Personal Document Finder — starting up")
    print(f"  LLM Provider : {settings.LLM_PROVIDER}")
    print(f"  OCR Engine   : {settings.OCR_ENGINE}")
    print(f"  Vector DB    : {settings.VECTOR_DB}")
    print(f"  Database     : {settings.DATABASE_URL}")
    print(f"  Debug        : {settings.DEBUG}")
    print("=" * 60)

    # Initialize SQLite/Postgres tables
    create_tables()

    yield
    # Shutdown cleanup (future phases will close DB sessions, etc.)
    print("Personal Document Finder — shutting down")


# ------------------------------------------------------------------
# Create the FastAPI application
# ------------------------------------------------------------------
app = FastAPI(
    title="Personal Document Finder",
    description=(
        "AI-powered Personal Document Intelligence platform. "
        "Upload, OCR, classify, search, and query your personal documents "
        "with agentic RAG, knowledge-graph linking, and proactive alerts."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    """Point browser visitors to the separate Streamlit frontend."""
    return {
        "service": "Personal Document Finder API",
        "health": "/health",
        "docs": "/docs",
        "ui": "http://localhost:8501",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return an empty successful response for browser favicon probes."""
    return Response(status_code=204)


# ------------------------------------------------------------------
# CORS middleware — allow all origins in dev; restrict in production
# ------------------------------------------------------------------
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Register API Routers
# ------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(query_router)
app.include_router(services_router)
app.include_router(graph_router)
app.include_router(alerts_router)
app.include_router(audit_router)


# ------------------------------------------------------------------
# Health-check endpoint
# ------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """Return 200 to confirm the service is alive.

    Future phases will add deeper checks (DB connectivity, vector store, etc.).
    """
    return {
        "status": "healthy",
        "service": "Personal Document Finder",
        "version": "0.1.0",
    }
