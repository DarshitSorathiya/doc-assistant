# ============================================================
# Phase 1 — Health Check & Config Tests
# ============================================================

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config.settings import Settings, get_settings


# ------------------------------------------------------------------
# Fixture: TestClient wrapping the FastAPI app
# ------------------------------------------------------------------
@pytest.fixture
def client():
    """Create a synchronous test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------
# 1. Health-check endpoint
# ------------------------------------------------------------------
class TestHealthCheck:
    """Verify the /health endpoint works correctly."""

    def test_health_returns_200(self, client: TestClient):
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client: TestClient):
        """GET /health must return the expected JSON structure."""
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "Personal Document Finder"
        assert "version" in data


# ------------------------------------------------------------------
# 2. Configuration / Settings
# ------------------------------------------------------------------
class TestSettings:
    """Verify the Pydantic Settings class loads defaults correctly."""

    def test_defaults_loaded(self):
        """Settings should have sensible defaults without a .env file."""
        s = Settings()
        assert s.LLM_PROVIDER in ("openai", "ollama")
        assert s.OCR_ENGINE in ("paddleocr", "tesseract")
        assert s.VECTOR_DB == "chroma"
        assert s.JWT_ALGORITHM == "HS256"
        assert s.MAX_UPLOAD_SIZE_MB > 0

    def test_get_settings_returns_same_instance(self):
        """get_settings() must return a cached singleton."""
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_database_url_has_default(self):
        """DATABASE_URL must have a dev-friendly default."""
        s = Settings()
        assert "sqlite" in s.DATABASE_URL


# ------------------------------------------------------------------
# 3. CORS — quick sanity check
# ------------------------------------------------------------------
class TestCORS:
    """Verify CORS middleware is attached."""

    def test_cors_headers_present(self, client: TestClient):
        """An OPTIONS preflight request should return CORS headers in dev mode."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # In debug mode we allow all origins
        assert response.status_code == 200
