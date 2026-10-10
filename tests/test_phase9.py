# ============================================================
# Phase 9 — Agentic RAG & Tool Calling Tests
# ============================================================

import io
import pytest
import fitz
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import AuditLog
from app.rag.memory import conversation_memory


# ------------------------------------------------------------------
# Test Database Fixture (In-Memory SQLite with StaticPool)
# ------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    conversation_memory._sessions.clear()
    yield
    Base.metadata.drop_all(bind=engine)
    conversation_memory._sessions.clear()


@pytest.fixture
def client():
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ------------------------------------------------------------------
# Auth Helpers
# ------------------------------------------------------------------
@pytest.fixture
def auth_headers_user1(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "agent_user1@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "agent_user1@example.com", "password": "Password123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "agent_user2@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "agent_user2@example.com", "password": "Password123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_sample_pdf(tmp_path, filename: str, content: str) -> bytes:
    """Helper to generate a born-digital PDF byte stream."""
    path = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content)
    doc.save(str(path))
    doc.close()
    with open(path, "rb") as f:
        return f.read()


# ------------------------------------------------------------------
# Test Setup: Upload Corpus with Multiple Insurance Policies
# ------------------------------------------------------------------
@pytest.fixture
def setup_agent_corpus(client: TestClient, auth_headers_user1, auth_headers_user2, tmp_path):
    # 1. User 1: Policy A (HDFC Ergo)
    pdf1 = _create_sample_pdf(
        tmp_path, "hdfc_health_policy.pdf",
        "HDFC ERGO Health Insurance Policy\n"
        "Policy Number: POL-HDFC-100\n"
        "Sum Insured: Rs. 5,00,000\n"
        "Premium: Rs. 15,000\n"
        "Expiry Date: 15-Jun-2027\n"
        "PAN: ABCDE1234F"
    )
    res1 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("hdfc_health_policy.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    doc1_id = res1.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc1_id}/process", headers=auth_headers_user1)

    # 2. User 1: Policy B (Star Health)
    pdf2 = _create_sample_pdf(
        tmp_path, "star_health_policy.pdf",
        "Star Health Comprehensive Insurance Policy\n"
        "Policy Number: POL-STAR-200\n"
        "Sum Insured: Rs. 10,00,000\n"
        "Premium: Rs. 22,000\n"
        "Expiry Date: 20-Dec-2026\n"
        "PAN: ABCDE1234F"
    )
    res2 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("star_health_policy.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    doc2_id = res2.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc2_id}/process", headers=auth_headers_user1)

    # 3. User 2: Private Policy (User 2)
    pdf3 = _create_sample_pdf(
        tmp_path, "user2_private_policy.pdf",
        "Private Executive Coverage Policy\n"
        "Policy Number: POL-PRIV-999\n"
        "Sum Insured: Rs. 50,00,000\n"
        "Expiry Date: 01-Jan-2030"
    )
    res3 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user2,
        files={"file": ("user2_private_policy.pdf", io.BytesIO(pdf3), "application/pdf")},
    )
    doc3_id = res3.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc3_id}/process", headers=auth_headers_user2)

    return {"doc1_id": doc1_id, "doc2_id": doc2_id, "doc3_id": doc3_id}


# ==================================================================
# Test Suites
# ==================================================================
class TestAgenticToolCalling:
    """Test dynamic tool selection, execution, and multi-step planning."""

    def test_compare_policies_scenario(self, client: TestClient, auth_headers_user1, setup_agent_corpus):
        """Test multi-step comparison query invokes compare_documents and get_expiring_documents."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Compare my two health insurance policies and tell me which expires first.",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()

        assert "Comparison" in data["answer"] or "hdfc" in data["answer"].lower()
        assert "compare_documents" in data["tools_invoked"]
        assert len(data["tools_invoked"]) >= 1

    def test_entity_directed_tool_calling(self, client: TestClient, auth_headers_user1, setup_agent_corpus):
        """Test query mentioning a PAN number invokes search_by_entity tool."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Which documents contain PAN ABCDE1234F?",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()

        assert "search_by_entity" in data["tools_invoked"]
        assert "ABCDE1234F" in data["answer"] or "hdfc" in data["answer"].lower()

    def test_timeline_tool_calling(self, client: TestClient, auth_headers_user1, setup_agent_corpus):
        """Test timeline query invokes extract_timeline tool."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Show me the timeline and important dates across my documents.",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()

        assert "extract_timeline" in data["tools_invoked"]
        assert "Timeline" in data["answer"] or "2026" in data["answer"] or "2027" in data["answer"]


class TestAuditLogPersistence:
    """Test that every agent tool invocation is recorded in the audit trail."""

    def test_agent_tool_calls_are_audited(self, client: TestClient, auth_headers_user1, setup_agent_corpus):
        """Verify AuditLog table records tool call entries upon query execution."""
        # Execute query that triggers tools
        client.post(
            "/api/v1/query",
            json={
                "query": "Which documents are expiring soon?",
            },
            headers=auth_headers_user1,
        )

        # Inspect database audit log
        db = TestingSessionLocal()
        try:
            logs = db.query(AuditLog).filter(AuditLog.action.like("agent_tool_call%")).all()
            assert len(logs) >= 1
            actions = [l.action for l in logs]
            assert any("get_expiring_documents" in a or "hybrid_search" in a for a in actions)
        finally:
            db.close()


class TestAgentUserIsolation:
    """Test that the agent cannot access another user's documents."""

    def test_agent_cannot_access_other_user_documents(self, client: TestClient, auth_headers_user1, setup_agent_corpus):
        """Verify User 1 querying for User 2's policy gets no matches or tool leakage."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Compare POL-PRIV-999 with my policies.",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()

        assert "user2_private_policy.pdf" not in data["answer"]
        cited_files = [s["filename"] for s in data["sources"]]
        assert "user2_private_policy.pdf" not in cited_files
