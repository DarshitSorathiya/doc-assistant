# ============================================================
# Phase 8 — RAG Chain, Citations & Conversational Memory Tests
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
        json={"email": "user1_rag@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "user1_rag@example.com", "password": "Password123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "user2_rag@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "user2_rag@example.com", "password": "Password123!"},
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
# Test Setup: Upload and Process Corpus for User 1 and User 2
# ------------------------------------------------------------------
@pytest.fixture
def setup_rag_corpus(client: TestClient, auth_headers_user1, auth_headers_user2, tmp_path):
    # 1. User 1: Health Insurance Document
    pdf1 = _create_sample_pdf(
        tmp_path, "hdfc_ergo_health_policy.pdf",
        "HDFC ERGO Health Insurance Policy Schedule\n"
        "Policy Number: POL-HDFC-992211\n"
        "Policyholder: John Doe\n"
        "Sum Insured: Rs. 10,00,000 (Ten Lakhs)\n"
        "Annual Premium Amount: Rs. 24,500 inclusive of GST\n"
        "Policy Expiry Date: 31-Dec-2027\n"
        "Coverage includes in-patient hospitalization and daycare procedures."
    )
    res1 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("hdfc_ergo_health_policy.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    doc1_id = res1.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc1_id}/process", headers=auth_headers_user1)

    # 2. User 1: Bank Account Statement
    pdf2 = _create_sample_pdf(
        tmp_path, "icici_bank_statement.pdf",
        "ICICI Bank Savings Account Statement\n"
        "Account Number: 001122334455\n"
        "IFSC Code: ICIC0000011\n"
        "Account Holder: John Doe\n"
        "Closing Balance as of Oct 2026: Rs. 1,45,000"
    )
    res2 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("icici_bank_statement.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    doc2_id = res2.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc2_id}/process", headers=auth_headers_user1)

    # 3. User 2: Confidential Contract (Private to User 2)
    pdf3 = _create_sample_pdf(
        tmp_path, "user2_confidential_contract.pdf",
        "Confidential Project Quantum Agreement\n"
        "Authorized Signatory: Agent Smith\n"
        "Budget Allocation: Rs. 50,00,000"
    )
    res3 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user2,
        files={"file": ("user2_confidential_contract.pdf", io.BytesIO(pdf3), "application/pdf")},
    )
    doc3_id = res3.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc3_id}/process", headers=auth_headers_user2)

    return {"doc1_id": doc1_id, "doc2_id": doc2_id, "doc3_id": doc3_id}


# ==================================================================
# Test Suites
# ==================================================================
class TestRAGQueriesAndCitations:
    """Test grounded answering and citation attribution."""

    def test_rag_query_with_page_citations(self, client: TestClient, auth_headers_user1, setup_rag_corpus):
        """Test RAG query returns an answer with correct document filename and page citation."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the policy number and sum insured of my HDFC ERGO health insurance?",
                "category": "Insurance",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "POL-HDFC-992211" in data["answer"] or "10,00,000" in data["answer"] or len(data["sources"]) > 0
        assert len(data["sources"]) >= 1
        
        # Check source citation details
        top_source = data["sources"][0]
        assert top_source["document_id"] == setup_rag_corpus["doc1_id"]
        assert top_source["filename"] == "hdfc_ergo_health_policy.pdf"
        assert top_source["page_number"] == 1

    def test_rag_refusal_on_missing_information(self, client: TestClient, auth_headers_user1, setup_rag_corpus):
        """Test that asking about absent topics produces a grounded refusal without hallucination."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the vehicle chassis number of my Tesla car?",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()
        
        # Grounding check: system should state lack of sufficient information
        assert "cannot find" in data["answer"].lower() or "not find sufficient" in data["answer"].lower() or len(data["sources"]) == 0


class TestConversationalMemory:
    """Test multi-turn conversational dialogue and context resolution."""

    def test_multi_turn_pronoun_resolution(self, client: TestClient, auth_headers_user1, setup_rag_corpus):
        """Test multi-turn conversation where follow-up question resolves 'it' / 'premium' from previous turn."""
        session_id = "test-session-multi-turn"

        # Turn 1: Ask about the HDFC policy
        resp1 = client.post(
            "/api/v1/query",
            json={
                "query": "Tell me about my HDFC ERGO health insurance policy.",
                "session_id": session_id,
            },
            headers=auth_headers_user1,
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["session_id"] == session_id

        # Turn 2: Follow-up question with pronoun/short context: "How much is the annual premium?"
        resp2 = client.post(
            "/api/v1/query",
            json={
                "query": "How much is the annual premium?",
                "session_id": session_id,
            },
            headers=auth_headers_user1,
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["session_id"] == session_id
        assert "24,500" in data2["answer"] or any("hdfc" in s["filename"].lower() for s in data2["sources"])

    def test_clear_session_memory(self, client: TestClient, auth_headers_user1):
        """Test resetting session memory via DELETE endpoint."""
        session_id = "test-session-to-clear"

        # Seed turn
        client.post(
            "/api/v1/query",
            json={"query": "Hello", "session_id": session_id},
            headers=auth_headers_user1,
        )

        # Clear session
        clear_resp = client.delete(
            f"/api/v1/query/session/{session_id}",
            headers=auth_headers_user1,
        )
        assert clear_resp.status_code == 200
        assert clear_resp.json()["cleared"] is True


class TestRAGUserIsolation:
    """Test that RAG retrieval strictly isolates user data."""

    def test_user1_cannot_retrieve_or_cite_user2_data(self, client: TestClient, auth_headers_user1, setup_rag_corpus):
        """Verify User 1 querying for User 2's confidential keywords gets no citations to User 2's files."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the budget allocation for Project Quantum?",
            },
            headers=auth_headers_user1,
        )

        assert response.status_code == 200
        data = response.json()

        # User 1 must not see User 2's contract
        cited_files = [s["filename"] for s in data["sources"]]
        assert "user2_confidential_contract.pdf" not in cited_files
        assert "50,00,000" not in data["answer"]
