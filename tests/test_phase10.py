# ============================================================
# Phase 10 — Document Services Tests (Summary, Compare, Timeline, Duplicates, Insights)
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
        json={"email": "services_user1@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "services_user1@example.com", "password": "Password123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "services_user2@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "services_user2@example.com", "password": "Password123!"},
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
# Test Setup: Upload Corpus for User 1 & User 2
# ------------------------------------------------------------------
@pytest.fixture
def setup_services_corpus(client: TestClient, auth_headers_user1, auth_headers_user2, tmp_path):
    # 1. User 1: Health Policy 2026
    pdf1 = _create_sample_pdf(
        tmp_path, "hdfc_health_2026.pdf",
        "HDFC ERGO Health Insurance Policy\n"
        "Policy Number: POL-HDFC-101\n"
        "Sum Insured: Rs. 5,00,000\n"
        "Annual Premium: Rs. 15,000\n"
        "Policy Expiry Date: 15-Jun-2026\n"
        "PAN: ABCDE1234F"
    )
    res1 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("hdfc_health_2026.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    doc1_id = res1.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc1_id}/process", headers=auth_headers_user1)

    # 2. User 1: Health Policy 2027 (Higher coverage renewal)
    pdf2 = _create_sample_pdf(
        tmp_path, "hdfc_health_2027.pdf",
        "HDFC ERGO Health Insurance Policy Renewal\n"
        "Policy Number: POL-HDFC-101\n"
        "Sum Insured: Rs. 10,00,000\n"
        "Annual Premium: Rs. 22,000\n"
        "Policy Expiry Date: 15-Jun-2027\n"
        "PAN: ABCDE1234F"
    )
    res2 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("hdfc_health_2027.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    doc2_id = res2.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc2_id}/process", headers=auth_headers_user1)

    # 3. User 1: Bank Statement (Distinct document)
    pdf3 = _create_sample_pdf(
        tmp_path, "icici_bank_statement.pdf",
        "ICICI Bank Savings Account Statement\n"
        "Account Number: 001122334455\n"
        "IFSC Code: ICIC0000011\n"
        "Account Holder: John Doe\n"
        "Closing Balance as of Oct 2026: Rs. 1,45,000"
    )
    res3 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("icici_bank_statement.pdf", io.BytesIO(pdf3), "application/pdf")},
    )
    doc3_id = res3.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc3_id}/process", headers=auth_headers_user1)

    # 4. User 2: Private Document
    pdf4 = _create_sample_pdf(
        tmp_path, "user2_private.pdf",
        "Private Contract Agreement\n"
        "Policy Number: POL-SECRET-999\n"
        "Expiry Date: 01-Jan-2030"
    )
    res4 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user2,
        files={"file": ("user2_private.pdf", io.BytesIO(pdf4), "application/pdf")},
    )
    doc4_id = res4.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc4_id}/process", headers=auth_headers_user2)

    return {
        "doc1_id": doc1_id,
        "doc2_id": doc2_id,
        "doc3_id": doc3_id,
        "doc4_id": doc4_id,
    }


# ==================================================================
# Test Suites
# ==================================================================
class TestSummaryService:
    """Test single-document and category-level summarization."""

    def test_single_document_summary(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        doc1_id = setup_services_corpus["doc1_id"]
        response = client.post(
            "/api/v1/summary",
            json={"document_id": doc1_id},
            headers=auth_headers_user1,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc1_id
        assert "hdfc_health_2026.pdf" in data["filename"]
        assert len(data["summary"]) > 20
        assert data["category"] == "Insurance"

    def test_category_cross_document_summary(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        response = client.post(
            "/api/v1/summary",
            json={"category": "Insurance"},
            headers=auth_headers_user1,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] >= 2
        assert len(data["items"]) >= 2
        assert "Summary of" in data["summary"]


class TestComparisonService:
    """Test policy & document version comparison."""

    def test_compare_two_policies(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        doc1_id = setup_services_corpus["doc1_id"]
        doc2_id = setup_services_corpus["doc2_id"]

        response = client.post(
            "/api/v1/compare",
            json={"doc_id_1": doc1_id, "doc_id_2": doc2_id},
            headers=auth_headers_user1,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["document_1"]["id"] == doc1_id
        assert data["document_2"]["id"] == doc2_id
        assert "differences" in data
        assert len(data["differences"]) >= 1


class TestTimelineService:
    """Test chronological date event extraction."""

    def test_extract_timeline(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        response = client.post(
            "/api/v1/timeline",
            json={"sort_order": "asc"},
            headers=auth_headers_user1,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total_events"] >= 2
        events = data["events"]
        assert all("date_string" in ev and "filename" in ev for ev in events)


class TestDuplicatesService:
    """Test exact hash and near duplicate detection."""

    def test_find_exact_and_near_duplicates(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        response = client.post(
            "/api/v1/duplicates",
            json={"threshold": 0.65},
            headers=auth_headers_user1,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents_scanned"] >= 3
        assert "exact_duplicates" in data
        assert "near_duplicates" in data
        assert data["near_duplicate_pairs"] >= 1


class TestInsightsService:
    """Test aggregate portfolio analytics."""

    def test_portfolio_insights(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        response = client.post(
            "/api/v1/insights",
            headers=auth_headers_user1,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total_documents"] >= 3
        assert data["total_pages"] >= 3
        assert "Insurance" in data["category_distribution"]
        assert data["expiring_documents_count"] >= 2


class TestServicesUserIsolation:
    """Test multi-tenant isolation across all document services."""

    def test_user1_cannot_summarize_or_compare_user2_document(self, client: TestClient, auth_headers_user1, setup_services_corpus):
        doc4_id = setup_services_corpus["doc4_id"]

        # Summarize attempt on other user's doc
        res_summary = client.post(
            "/api/v1/summary",
            json={"document_id": doc4_id},
            headers=auth_headers_user1,
        )
        assert res_summary.status_code == 404

        # Compare attempt
        res_compare = client.post(
            "/api/v1/compare",
            json={"doc_id_1": setup_services_corpus["doc1_id"], "doc_id_2": doc4_id},
            headers=auth_headers_user1,
        )
        assert res_compare.status_code == 404
