# ============================================================
# Phase 7 — Hybrid Search (Vector + BM25 + RRF + Reranking) Tests
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
    yield
    Base.metadata.drop_all(bind=engine)


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


@pytest.fixture
def auth_headers_user1(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "search_user1@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "search_user1@example.com", "password": "password123"},
    )
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture
def auth_headers_user2(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "search_user2@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "search_user2@example.com", "password": "password123"},
    )
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


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
def setup_search_corpus(client: TestClient, auth_headers_user1, auth_headers_user2, tmp_path):
    # 1. User 1: Health Insurance Document
    pdf1 = _create_sample_pdf(
        tmp_path, "health_policy.pdf",
        "Star Health Insurance Comprehensive Policy Schedule\n"
        "Policy Number: POL-77889900\n"
        "Sum Insured: Rs. 1,000,000\n"
        "Insured Name: Aarav Patel\n"
        "PAN: ABCDE1234F\n"
        "Covers intensive care unit (ICU) charges and emergency ambulance expenses."
    )
    res1 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("health_policy.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    doc1_id = res1.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc1_id}/process", headers=auth_headers_user1)

    # 2. User 1: Property Sale Deed Document
    pdf2 = _create_sample_pdf(
        tmp_path, "property_deed.pdf",
        "Sub-Registrar Office Conveyance Deed of Sale\n"
        "Property Address: Flat 402, Greenfield Apartments, Sector 62, Noida\n"
        "Plot Survey Number: SN-90210\n"
        "Total Consideration: Rs. 8,500,000\n"
        "Purchaser: Aarav Patel\n"
        "Khata Certificate Number: KHT-5544"
    )
    res2 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user1,
        files={"file": ("property_deed.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    doc2_id = res2.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc2_id}/process", headers=auth_headers_user1)

    # 3. User 2: Bank Statement (Private to User 2)
    pdf3 = _create_sample_pdf(
        tmp_path, "bank_statement.pdf",
        "State Bank of India Savings Account Statement\n"
        "Account Number: 301234567890\n"
        "IFSC Code: SBIN0001234\n"
        "Account Holder: Priya Singh\n"
        "PAN: WXYZP9876Q\n"
        "Available Balance: Rs. 340,000"
    )
    res3 = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers_user2,
        files={"file": ("bank_statement.pdf", io.BytesIO(pdf3), "application/pdf")},
    )
    doc3_id = res3.json()["document"]["id"]
    client.post(f"/api/v1/documents/{doc3_id}/process", headers=auth_headers_user2)

    return {"user1_doc1": doc1_id, "user1_doc2": doc2_id, "user2_doc3": doc3_id}


# ------------------------------------------------------------------
# 1. Keyword Search (BM25) Tests
# ------------------------------------------------------------------
class TestKeywordSearch:
    def test_bm25_keyword_exact_match(self, client: TestClient, auth_headers_user1, setup_search_corpus):
        res = client.post(
            "/api/v1/search",
            headers=auth_headers_user1,
            json={"query": "POL-77889900", "top_k": 3},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_results"] >= 1
        assert "POL-77889900" in data["items"][0]["text"]
        assert data["items"][0]["filename"] == "health_policy.pdf"

    def test_bm25_with_category_filter(self, client: TestClient, auth_headers_user1, setup_search_corpus):
        # Search with matching category
        res = client.post(
            "/api/v1/search",
            headers=auth_headers_user1,
            json={"query": "Aarav Patel", "category": "Property", "top_k": 3},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_results"] >= 1
        assert data["items"][0]["filename"] == "property_deed.pdf"


# ------------------------------------------------------------------
# 2. Semantic Search Tests
# ------------------------------------------------------------------
class TestSemanticSearch:
    def test_semantic_concept_retrieval(self, client: TestClient, auth_headers_user1, setup_search_corpus):
        # Query uses natural phrasing without exact keywords
        res = client.post(
            "/api/v1/semantic-search",
            headers=auth_headers_user1,
            json={"query": "hospital emergency ambulance medical coverage", "top_k": 3},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_results"] >= 1
        assert data["items"][0]["filename"] == "health_policy.pdf"


# ------------------------------------------------------------------
# 3. Hybrid Search & Reranking Tests
# ------------------------------------------------------------------
class TestHybridSearch:
    def test_hybrid_search_combines_vector_and_bm25(
        self, client: TestClient, auth_headers_user1, setup_search_corpus
    ):
        res = client.post(
            "/api/v1/hybrid-search",
            headers=auth_headers_user1,
            json={"query": "Noida Greenfield apartment house purchase", "top_k": 3, "use_reranker": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_results"] >= 1
        assert data["items"][0]["filename"] == "property_deed.pdf"
        assert data["items"][0]["score"] > 0.0


# ------------------------------------------------------------------
# 4. Entity Search Tests
# ------------------------------------------------------------------
class TestEntitySearch:
    def test_entity_search_by_pan(self, client: TestClient, auth_headers_user1, setup_search_corpus):
        res = client.post(
            "/api/v1/entity-search",
            headers=auth_headers_user1,
            json={"entity_type": "PAN", "entity_value": "ABCDE1234F"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_results"] >= 1
        assert data["items"][0]["value"] == "ABCDE1234F"
        assert data["items"][0]["entity_type"] == "PAN"


# ------------------------------------------------------------------
# 5. Strict User Isolation Tests Across All Search Routes
# ------------------------------------------------------------------
class TestSearchUserIsolation:
    def test_user1_cannot_search_user2_data(
        self, client: TestClient, auth_headers_user1, auth_headers_user2, setup_search_corpus
    ):
        # User 1 searches for User 2's specific bank account / PAN
        kw_res = client.post(
            "/api/v1/search",
            headers=auth_headers_user1,
            json={"query": "SBIN0001234 Priya Singh 301234567890"},
        )
        assert kw_res.json()["total_results"] == 0

        sem_res = client.post(
            "/api/v1/semantic-search",
            headers=auth_headers_user1,
            json={"query": "State Bank of India savings account balance"},
        )
        # Should not return User 2's bank statement
        for item in sem_res.json()["items"]:
            assert item["filename"] != "bank_statement.pdf"

        ent_res = client.post(
            "/api/v1/entity-search",
            headers=auth_headers_user1,
            json={"entity_type": "IFSC", "entity_value": "SBIN0001234"},
        )
        assert ent_res.json()["total_results"] == 0

        # Conversely, User 2 searching for User 2's own IFSC should succeed
        user2_ent_res = client.post(
            "/api/v1/entity-search",
            headers=auth_headers_user2,
            json={"entity_type": "IFSC", "entity_value": "SBIN0001234"},
        )
        assert user2_ent_res.json()["total_results"] == 1
        assert user2_ent_res.json()["items"][0]["value"] == "SBIN0001234"
