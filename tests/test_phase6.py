# ============================================================
# Phase 6 — Embeddings, Vector Store & Chunking Tests
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
from app.ocr.base import PageResult
from app.embeddings.chunker import PageAwareChunker, DocumentChunk
from app.embeddings.embedder import Embedder, embedder
from app.vectorstore.chroma import ChromaVectorStore, vector_store


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
        json={"email": "vector_user1@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "vector_user1@example.com", "password": "password123"},
    )
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture
def auth_headers_user2(client: TestClient):
    client.post(
        "/api/v1/auth/register",
        json={"email": "vector_user2@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "vector_user2@example.com", "password": "password123"},
    )
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


# ------------------------------------------------------------------
# 1. PageAwareChunker Unit Tests
# ------------------------------------------------------------------
class TestPageAwareChunker:
    def test_chunking_preserves_page_numbers_and_metadata(self):
        chunker = PageAwareChunker(target_chunk_size=150, chunk_overlap=30)
        pages = [
            PageResult(page_num=1, text="Page 1 Content. This is the first section of insurance terms and rules.", confidence=1.0),
            PageResult(page_num=2, text="Page 2 Content. Here are the claim limits, coverage criteria, and hospital list.", confidence=1.0),
        ]

        chunks = chunker.create_document_chunks(
            document_id=10,
            user_id=1,
            filename="policy.pdf",
            category="Insurance",
            pages=pages,
        )

        assert len(chunks) >= 2
        page_nums = {c.page_number for c in chunks}
        assert page_nums == {1, 2}
        assert all(c.user_id == 1 for c in chunks)
        assert all(c.document_id == 10 for c in chunks)
        assert all(c.category == "Insurance" for c in chunks)


# ------------------------------------------------------------------
# 2. Embedder Unit Tests
# ------------------------------------------------------------------
class TestEmbedder:
    def test_embedding_generation_and_cosine_similarity(self):
        emb1 = embedder.embed_text("Health insurance hospitalization coverage")
        emb2 = embedder.embed_text("Medical insurance hospital cash benefit")
        emb3 = embedder.embed_text("Motor vehicle traffic fine receipt")

        assert len(emb1) > 0
        sim_related = Embedder.similarity(emb1, emb2)
        sim_unrelated = Embedder.similarity(emb1, emb3)
        assert sim_related > sim_unrelated


# ------------------------------------------------------------------
# 3. Vector Store & Strict User Isolation Tests
# ------------------------------------------------------------------
class TestChromaVectorStore:
    def test_strict_user_isolation_in_vector_search(self):
        store = ChromaVectorStore(collection_name="test_user_isolation")

        # Chunks for User 1
        user1_chunks = [
            DocumentChunk(
                chunk_id="doc1_p1_c0",
                text="User 1 confidential medical prescription for antibiotics.",
                document_id=1,
                user_id=100,
                page_number=1,
                category="Medical",
                filename="meds.pdf",
            )
        ]
        # Chunks for User 2
        user2_chunks = [
            DocumentChunk(
                chunk_id="doc2_p1_c0",
                text="User 2 confidential tax return document with refunds.",
                document_id=2,
                user_id=200,
                page_number=1,
                category="Tax",
                filename="tax.pdf",
            )
        ]

        store.upsert_chunks(user1_chunks)
        store.upsert_chunks(user2_chunks)

        # User 1 searches: should NEVER see User 2's chunks
        res1 = store.query_similar(query_text="tax return refunds", user_id=100, top_k=5)
        for hit in res1:
            assert hit["metadata"]["user_id"] == 100
            assert hit["document_id"] != 2

        # User 2 searches: should NEVER see User 1's chunks
        res2 = store.query_similar(query_text="medical prescription antibiotics", user_id=200, top_k=5)
        for hit in res2:
            assert hit["metadata"]["user_id"] == 200
            assert hit["document_id"] != 1

    def test_near_duplicate_detection(self):
        store = ChromaVectorStore(collection_name="test_near_duplicates")

        # Original document chunk
        original_chunk = DocumentChunk(
            chunk_id="orig_p1_c0",
            text="HDFC Ergo Health Insurance Policy Schedule 2025. Sum Insured Rs 1,000,000. Premium Paid Rs 15,000.",
            document_id=1,
            user_id=1,
            page_number=1,
            category="Insurance",
            filename="hdfc_v1.pdf",
        )
        store.upsert_chunks([original_chunk])

        # Near-duplicate text (minor renewal year change)
        renewal_text = "HDFC Ergo Health Insurance Policy Schedule 2026. Sum Insured Rs 1,000,000. Premium Paid Rs 16,000."
        dups = store.find_near_duplicates(renewal_text, user_id=1, threshold=0.75, exclude_doc_id=2)
        assert len(dups) >= 1
        assert dups[0]["document_id"] == 1


# ------------------------------------------------------------------
# 4. End-to-End Upload, Processing & Vector Indexing via API
# ------------------------------------------------------------------
class TestVectorIndexingPipeline:
    def test_upload_and_process_indexes_into_vectorstore(
        self, client: TestClient, auth_headers_user1, tmp_path
    ):
        pdf_path = tmp_path / "vehicle_insurance.pdf"
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text(
            (50, 72),
            "Motor Insurance Policy Certificate\n"
            "Vehicle Reg No: DL-01-AB-1234\n"
            "Chassis Number: MAT61234567890123\n"
            "Third Party Liability Cover: Active\n"
            "Valid till: 20-Oct-2027",
        )
        doc.save(str(pdf_path))
        doc.close()

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # 1. Upload
        up_res = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("vehicle_insurance.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert up_res.status_code == 201
        doc_id = up_res.json()["document"]["id"]
        user_id = up_res.json()["document"]["user_id"]

        # 2. Process (runs OCR, classification, entity extraction & vector chunking)
        proc_res = client.post(f"/api/v1/documents/{doc_id}/process", headers=auth_headers_user1)
        assert proc_res.status_code == 200

        # 3. Query Vector Store directly for this user
        search_hits = vector_store.query_similar(
            query_text="Chassis Number MAT61234567890123",
            user_id=user_id,
            top_k=3,
        )
        assert len(search_hits) >= 1
        assert search_hits[0]["document_id"] == doc_id
        assert "Chassis Number" in search_hits[0]["text"]
