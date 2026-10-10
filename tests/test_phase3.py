# ============================================================
# Phase 3 — Document Storage, Ingestion & Metadata Tests
# ============================================================

import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import User, UserRole, Document, DocumentStatus, DocumentCategory
from app.auth.security import hash_password
from app.auth.jwt import create_access_token


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
    """Register and login User 1, returning Bearer auth headers."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "user1@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "user1@example.com", "password": "password123"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2(client: TestClient):
    """Register and login User 2, returning Bearer auth headers."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "user2@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "user2@example.com", "password": "password123"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------
# 1. Document Upload Tests
# ------------------------------------------------------------------
class TestDocumentUpload:
    def test_upload_pdf_success(self, client: TestClient, auth_headers_user1):
        file_content = b"%PDF-1.4 Mock PDF content for test"
        files = {"file": ("test_doc.pdf", io.BytesIO(file_content), "application/pdf")}
        data = {"category": "Insurance", "tags": "health,2026"}

        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files=files,
            data=data,
        )
        assert response.status_code == 201
        res_data = response.json()
        assert res_data["is_duplicate"] is False
        doc = res_data["document"]
        assert doc["filename"] == "test_doc.pdf"
        assert doc["mime_type"] == "application/pdf"
        assert doc["category"] == "Insurance"
        assert doc["status"] == "uploaded"
        assert doc["version"] == 1
        assert len(doc["tags"]) == 2

    def test_upload_image_success(self, client: TestClient, auth_headers_user1):
        file_content = b"\x89PNG\r\n\x1a\nMock PNG bytes"
        files = {"file": ("id_card.png", io.BytesIO(file_content), "image/png")}

        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files=files,
        )
        assert response.status_code == 201
        doc = response.json()["document"]
        assert doc["filename"] == "id_card.png"
        assert doc["mime_type"] == "image/png"

    def test_upload_unsupported_file_type(self, client: TestClient, auth_headers_user1):
        files = {"file": ("malicious.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")}
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files=files,
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_empty_file(self, client: TestClient, auth_headers_user1):
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files=files,
        )
        assert response.status_code == 400
        assert "Uploaded file is empty" in response.json()["detail"]

    def test_exact_duplicate_detection(self, client: TestClient, auth_headers_user1):
        file_content = b"%PDF-1.4 Unique Duplicate Test Content"
        files1 = {"file": ("doc_first.pdf", io.BytesIO(file_content), "application/pdf")}

        res1 = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files=files1,
        )
        assert res1.status_code == 201
        first_id = res1.json()["document"]["id"]
        assert res1.json()["is_duplicate"] is False

        # Upload exact same bytes
        files2 = {"file": ("doc_second.pdf", io.BytesIO(file_content), "application/pdf")}
        res2 = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files=files2,
        )
        assert res2.status_code == 201
        assert res2.json()["is_duplicate"] is True
        assert res2.json()["document"]["id"] == first_id

    def test_document_versioning_reupload(self, client: TestClient, auth_headers_user1):
        # Initial version
        v1_bytes = b"%PDF-1.4 Policy Version 1 content"
        res1 = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("policy.pdf", io.BytesIO(v1_bytes), "application/pdf")},
        )
        assert res1.status_code == 201
        doc_id = res1.json()["document"]["id"]
        assert res1.json()["document"]["version"] == 1

        # Re-upload version 2 tied to same document_id
        v2_bytes = b"%PDF-1.4 Policy Version 2 updated content"
        res2 = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("policy_renewed.pdf", io.BytesIO(v2_bytes), "application/pdf")},
            data={"document_id": doc_id},
        )
        assert res2.status_code == 201
        doc_v2 = res2.json()["document"]
        assert doc_v2["id"] == doc_id
        assert doc_v2["version"] == 2
        assert doc_v2["filename"] == "policy_renewed.pdf"

        # Check detail endpoint returns versions history
        detail_res = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers_user1)
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert len(detail_data["versions"]) == 2
        assert detail_data["versions"][0]["version_number"] == 1
        assert detail_data["versions"][1]["version_number"] == 2


# ------------------------------------------------------------------
# 2. Document Access & User Isolation Tests
# ------------------------------------------------------------------
class TestUserIsolation:
    def test_user_cannot_access_other_users_document(
        self, client: TestClient, auth_headers_user1, auth_headers_user2
    ):
        # User 1 uploads a confidential document
        res = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("tax_return_user1.pdf", io.BytesIO(b"%PDF user1 secret"), "application/pdf")},
        )
        user1_doc_id = res.json()["document"]["id"]

        # User 2 tries to GET User 1's document
        get_res = client.get(f"/api/v1/documents/{user1_doc_id}", headers=auth_headers_user2)
        assert get_res.status_code == 404

        # User 2 tries to download User 1's document
        dl_res = client.get(f"/api/v1/documents/{user1_doc_id}/download", headers=auth_headers_user2)
        assert dl_res.status_code == 404

        # User 2 tries to delete User 1's document
        del_res = client.delete(f"/api/v1/documents/{user1_doc_id}", headers=auth_headers_user2)
        assert del_res.status_code == 404

        # User 2 lists documents -> should be empty
        list_res = client.get("/api/v1/documents", headers=auth_headers_user2)
        assert list_res.status_code == 200
        assert list_res.json()["total"] == 0
        assert len(list_res.json()["items"]) == 0


# ------------------------------------------------------------------
# 3. Document Listing, Filtering & Deletion Tests
# ------------------------------------------------------------------
class TestDocumentListAndManage:
    def test_list_and_filter_documents(self, client: TestClient, auth_headers_user1):
        # Upload 3 different docs
        client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("insurance_policy.pdf", io.BytesIO(b"ins bytes 1"), "application/pdf")},
            data={"category": "Insurance"},
        )
        client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("bank_statement.pdf", io.BytesIO(b"bank bytes 2"), "application/pdf")},
            data={"category": "Banking"},
        )
        client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("passport.jpg", io.BytesIO(b"img bytes 3"), "image/jpeg")},
            data={"category": "Government ID"},
        )

        # List all
        all_res = client.get("/api/v1/documents", headers=auth_headers_user1)
        assert all_res.status_code == 200
        assert all_res.json()["total"] == 3

        # Filter by category
        ins_res = client.get("/api/v1/documents?category=Insurance", headers=auth_headers_user1)
        assert ins_res.status_code == 200
        assert ins_res.json()["total"] == 1
        assert ins_res.json()["items"][0]["filename"] == "insurance_policy.pdf"

        # Search by filename
        search_res = client.get("/api/v1/documents?search=bank", headers=auth_headers_user1)
        assert search_res.status_code == 200
        assert search_res.json()["total"] == 1
        assert search_res.json()["items"][0]["filename"] == "bank_statement.pdf"

        # Pagination
        page_res = client.get("/api/v1/documents?page=1&page_size=2", headers=auth_headers_user1)
        assert page_res.status_code == 200
        assert len(page_res.json()["items"]) == 2
        assert page_res.json()["total_pages"] == 2

    def test_download_and_delete_document(self, client: TestClient, auth_headers_user1):
        raw_bytes = b"%PDF-1.4 Downloadable content"
        upload_res = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers_user1,
            files={"file": ("receipt.pdf", io.BytesIO(raw_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document"]["id"]

        # Download
        dl_res = client.get(f"/api/v1/documents/{doc_id}/download", headers=auth_headers_user1)
        assert dl_res.status_code == 200
        assert dl_res.content == raw_bytes

        # Delete
        del_res = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_user1)
        assert del_res.status_code == 204

        # Confirm 404 after deletion
        get_res = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers_user1)
        assert get_res.status_code == 404
