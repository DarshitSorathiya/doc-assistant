# ============================================================
# Phase 2 — Database Models, Auth & RBAC Tests
# ============================================================

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import User, UserRole, Document, DocumentStatus, DocumentCategory, Entity, Tag, AuditLog
from app.auth.security import hash_password, verify_password
from app.auth.jwt import create_access_token, require_role
from fastapi import Depends, APIRouter


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
def db_session():
    """Yield a standalone SQLAlchemy session for unit tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


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
# 1. Password Hashing & Verification Unit Tests
# ------------------------------------------------------------------
class TestSecurity:
    def test_password_hash_and_verify(self):
        password = "SuperSecretPassword123!"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("WrongPassword", hashed) is False


# ------------------------------------------------------------------
# 2. JWT Generation & Payload Tests
# ------------------------------------------------------------------
class TestJWT:
    def test_create_access_token(self):
        token = create_access_token(data={"sub": "user@example.com", "role": "user"})
        assert isinstance(token, str)
        assert len(token) > 20


# ------------------------------------------------------------------
# 3. User Registration Endpoint Tests
# ------------------------------------------------------------------
class TestAuthRegister:
    def test_register_success(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "newuser@example.com", "password": "securepassword123"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["role"] == "user"
        assert data["is_active"] is True
        assert "id" in data

    def test_register_duplicate_email(self, client: TestClient):
        payload = {"email": "duplicate@example.com", "password": "securepassword123"}
        client.post("/api/v1/auth/register", json=payload)
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_register_with_custom_role(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "admin@example.com", "password": "securepassword123", "role": "admin"},
        )
        assert response.status_code == 201
        assert response.json()["role"] == "admin"

    def test_register_invalid_role(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "badrole@example.com", "password": "securepassword123", "role": "superman"},
        )
        assert response.status_code == 422


# ------------------------------------------------------------------
# 4. User Login Endpoint Tests
# ------------------------------------------------------------------
class TestAuthLogin:
    def test_login_success(self, client: TestClient):
        client.post(
            "/api/v1/auth/register",
            json={"email": "loginuser@example.com", "password": "mypassword"},
        )
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "loginuser@example.com", "password": "mypassword"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "loginuser@example.com"

    def test_login_wrong_password(self, client: TestClient):
        client.post(
            "/api/v1/auth/register",
            json={"email": "wrongpwd@example.com", "password": "correctpassword"},
        )
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "wrongpwd@example.com", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "password"},
        )
        assert response.status_code == 401


# ------------------------------------------------------------------
# 5. Protected Route (/me) Tests
# ------------------------------------------------------------------
class TestAuthMe:
    def test_get_me_authenticated(self, client: TestClient):
        client.post(
            "/api/v1/auth/register",
            json={"email": "me@example.com", "password": "password123"},
        )
        login_res = client.post(
            "/api/v1/auth/login",
            json={"email": "me@example.com", "password": "password123"},
        )
        token = login_res.json()["access_token"]

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "me@example.com"

    def test_get_me_unauthenticated(self, client: TestClient):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


# ------------------------------------------------------------------
# 6. RBAC Role Enforcement Tests
# ------------------------------------------------------------------
class TestRBAC:
    def test_require_role_allows_permitted_role(self, client: TestClient):
        # Register and login an admin
        client.post(
            "/api/v1/auth/register",
            json={"email": "rbac_admin@example.com", "password": "password123", "role": "admin"},
        )
        login_res = client.post(
            "/api/v1/auth/login",
            json={"email": "rbac_admin@example.com", "password": "password123"},
        )
        admin_token = login_res.json()["access_token"]

        # Create temporary test router to test require_role
        test_router = APIRouter(prefix="/api/v1/test-rbac")
        @test_router.get("/admin-only")
        def admin_route(user: User = Depends(require_role(UserRole.ADMIN))):
            return {"message": "welcome admin"}

        app.include_router(test_router)

        # Admin access should succeed
        res = client.get(
            "/api/v1/test-rbac/admin-only",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        assert res.json()["message"] == "welcome admin"

    def test_require_role_forbids_non_permitted_role(self, client: TestClient):
        # Register and login a standard user
        client.post(
            "/api/v1/auth/register",
            json={"email": "rbac_user@example.com", "password": "password123", "role": "user"},
        )
        login_res = client.post(
            "/api/v1/auth/login",
            json={"email": "rbac_user@example.com", "password": "password123"},
        )
        user_token = login_res.json()["access_token"]

        # Standard user accessing admin-only should receive 403 Forbidden
        res = client.get(
            "/api/v1/test-rbac/admin-only",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert res.status_code == 403
        assert "not authorized" in res.json()["detail"]


# ------------------------------------------------------------------
# 7. Database Models & Relationships Tests
# ------------------------------------------------------------------
class TestDatabaseModels:
    def test_user_document_relationship(self, db_session):
        user = User(email="docowner@example.com", hashed_password="pwd", role=UserRole.USER)
        db_session.add(user)
        db_session.commit()

        doc = Document(
            filename="passport.pdf",
            mime_type="application/pdf",
            file_hash="dummyhash123",
            file_path="/data/docs/passport.pdf",
            file_size=1024,
            category=DocumentCategory.GOVERNMENT_ID,
            status=DocumentStatus.UPLOADED,
            user_id=user.id,
        )
        db_session.add(doc)
        db_session.commit()

        entity = Entity(entity_type="PASSPORT_NO", value="A12345678", confidence=0.98, document_id=doc.id)
        tag = Tag(name="identification", document_id=doc.id)
        audit = AuditLog(user_id=user.id, action="UPLOAD", resource_type="document", resource_id=str(doc.id))

        db_session.add_all([entity, tag, audit])
        db_session.commit()

        # Query back
        fetched_user = db_session.query(User).filter_by(email="docowner@example.com").first()
        assert len(fetched_user.documents) == 1
        assert fetched_user.documents[0].filename == "passport.pdf"
        assert len(fetched_user.documents[0].entities) == 1
        assert fetched_user.documents[0].entities[0].value == "A12345678"
        assert len(fetched_user.documents[0].tags) == 1
        assert len(fetched_user.audit_logs) == 1

