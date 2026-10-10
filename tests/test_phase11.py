# ============================================================
# Phase 11 — Knowledge Graph & Proactive Intelligence Tests
# ============================================================

import io
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import User, Document, DocumentStatus, DocumentCategory, Entity, DocumentVersion
from app.auth.security import hash_password
from app.knowledge_graph.graph import kg_manager
from app.knowledge_graph.entities import NodeType
from app.services.alert_service import alert_service, AlertSeverity, AlertType


# ------------------------------------------------------------------
# Test Database Fixture
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
    kg_manager._user_graphs.clear()
    yield
    Base.metadata.drop_all(bind=engine)
    kg_manager._user_graphs.clear()


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
def auth_headers(client):
    """Register and login a default user, returning auth headers."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "kguser@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "kguser@example.com", "password": "Password123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user2_headers(client):
    """Register and login a second user for multi-tenant isolation tests."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "otheruser@example.com", "password": "Password123!", "role": "user"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "otheruser@example.com", "password": "Password123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------
# Phase 11 Tests
# ------------------------------------------------------------------

def test_knowledge_graph_construction_and_query(client, auth_headers):
    """Test building Knowledge Graph from documents and extracted entities."""
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "kguser@example.com").first()

    # Document 1: Health Policy
    doc1 = Document(
        user_id=user.id,
        filename="star_health_policy.pdf",
        file_path="/storage/1/doc1.pdf",
        file_hash="hash_policy_1",
        mime_type="application/pdf",
        file_size=2048,
        category=DocumentCategory.INSURANCE,
        status=DocumentStatus.READY,
        page_count=2,
    )
    db.add(doc1)
    db.flush()

    db.add_all([
        Entity(document_id=doc1.id, entity_type="policy_number", value="POL-998877", confidence=0.95),
        Entity(document_id=doc1.id, entity_type="person_name", value="Rahul Sharma", confidence=0.90),
        Entity(document_id=doc1.id, entity_type="organization", value="Star Health Insurance", confidence=0.92),
    ])

    # Document 2: Property Sale Deed
    doc2 = Document(
        user_id=user.id,
        filename="bangalore_flat_deed.pdf",
        file_path="/storage/1/doc2.pdf",
        file_hash="hash_deed_1",
        mime_type="application/pdf",
        file_size=4096,
        category=DocumentCategory.PROPERTY,
        status=DocumentStatus.READY,
        page_count=4,
    )
    db.add(doc2)
    db.flush()

    db.add_all([
        Entity(document_id=doc2.id, entity_type="property_khata", value="Khata-450-Koramangala", confidence=0.91),
        Entity(document_id=doc2.id, entity_type="person_name", value="Rahul Sharma", confidence=0.90),
    ])

    db.commit()
    db.close()

    # Query full graph via API
    res = client.get("/api/v1/graph", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    assert data["total_nodes"] >= 4
    assert data["total_edges"] >= 3

    node_labels = [n["label"] for n in data["nodes"]]
    assert "star_health_policy.pdf" in node_labels
    assert "bangalore_flat_deed.pdf" in node_labels
    assert "Rahul Sharma" in node_labels

    # Check query filtering by entity query
    res_filtered = client.get("/api/v1/graph?query=Rahul", headers=auth_headers)
    assert res_filtered.status_code == 200
    filtered_data = res_filtered.json()
    assert any("Rahul" in n["label"] for n in filtered_data["nodes"])


def test_knowledge_graph_canonicalizes_entities_and_combines_filters(client, auth_headers):
    """Equivalent entity values should connect, and type+text filters should narrow together."""
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "kguser@example.com").first()
    docs = [
        Document(
            user_id=user.id,
            filename=f"canonical_{index}.pdf",
            file_path=f"/p/canonical_{index}.pdf",
            file_hash=f"canonical_hash_{index}",
            mime_type="application/pdf",
            file_size=100,
            status=DocumentStatus.READY,
        )
        for index in range(2)
    ]
    db.add_all(docs)
    db.flush()
    db.add_all([
        Entity(document_id=docs[0].id, entity_type="policy_number", value="POL-CASE-77"),
        Entity(document_id=docs[1].id, entity_type="policy_number", value="  pol-case-77  "),
        Entity(document_id=docs[1].id, entity_type="person_name", value="Unrelated Person"),
    ])
    db.commit()
    db.close()

    result = client.get(
        "/api/v1/graph",
        params={"entity_type": "Policy", "query": "POL-CASE-77"},
        headers=auth_headers,
    )
    assert result.status_code == 200
    data = result.json()
    policy_nodes = [node for node in data["nodes"] if node["node_type"] == "Policy"]
    assert len(policy_nodes) == 1
    assert len([edge for edge in data["edges"] if edge["relation"] == "has_policy"]) == 2
    assert not any(node.get("label") == "Unrelated Person" for node in data["nodes"])


def test_connected_documents_lookup(client, auth_headers):
    """Test finding all documents connected to a person or entity in the graph."""
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "kguser@example.com").first()

    d1 = Document(user_id=user.id, filename="doc_a.pdf", file_path="/p/a.pdf", file_hash="h1", mime_type="application/pdf", file_size=1024, status=DocumentStatus.READY)
    d2 = Document(user_id=user.id, filename="doc_b.pdf", file_path="/p/b.pdf", file_hash="h2", mime_type="application/pdf", file_size=1024, status=DocumentStatus.READY)
    db.add_all([d1, d2])
    db.flush()

    db.add(Entity(document_id=d1.id, entity_type="person_name", value="Aarav Verma"))
    db.add(Entity(document_id=d2.id, entity_type="person_name", value="Aarav Verma"))
    db.commit()

    connected = kg_manager.get_connected_documents(db, user.id, "Aarav Verma")
    db.close()

    assert len(connected) == 2
    filenames = [c["filename"] for c in connected]
    assert "doc_a.pdf" in filenames
    assert "doc_b.pdf" in filenames


def test_proactive_expiry_alerts(client, auth_headers):
    """Test expiry alerts firing for expired and upcoming expiration dates."""
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "kguser@example.com").first()

    now = datetime(2026, 9, 29)

    # Expired Doc
    d_expired = Document(user_id=user.id, filename="expired_passport.pdf", file_path="/p/exp.pdf", file_hash="hx1", mime_type="application/pdf", file_size=1024, status=DocumentStatus.READY)
    # Expiring in 20 days
    d_soon = Document(user_id=user.id, filename="motor_insurance.pdf", file_path="/p/motor.pdf", file_hash="hx2", mime_type="application/pdf", file_size=1024, status=DocumentStatus.READY)
    # Expiring in 200 days (beyond 60 day warning window)
    d_far = Document(user_id=user.id, filename="term_life.pdf", file_path="/p/term.pdf", file_hash="hx3", mime_type="application/pdf", file_size=1024, status=DocumentStatus.READY)

    db.add_all([d_expired, d_soon, d_far])
    db.flush()

    db.add(Entity(document_id=d_expired.id, entity_type="expiry_date", value="15-Jan-2026"))
    db.add(Entity(document_id=d_soon.id, entity_type="policy_expiry", value="19-Oct-2026"))
    db.add(Entity(document_id=d_far.id, entity_type="expiry_date", value="20-Jun-2027"))
    db.commit()

    alerts_res = alert_service.get_expiry_alerts(db, user.id, warning_days=60, now=now)
    db.close()

    assert len(alerts_res) == 2
    severities = [a["severity"] for a in alerts_res]
    assert AlertSeverity.CRITICAL.value in severities
    assert AlertSeverity.WARNING.value in severities

    filenames = [a["filename"] for a in alerts_res]
    assert "expired_passport.pdf" in filenames
    assert "motor_insurance.pdf" in filenames
    assert "term_life.pdf" not in filenames


def test_missing_document_heuristics(client, auth_headers):
    """Test heuristic detection for missing insurance or government IDs."""
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "kguser@example.com").first()

    # User uploads property document but no insurance
    prop_doc = Document(
        user_id=user.id,
        filename="house_registration.pdf",
        file_path="/p/prop.pdf",
        file_hash="hprop",
        mime_type="application/pdf",
        file_size=1024,
        category=DocumentCategory.PROPERTY,
        extracted_text="Property registration deed for flat 202.",
        status=DocumentStatus.READY,
    )
    db.add(prop_doc)
    db.commit()

    missing_alerts = alert_service.get_missing_document_alerts(db, user.id)
    db.close()

    assert len(missing_alerts) >= 1
    assert any("Insurance" in a["title"] for a in missing_alerts)


def test_alerts_endpoint_and_filtering(client, auth_headers):
    """Test GET /api/v1/alerts endpoint with type filtering."""
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "kguser@example.com").first()

    d = Document(
        user_id=user.id,
        filename="car_dl.pdf",
        file_path="/p/dl.pdf",
        file_hash="hdl",
        mime_type="application/pdf",
        file_size=1024,
        category=DocumentCategory.GOVERNMENT_ID,
        extracted_text="Driving licence holder DL-9988.",
        version=2,
        status=DocumentStatus.READY,
    )
    db.add(d)
    db.commit()
    db.close()

    res = client.get("/api/v1/alerts", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total_alerts"] >= 1

    # Filter by alert_type=version_modified
    res_filtered = client.get("/api/v1/alerts?alert_type=version_modified", headers=auth_headers)
    assert res_filtered.status_code == 200
    filtered_data = res_filtered.json()
    assert len(filtered_data["alerts"]) >= 1
    assert filtered_data["alerts"][0]["alert_type"] == "version_modified"


def test_multi_tenant_isolation_graph_and_alerts(client, auth_headers, user2_headers):
    """Verify User 1 cannot view User 2's knowledge graph or alerts."""
    db = TestingSessionLocal()
    user2 = db.query(User).filter(User.email == "otheruser@example.com").first()

    secret_doc = Document(
        user_id=user2.id,
        filename="confidential_patent.pdf",
        file_path="/p/sec.pdf",
        file_hash="hsec",
        mime_type="application/pdf",
        file_size=1024,
        category=DocumentCategory.LEGAL,
        status=DocumentStatus.READY,
    )
    db.add(secret_doc)
    db.flush()
    db.add(Entity(document_id=secret_doc.id, entity_type="person_name", value="Secret Inventor"))
    db.commit()
    db.close()

    # User 1 queries graph
    res1 = client.get("/api/v1/graph", headers=auth_headers)
    assert res1.status_code == 200
    assert "confidential_patent.pdf" not in [n["label"] for n in res1.json()["nodes"]]
    assert "Secret Inventor" not in [n["label"] for n in res1.json()["nodes"]]

    # User 2 queries graph
    res2 = client.get("/api/v1/graph", headers=user2_headers)
    assert res2.status_code == 200
    assert "confidential_patent.pdf" in [n["label"] for n in res2.json()["nodes"]]
    assert "Secret Inventor" in [n["label"] for n in res2.json()["nodes"]]
