# ============================================================
# Phase 5 — Classification & Entity Extraction Tests
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
from app.models import DocumentCategory, Entity
from app.extraction.classification import DocumentClassifier
from app.extraction.entities import EntityExtractor, validate_verhoeff
from app.extraction.metadata import MetadataExtractor


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
def auth_headers(client: TestClient):
    """Register and login a user, returning Bearer auth headers."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "extraction_user@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "extraction_user@example.com", "password": "password123"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------
# 1. Document Classification Unit Tests
# ------------------------------------------------------------------
class TestDocumentClassification:
    def test_classify_insurance_text(self):
        text = (
            "Star Health and Allied Insurance Policy Schedule.\n"
            "Policy Number: POL-99887766\n"
            "Sum Insured: Rs. 1,000,000\n"
            "Premium Paid: Rs. 18,500\n"
            "Claim Helpline: 1800-425-2255"
        )
        res = DocumentClassifier.classify(text)
        assert res.category == DocumentCategory.INSURANCE
        assert res.confidence >= 0.70
        assert res.method in ("rule", "hybrid")

    def test_classify_banking_text(self):
        text = (
            "HDFC Bank Account Statement\n"
            "Account Number: 50100234567890\n"
            "IFSC Code: HDFC0001234\n"
            "Opening Balance: Rs. 45,000 | Closing Balance: Rs. 62,300\n"
            "Withdrawal: Rs. 5,000 | Deposit: Rs. 22,300"
        )
        res = DocumentClassifier.classify(text)
        assert res.category == DocumentCategory.BANKING
        assert res.confidence >= 0.70

    def test_classify_government_id_text(self):
        text = (
            "INCOME TAX DEPARTMENT\n"
            "GOVT OF INDIA\n"
            "Permanent Account Number Card\n"
            "PAN: ABCDE1234F\n"
            "Name: Rajesh Sharma"
        )
        res = DocumentClassifier.classify(text)
        assert res.category == DocumentCategory.GOVERNMENT_ID
        assert res.confidence >= 0.70

    def test_classify_tax_text(self):
        text = (
            "Indian Income Tax Return Verification Form (ITR-V)\n"
            "Assessment Year: 2024-25\n"
            "Gross Total Income: Rs. 1,250,000\n"
            "Total Income Tax Payable: Rs. 115,000\n"
            "Central Board of Direct Taxes (CBDT)"
        )
        res = DocumentClassifier.classify(text)
        assert res.category == DocumentCategory.TAX
        assert res.confidence >= 0.70

    def test_classify_utility_text(self):
        text = (
            "BSES Yamuna Power Limited - Electricity Bill\n"
            "Consumer Number: CA100293847\n"
            "Meter Number: MTR-987654\n"
            "Units Consumed: 340 kWh\n"
            "Due Date: 15-Nov-2025\n"
            "Amount Due: Rs. 2,450"
        )
        res = DocumentClassifier.classify(text)
        assert res.category == DocumentCategory.UTILITY
        assert res.confidence >= 0.70


# ------------------------------------------------------------------
# 2. Entity Extraction Unit Tests (Indian Identifiers)
# ------------------------------------------------------------------
class TestEntityExtraction:
    def test_extract_pan_number(self):
        text = "Taxpayer PAN is ABCDE1234F for assessment."
        entities = EntityExtractor.extract_from_text(text)
        pan_ents = [e for e in entities if e.entity_type == "PAN"]
        assert len(pan_ents) == 1
        assert pan_ents[0].value == "ABCDE1234F"
        assert pan_ents[0].confidence >= 0.90

    def test_extract_aadhaar_number(self):
        # 3675 9834 6012 is a valid Verhoeff-compliant dummy Aadhaar
        text = "Resident Aadhaar: 3675 9834 6012 verified."
        entities = EntityExtractor.extract_from_text(text)
        aadhaar_ents = [e for e in entities if e.entity_type == "Aadhaar"]
        assert len(aadhaar_ents) == 1
        assert aadhaar_ents[0].value == "3675 9834 6012"

    def test_extract_ifsc_and_account(self):
        text = "Bank: HDFC Bank | IFSC: HDFC0001234 | Account Number: 50100456789012"
        entities = EntityExtractor.extract_from_text(text)
        ifsc_ents = [e for e in entities if e.entity_type == "IFSC"]
        acc_ents = [e for e in entities if e.entity_type == "Bank Account Number"]
        assert len(ifsc_ents) == 1
        assert ifsc_ents[0].value == "HDFC0001234"
        assert len(acc_ents) == 1
        assert acc_ents[0].value == "50100456789012"

    def test_extract_passport_and_license(self):
        text = "Passport No: Z1234567 | Driving License: DL-1420110012345"
        entities = EntityExtractor.extract_from_text(text)
        pass_ents = [e for e in entities if e.entity_type == "Passport"]
        dl_ents = [e for e in entities if e.entity_type == "Driving License"]
        assert len(pass_ents) == 1
        assert pass_ents[0].value == "Z1234567"
        assert len(dl_ents) == 1
        assert "DL" in dl_ents[0].value

    def test_extract_expiry_dates_and_amounts(self):
        text = (
            "Policy Number: POL-123456\n"
            "Sum Insured: Rs. 500,000\n"
            "Valid till: 31-Dec-2025\n"
            "Contact: support@starhealth.in or +91 9876543210"
        )
        entities = EntityExtractor.extract_from_text(text)
        pol_ents = [e for e in entities if e.entity_type == "Policy Number"]
        expiry_ents = [e for e in entities if e.entity_type == "Expiry Date"]
        amt_ents = [e for e in entities if e.entity_type == "Amount"]
        email_ents = [e for e in entities if e.entity_type == "Email"]
        phone_ents = [e for e in entities if e.entity_type == "Phone Number"]

        assert len(pol_ents) == 1
        assert pol_ents[0].value == "POL-123456"
        assert len(expiry_ents) == 1
        assert expiry_ents[0].value == "31-Dec-2025"
        assert len(amt_ents) >= 1
        assert "500,000" in [a.value for a in amt_ents]
        assert len(email_ents) == 1
        assert email_ents[0].value == "support@starhealth.in"
        assert len(phone_ents) == 1
        assert phone_ents[0].value == "9876543210"


# ------------------------------------------------------------------
# 3. Metadata Extractor Tests
# ------------------------------------------------------------------
class TestMetadataExtraction:
    def test_extract_issuer_and_dates(self):
        text = (
            "Income Tax Department, Government of India\n"
            "Permanent Account Number Card\n"
            "PAN: ABCDE1234F\n"
            "Date of Issue: 12-04-2021"
        )
        entities = EntityExtractor.extract_from_text(text)
        meta = MetadataExtractor.extract_metadata(text, entities)
        assert meta.issuer == "Income Tax Department"
        assert meta.document_date == "12-04-2021"


# ------------------------------------------------------------------
# 4. End-to-End Extraction & DB Persistence via API
# ------------------------------------------------------------------
class TestExtractionWorkflow:
    def test_process_document_auto_classifies_and_saves_entities(
        self, client: TestClient, auth_headers, tmp_path
    ):
        # Create sample PDF
        pdf_path = tmp_path / "lic_policy.pdf"
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text(
            (50, 72),
            "Life Insurance Corporation of India (LIC)\n"
            "Policy Schedule: POL-99881122\n"
            "Sum Insured: Rs. 2,000,000\n"
            "Nominee: Anita Sharma\n"
            "PAN: ABCDE1234F\n"
            "Valid upto: 15-Aug-2035\n"
            "Contact: lic_support@licindia.com",
        )
        doc.save(str(pdf_path))
        doc.close()

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # 1. Upload
        up_res = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files={"file": ("lic_policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert up_res.status_code == 201
        doc_id = up_res.json()["document"]["id"]

        # 2. Process
        proc_res = client.post(f"/api/v1/documents/{doc_id}/process", headers=auth_headers)
        assert proc_res.status_code == 200
        proc_data = proc_res.json()

        # Verify auto-classification
        assert proc_data["category"] == "Insurance"
        assert proc_data["classification_confidence"] >= 0.70

        # Verify extracted entities in response
        entities = proc_data["entities"]
        assert len(entities) >= 3

        entity_types = {e["entity_type"] for e in entities}
        assert "PAN" in entity_types
        assert "Policy Number" in entity_types
        assert "Expiry Date" in entity_types
        assert "Email" in entity_types
