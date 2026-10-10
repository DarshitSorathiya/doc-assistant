# ============================================================
# Phase 4 — OCR & Text Extraction Pipeline Tests
# ============================================================

import io
import pytest
import fitz
from PIL import Image, ImageDraw
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.ocr.base import PageResult
from app.ocr.digital_extractor import DigitalPDFExtractor
from app.ocr.pipeline import OCRPipeline, ocr_pipeline
from app.ocr.paddleocr_engine import PaddleOCREngine
from app.ocr.tesseract_engine import TesseractOCREngine


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
        json={"email": "ocr_tester@example.com", "password": "password123"},
    )
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "ocr_tester@example.com", "password": "password123"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_digital_pdf(tmp_path):
    """Generate a valid multi-page born-digital PDF file."""
    pdf_path = tmp_path / "sample_policy.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "HEALTH INSURANCE POLICY SCHEDULE\n"
        "Policy Number: POL-987654321\n"
        "Insured Name: John Doe\n"
        "Sum Insured: Rs. 500,000\n"
        "Policy Start Date: 01-Jan-2025\n"
        "Policy Expiry Date: 31-Dec-2025\n"
        "Premium Paid: Rs. 15,000",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "TERMS & CONDITIONS\n"
        "1. Coverage applies to hospitalization expenses.\n"
        "2. Pre-existing disease waiting period is 24 months.\n"
        "Claim Helpline: 1800-123-4567",
    )

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def sample_image(tmp_path):
    """Generate a sample PNG image containing mock document text."""
    img_path = tmp_path / "scanned_pan.png"
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "INCOME TAX DEPARTMENT", fill=(0, 0, 0))
    draw.text((20, 60), "GOVT OF INDIA", fill=(0, 0, 0))
    draw.text((20, 90), "PAN: ABCDE1234F", fill=(0, 0, 0))
    img.save(str(img_path))
    return str(img_path)


# ------------------------------------------------------------------
# 1. Digital PDF Extraction Tests
# ------------------------------------------------------------------
class TestDigitalPDFExtractor:
    def test_is_born_digital_detection(self, sample_digital_pdf):
        assert DigitalPDFExtractor.is_born_digital(sample_digital_pdf) is True

    def test_extract_born_digital_pdf(self, sample_digital_pdf):
        results = DigitalPDFExtractor.extract(sample_digital_pdf)
        assert len(results) == 2
        assert results[0].page_num == 1
        assert "HEALTH INSURANCE POLICY" in results[0].text
        assert "POL-987654321" in results[0].text
        assert results[0].confidence == 1.0
        assert results[0].method == "digital"
        assert len(results[0].layout_blocks) > 0

        assert results[1].page_num == 2
        assert "TERMS & CONDITIONS" in results[1].text


# ------------------------------------------------------------------
# 2. Unified OCR Pipeline Tests & Fallback
# ------------------------------------------------------------------
class TestOCRPipeline:
    def test_pipeline_on_digital_pdf(self, sample_digital_pdf):
        pipeline = OCRPipeline()
        full_text, page_count, avg_conf, pages = pipeline.process_file(sample_digital_pdf)
        assert page_count == 2
        assert avg_conf == 1.0
        assert "HEALTH INSURANCE POLICY" in full_text
        assert "--- Page 1 ---" in full_text
        assert "--- Page 2 ---" in full_text

    def test_pipeline_fallback_on_primary_engine_failure(self, sample_image):
        """When PaddleOCR fails, the pipeline should fall back gracefully."""
        pipeline = OCRPipeline(preferred_engine="paddleocr")

        # Mock PaddleOCR to raise an exception
        with patch.object(
            PaddleOCREngine, "extract_file", side_effect=RuntimeError("PaddleOCR error")
        ):
            # Should catch exception and try fallback or graceful scrape
            results = pipeline.extract_document(sample_image)
            assert len(results) >= 1
            assert isinstance(results[0], PageResult)


# ------------------------------------------------------------------
# 3. End-to-End OCR Processing Route Tests
# ------------------------------------------------------------------
class TestDocumentOCRRoute:
    def test_process_document_endpoint_success(
        self, client: TestClient, auth_headers, sample_digital_pdf
    ):
        with open(sample_digital_pdf, "rb") as f:
            pdf_bytes = f.read()

        # 1. Upload document
        upload_res = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files={"file": ("policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"category": "Insurance"},
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document"]["id"]

        # 2. Trigger OCR / text extraction
        proc_res = client.post(
            f"/api/v1/documents/{doc_id}/process",
            headers=auth_headers,
        )
        assert proc_res.status_code == 200
        proc_data = proc_res.json()
        assert proc_data["status"] == "ready"
        assert proc_data["page_count"] == 2
        assert proc_data["extracted_text_preview"] is not None
        assert "HEALTH INSURANCE" in proc_data["extracted_text_preview"]

    def test_process_nonexistent_or_unauthorized_document(
        self, client: TestClient, auth_headers
    ):
        res = client.post("/api/v1/documents/99999/process", headers=auth_headers)
        assert res.status_code == 404
