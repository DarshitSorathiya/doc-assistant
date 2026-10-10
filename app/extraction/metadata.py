# ============================================================
# Document Metadata Extraction
# ============================================================
# Extracts high-level document attributes: issuer, issue date,
# expiry date, and structural summary.
# ============================================================

import re
from typing import Optional, List
from pydantic import BaseModel, Field

from app.extraction.entities import ExtractedEntity


class DocumentMetadata(BaseModel):
    """Extracted high-level metadata of a document."""

    issuer: Optional[str] = Field(None, description="Issuing authority or organization")
    document_date: Optional[str] = Field(None, description="Date of document issuance or transaction")
    expiry_date: Optional[str] = Field(None, description="Date of expiry or renewal")
    summary_preview: Optional[str] = Field(None, description="Brief snippet or title")


KNOWN_ISSUERS = [
    ("Income Tax Department", ["income tax department", "govt of india income tax", "central board of direct taxes"]),
    ("UIDAI", ["unique identification authority of india", "uidai", "government of india aadhaar"]),
    ("Ministry of External Affairs", ["passport seva", "ministry of external affairs", "republic of india passport"]),
    ("Transport Department", ["transport department", "motor vehicles department", "rto", "driving licence authority"]),
    ("Election Commission of India", ["election commission of india", "eci", "electoral registration officer"]),
    ("Life Insurance Corporation of India", ["life insurance corporation", "lic of india"]),
    ("Star Health Insurance", ["star health and allied insurance", "star health"]),
    ("HDFC ERGO General Insurance", ["hdfc ergo general insurance", "hdfc ergo"]),
    ("HDFC Bank", ["hdfc bank limited", "hdfc bank"]),
    ("State Bank of India", ["state bank of india", "sbi"]),
    ("ICICI Bank", ["icici bank limited", "icici bank"]),
    ("Axis Bank", ["axis bank limited", "axis bank"]),
]


class MetadataExtractor:
    """Extracts overall document metadata from text and extracted entities."""

    @classmethod
    def extract_metadata(
        cls,
        text: str,
        entities: List[ExtractedEntity],
    ) -> DocumentMetadata:
        """Derive issuer, dates, and preview from raw text and extracted entities."""
        lower_text = text.lower()

        # 1. Identify Issuer
        detected_issuer = None
        for canonical_name, patterns in KNOWN_ISSUERS:
            for pat in patterns:
                if pat in lower_text:
                    detected_issuer = canonical_name
                    break
            if detected_issuer:
                break

        # 2. Identify Expiry Date
        detected_expiry = None
        for ent in entities:
            if ent.entity_type == "Expiry Date":
                detected_expiry = ent.value
                break

        # 3. Identify Document Date (if not expiry date)
        detected_doc_date = None
        for ent in entities:
            if ent.entity_type == "Date" and ent.value != detected_expiry:
                detected_doc_date = ent.value
                break

        # 4. Summary Preview
        clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
        preview = " | ".join(clean_lines[:3]) if clean_lines else None
        if preview and len(preview) > 150:
            preview = preview[:147] + "..."

        return DocumentMetadata(
            issuer=detected_issuer,
            document_date=detected_doc_date,
            expiry_date=detected_expiry,
            summary_preview=preview,
        )
