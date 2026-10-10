# ============================================================
# Schemas for Phase 10 Document Services API Endpoints
# ============================================================

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Summary Schemas
# ------------------------------------------------------------
class SummaryRequest(BaseModel):
    document_id: Optional[int] = Field(None, description="Document ID to summarize (if single-doc)")
    category: Optional[str] = Field(None, description="Category filter (if cross-document summary)")


class SummaryResponse(BaseModel):
    document_id: Optional[int] = None
    filename: Optional[str] = None
    category: Optional[str] = None
    total_documents: Optional[int] = None
    summary: str
    key_entities: Optional[Dict[str, Any]] = None
    items: Optional[List[Dict[str, Any]]] = None


# ------------------------------------------------------------
# Compare Schemas
# ------------------------------------------------------------
class CompareRequest(BaseModel):
    doc_id_1: int = Field(..., description="First document ID")
    doc_id_2: int = Field(..., description="Second document ID")


class CompareResponse(BaseModel):
    document_1: Dict[str, Any]
    document_2: Dict[str, Any]
    differences: List[Dict[str, Any]]
    identical_entities: List[Dict[str, Any]]
    diff_summary: str


# ------------------------------------------------------------
# Timeline Schemas
# ------------------------------------------------------------
class TimelineRequest(BaseModel):
    category: Optional[str] = Field(None, description="Optional category filter")
    sort_order: str = Field(default="asc", description="Sort order: 'asc' or 'desc'")


class TimelineEvent(BaseModel):
    entity_id: int
    date_string: str
    parsed_timestamp: Optional[str] = None
    event_type: str
    confidence: Optional[float] = None
    document_id: int
    filename: str
    category: Optional[str] = None
    page: int


class TimelineResponse(BaseModel):
    total_events: int
    category_filter: Optional[str] = None
    events: List[TimelineEvent]


# ------------------------------------------------------------
# Duplicates Schemas
# ------------------------------------------------------------
class DuplicatesRequest(BaseModel):
    threshold: float = Field(default=0.88, ge=0.5, le=1.0, description="Cosine similarity threshold for near-duplicates")


class DuplicatesResponse(BaseModel):
    total_documents_scanned: int
    exact_duplicate_groups: int
    exact_duplicates: List[Dict[str, Any]]
    near_duplicate_pairs: int
    near_duplicates: List[Dict[str, Any]]


# ------------------------------------------------------------
# Insights Schemas
# ------------------------------------------------------------
class InsightsRequest(BaseModel):
    pass


class InsightsResponse(BaseModel):
    total_documents: int
    total_pages: int
    total_storage_bytes: int
    total_storage_mb: float
    category_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
    entity_counts: Dict[str, int]
    expiring_documents_count: int
    expiring_documents: List[Dict[str, Any]]
