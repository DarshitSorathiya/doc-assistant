# ============================================================
# Document Schemas (Pydantic Models)
# ============================================================

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class TagResponse(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(from_attributes=True)


class DocumentVersionResponse(BaseModel):
    id: int
    document_id: int
    version_number: int
    file_path: str
    file_hash: str
    file_size: int
    changes_summary: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: int
    filename: str
    mime_type: str
    file_hash: str
    file_size: int
    category: Optional[str] = None
    classification_confidence: Optional[float] = None
    classification_method: Optional[str] = None
    status: str
    page_count: Optional[int] = None
    version: int
    current_version: bool
    user_id: int
    created_at: datetime
    updated_at: datetime
    tags: List[TagResponse] = []

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    document: DocumentResponse
    is_duplicate: bool = False
    message: str = "Document uploaded successfully"


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class EntityResponse(BaseModel):
    id: int
    entity_type: str
    value: str
    confidence: Optional[float] = None
    source_page: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    versions: List[DocumentVersionResponse] = []
    entities: List[EntityResponse] = []
    extracted_text_preview: Optional[str] = None

