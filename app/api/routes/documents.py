# ============================================================
# Document API Routes — Upload, List, Get, Delete, Download
# ============================================================

from typing import Optional, List
from math import ceil

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Form,
    Query,
    status,
    HTTPException,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.jwt import get_current_user
from app.services.document_service import DocumentService
from app.api.schemas.document import (
    DocumentResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentVersionResponse,
    EntityResponse,
    TagResponse,
)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


def _to_doc_response(doc) -> DocumentResponse:
    """Helper to convert ORM Document to Pydantic DocumentResponse."""
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        mime_type=doc.mime_type,
        file_hash=doc.file_hash,
        file_size=doc.file_size,
        category=doc.category.value if doc.category else None,
        classification_confidence=doc.classification_confidence,
        classification_method=doc.classification_method,
        status=doc.status.value,
        page_count=doc.page_count,
        version=doc.version,
        current_version=doc.current_version,
        user_id=doc.user_id,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        tags=[TagResponse(id=t.id, name=t.name) for t in doc.tags] if doc.tags else [],
    )


# ------------------------------------------------------------------
# POST /api/v1/documents/upload
# ------------------------------------------------------------------
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document or new version",
)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[int] = Form(None, description="Provide to re-upload as a new version"),
    category: Optional[str] = Form(None, description="Optional document category"),
    tags: Optional[str] = Form(None, description="Comma-separated tags e.g. 'tax,2024'"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a document (PDF, PNG, JPG, JPEG, TIFF).

    - Computes SHA-256 hash for deduplication.
    - Stores file in user-isolated storage.
    - Supports re-upload versioning when ``document_id`` is supplied.
    """
    file_bytes = await file.read()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    doc, is_duplicate = DocumentService.upload_document(
        db=db,
        user=current_user,
        filename=file.filename or "uploaded_file",
        content_type=file.content_type or "application/octet-stream",
        file_bytes=file_bytes,
        document_id=document_id,
        category=category,
        tags=tag_list,
    )

    msg = "Exact duplicate document already exists" if is_duplicate else (
        f"Document version {doc.version} uploaded successfully" if document_id else "Document uploaded successfully"
    )

    return DocumentUploadResponse(
        document=_to_doc_response(doc),
        is_duplicate=is_duplicate,
        message=msg,
    )


# ------------------------------------------------------------------
# GET /api/v1/documents
# ------------------------------------------------------------------
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List user documents with filtering and pagination",
)
def list_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    search: Optional[str] = Query(None, description="Search in filename"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all documents owned by the authenticated user with optional filters."""
    items, total = DocumentService.list_documents(
        db=db,
        user_id=current_user.id,
        category=category,
        status_filter=status,
        search=search,
        page=page,
        page_size=page_size,
    )

    total_pages = ceil(total / page_size) if total > 0 else 1

    return DocumentListResponse(
        items=[_to_doc_response(d) for d in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ------------------------------------------------------------------
# GET /api/v1/documents/{id}
# ------------------------------------------------------------------
@router.get(
    "/{id}",
    response_model=DocumentDetailResponse,
    summary="Get document details including versions",
)
def get_document(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve full metadata, version history, and tags for a document."""
    doc = DocumentService.get_document(db=db, user_id=current_user.id, document_id=id)

    versions = [
        DocumentVersionResponse(
            id=v.id,
            document_id=v.document_id,
            version_number=v.version_number,
            file_path=v.file_path,
            file_hash=v.file_hash,
            file_size=v.file_size,
            changes_summary=v.changes_summary,
            created_at=v.created_at,
        )
        for v in sorted(doc.versions, key=lambda x: x.version_number)
    ]

    entities = [
        EntityResponse(
            id=e.id,
            entity_type=e.entity_type,
            value=e.value,
            confidence=e.confidence,
            source_page=e.source_page,
            created_at=e.created_at,
        )
        for e in doc.entities
    ]

    base_resp = _to_doc_response(doc)
    return DocumentDetailResponse(
        **base_resp.model_dump(),
        versions=versions,
        entities=entities,
        extracted_text_preview=doc.extracted_text[:200] if doc.extracted_text else None,
    )


# ------------------------------------------------------------------
# DELETE /api/v1/documents/{id}
# ------------------------------------------------------------------
@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its stored files",
)
def delete_document(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a document and clean up all stored version files."""
    DocumentService.delete_document(db=db, user_id=current_user.id, document_id=id)
    return None


# ------------------------------------------------------------------
# POST /api/v1/documents/{id}/process — Trigger OCR & Text Extraction
# ------------------------------------------------------------------
@router.post(
    "/{id}/process",
    response_model=DocumentDetailResponse,
    summary="Trigger OCR and text/table extraction on a document",
)
def process_document(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run OCR and table extraction pipeline on the document.

    - Detects born-digital PDF vs scanned images.
    - Extracts full text, page count, confidence, and structured tables.
    - Sets document status to READY.
    """
    doc, page_results = DocumentService.process_document_ocr(
        db=db, user_id=current_user.id, document_id=id
    )

    versions = [
        DocumentVersionResponse(
            id=v.id,
            document_id=v.document_id,
            version_number=v.version_number,
            file_path=v.file_path,
            file_hash=v.file_hash,
            file_size=v.file_size,
            changes_summary=v.changes_summary,
            created_at=v.created_at,
        )
        for v in sorted(doc.versions, key=lambda x: x.version_number)
    ]

    entities = [
        EntityResponse(
            id=e.id,
            entity_type=e.entity_type,
            value=e.value,
            confidence=e.confidence,
            source_page=e.source_page,
            created_at=e.created_at,
        )
        for e in doc.entities
    ]

    base_resp = _to_doc_response(doc)
    return DocumentDetailResponse(
        **base_resp.model_dump(),
        versions=versions,
        entities=entities,
        extracted_text_preview=doc.extracted_text[:200] if doc.extracted_text else None,
    )


# ------------------------------------------------------------------
# GET /api/v1/documents/{id}/download
# ------------------------------------------------------------------
@router.get(
    "/{id}/download",
    summary="Download the raw document file",
)
def download_document(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download the current file for this document."""
    doc = DocumentService.get_document(db=db, user_id=current_user.id, document_id=id)
    if not doc.file_path:
        raise HTTPException(status_code=404, detail="File path not found for this document")

    return FileResponse(
        path=doc.file_path,
        filename=doc.filename,
        media_type=doc.mime_type,
    )


