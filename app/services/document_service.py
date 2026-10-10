# ============================================================
# Document Service — Upload, Storage, Metadata & Versioning
# ============================================================
# Manages document persistence, file validation, SHA-256 deduplication,
# user-level storage directory hierarchy, and version tracking.
# ============================================================

import os
import re
import hashlib
import logging
from typing import Optional, List, Tuple
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.config.settings import get_settings
from app.ocr.pipeline import ocr_pipeline
from app.ocr.base import PageResult
from app.extraction.classification import DocumentClassifier
from app.extraction.entities import EntityExtractor
from app.extraction.metadata import MetadataExtractor
from app.embeddings.chunker import PageAwareChunker
from app.vectorstore.chroma import vector_store
from app.models import (
    User,
    Document,
    DocumentVersion,
    Entity,
    DocumentStatus,
    DocumentCategory,
    Tag,
    AuditLog,
)

logger = logging.getLogger(__name__)


# Allowed document extensions and mime types
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/pjpeg",
    "image/tiff",
    "application/octet-stream",  # often sent by generic clients for binary uploads
}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and invalid characters."""
    clean = os.path.basename(filename)
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
    return clean or "document"


def compute_sha256(content: bytes) -> str:
    """Compute SHA-256 checksum of raw file bytes."""
    return hashlib.sha256(content).hexdigest()


class DocumentService:
    """Service handling document storage, validation, versioning, and queries."""

    @staticmethod
    def validate_file(filename: str, content_type: Optional[str], file_bytes: bytes) -> None:
        """Validate file size, extension, and non-emptiness.

        Raises:
            HTTPException 400 or 413 if validation fails.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        settings = get_settings()
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
            # If MIME type is present and strictly disallowed
            if not content_type.startswith("image/") and not content_type == "application/pdf":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported MIME type '{content_type}'.",
                )

    @staticmethod
    def get_user_storage_dir(user_id: int, document_id: int) -> Path:
        """Return the directory path for storing a document's files."""
        settings = get_settings()
        storage_root = Path(settings.STORAGE_DIR)
        doc_dir = storage_root / str(user_id) / str(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    @classmethod
    def upload_document(
        cls,
        db: Session,
        user: User,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        document_id: Optional[int] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[Document, bool]:
        """Process an uploaded document, storing file to disk and creating DB records.

        If ``document_id`` is provided, creates a new version for that document.
        If ``document_id`` is None and an exact SHA-256 hash duplicate exists for this user,
        returns the existing document and ``is_duplicate=True``.

        Returns:
            Tuple of (Document, is_duplicate).
        """
        cls.validate_file(filename, content_type, file_bytes)

        clean_filename = sanitize_filename(filename)
        file_hash = compute_sha256(file_bytes)
        file_size = len(file_bytes)

        # Parse category enum if provided
        cat_enum = None
        if category:
            try:
                cat_enum = DocumentCategory(category)
            except ValueError:
                # Try case-insensitive matching
                for c in DocumentCategory:
                    if c.value.lower() == category.lower():
                        cat_enum = c
                        break

        # -------------------------------------------------------------
        # Case 1: Re-uploading an existing document -> New Version
        # -------------------------------------------------------------
        if document_id is not None:
            existing_doc = (
                db.query(Document)
                .filter(Document.id == document_id, Document.user_id == user.id)
                .first()
            )
            if not existing_doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document with ID {document_id} not found.",
                )

            new_version_num = existing_doc.version + 1
            doc_dir = cls.get_user_storage_dir(user.id, existing_doc.id)
            new_file_path = doc_dir / f"v{new_version_num}_{clean_filename}"

            with open(new_file_path, "wb") as f:
                f.write(file_bytes)

            # Update document record
            existing_doc.version = new_version_num
            existing_doc.filename = clean_filename
            existing_doc.mime_type = content_type or "application/octet-stream"
            existing_doc.file_hash = file_hash
            existing_doc.file_path = str(new_file_path)
            existing_doc.file_size = file_size
            existing_doc.status = DocumentStatus.UPLOADED
            existing_doc.updated_at = datetime.now(timezone.utc)
            if cat_enum:
                existing_doc.category = cat_enum

            # Create DocumentVersion history entry
            version_entry = DocumentVersion(
                document_id=existing_doc.id,
                version_number=new_version_num,
                file_path=str(new_file_path),
                file_hash=file_hash,
                file_size=file_size,
                changes_summary=f"Uploaded version {new_version_num}",
            )
            db.add(version_entry)

            # Audit log
            audit = AuditLog(
                user_id=user.id,
                action="DOCUMENT_REUPLOAD_VERSION",
                resource_type="document",
                resource_id=str(existing_doc.id),
                details={"version": new_version_num, "filename": clean_filename, "hash": file_hash},
            )
            db.add(audit)
            db.commit()
            db.refresh(existing_doc)

            # Ensure the newly uploaded version is immediately searchable for chat/RAG queries.
            try:
                cls.process_document_ocr(db=db, user_id=user.id, document_id=existing_doc.id)
            except Exception as exc:
                logger.warning(f"Automatic OCR processing failed after re-upload for document {existing_doc.id}: {exc}")

            return existing_doc, False

        # -------------------------------------------------------------
        # Case 2: Brand new upload -> Check for exact hash duplicates
        # -------------------------------------------------------------
        duplicate_doc = (
            db.query(Document)
            .filter(Document.user_id == user.id, Document.file_hash == file_hash)
            .first()
        )
        if duplicate_doc:
            # Audit log
            audit = AuditLog(
                user_id=user.id,
                action="DOCUMENT_UPLOAD_DUPLICATE_DETECTED",
                resource_type="document",
                resource_id=str(duplicate_doc.id),
                details={"duplicate_hash": file_hash, "existing_id": duplicate_doc.id},
            )
            db.add(audit)
            db.commit()
            return duplicate_doc, True

        # Create initial Document record in DB to get its generated ID
        new_doc = Document(
            filename=clean_filename,
            mime_type=content_type or "application/octet-stream",
            file_hash=file_hash,
            file_path="",  # will update once ID is generated
            file_size=file_size,
            category=cat_enum,
            status=DocumentStatus.UPLOADED,
            version=1,
            current_version=True,
            user_id=user.id,
        )
        db.add(new_doc)
        db.flush()  # flushes to populate new_doc.id

        # Write file to storage
        doc_dir = cls.get_user_storage_dir(user.id, new_doc.id)
        saved_file_path = doc_dir / f"v1_{clean_filename}"
        with open(saved_file_path, "wb") as f:
            f.write(file_bytes)

        new_doc.file_path = str(saved_file_path)

        # Create initial version entry (v1)
        v1_entry = DocumentVersion(
            document_id=new_doc.id,
            version_number=1,
            file_path=str(saved_file_path),
            file_hash=file_hash,
            file_size=file_size,
            changes_summary="Initial upload",
        )
        db.add(v1_entry)

        # Add tags if provided
        if tags:
            for tag_name in tags:
                tag_name = tag_name.strip()
                if tag_name:
                    db.add(Tag(name=tag_name, document_id=new_doc.id))

        # Audit log
        audit = AuditLog(
            user_id=user.id,
            action="DOCUMENT_UPLOAD",
            resource_type="document",
            resource_id=str(new_doc.id),
            details={"filename": clean_filename, "hash": file_hash, "size": file_size},
        )
        db.add(audit)
        db.commit()
        db.refresh(new_doc)

        # Immediate OCR + indexing makes uploaded documents usable in chat and retrieval immediately.
        try:
            cls.process_document_ocr(db=db, user_id=user.id, document_id=new_doc.id)
        except Exception as exc:
            logger.warning(f"Automatic OCR processing failed after upload for document {new_doc.id}: {exc}")

        return new_doc, False

    @staticmethod
    def list_documents(
        db: Session,
        user_id: int,
        category: Optional[str] = None,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Document], int]:
        """List documents owned by the user with pagination and filters."""
        query = db.query(Document).filter(Document.user_id == user_id)

        if category:
            for c in DocumentCategory:
                if c.value.lower() == category.lower():
                    query = query.filter(Document.category == c)
                    break

        if status_filter:
            for s in DocumentStatus:
                if s.value.lower() == status_filter.lower():
                    query = query.filter(Document.status == s)
                    break

        if search:
            query = query.filter(Document.filename.ilike(f"%{search}%"))

        total = query.count()
        offset = (page - 1) * page_size
        items = query.order_by(desc(Document.created_at)).offset(offset).limit(page_size).all()
        return items, total

    @staticmethod
    def get_document(db: Session, user_id: int, document_id: int) -> Document:
        """Retrieve a single document owned by user_id.

        Raises:
            HTTPException 404 if not found or not owned by user.
        """
        doc = (
            db.query(Document)
            .filter(Document.id == document_id, Document.user_id == user_id)
            .first()
        )
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found.",
            )
        return doc

    @staticmethod
    def delete_document(db: Session, user_id: int, document_id: int) -> None:
        """Delete a document and its stored files from disk."""
        doc = (
            db.query(Document)
            .filter(Document.id == document_id, Document.user_id == user_id)
            .first()
        )
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found.",
            )

        # Remove files from disk
        doc_dir = Path(doc.file_path).parent if doc.file_path else None
        if doc_dir and doc_dir.exists():
            for child in doc_dir.iterdir():
                try:
                    child.unlink()
                except OSError:
                    pass
            try:
                doc_dir.rmdir()
            except OSError:
                pass

        # Remove from vector store
        try:
            vector_store.delete_document_chunks(document_id=document_id, user_id=user_id)
        except Exception:
            pass

        # Audit log
        audit = AuditLog(
            user_id=user_id,
            action="DOCUMENT_DELETE",
            resource_type="document",
            resource_id=str(document_id),
            details={"filename": doc.filename},
        )
        db.add(audit)
        db.delete(doc)
        db.commit()

    @classmethod
    def process_document_ocr(
        cls,
        db: Session,
        user_id: int,
        document_id: int,
    ) -> Tuple[Document, List[PageResult]]:
        """Run OCR / text extraction on a document and persist results.

        Updates document status:
            UPLOADED -> PROCESSING -> READY (or FAILED on error)
        Populates:
            document.extracted_text, document.page_count
        Indexes:
            Vector chunks in ChromaDB with user_id metadata
        """
        doc = cls.get_document(db=db, user_id=user_id, document_id=document_id)

        # Transition to PROCESSING
        doc.status = DocumentStatus.PROCESSING
        db.commit()
        db.refresh(doc)

        try:
            full_text, page_count, avg_conf, page_results = ocr_pipeline.process_file(doc.file_path)

            doc.extracted_text = full_text
            doc.page_count = page_count

            # 1. Automatic Classification (if not manually overridden with high confidence)
            if not doc.category or doc.classification_confidence is None or doc.classification_confidence < 0.9:
                cls_res = DocumentClassifier.classify(full_text)
                doc.category = cls_res.category
                doc.classification_confidence = cls_res.confidence
                doc.classification_method = cls_res.method

            # 2. Extract & Persist Entities
            # Clear old entities if re-processing
            db.query(Entity).filter(Entity.document_id == doc.id).delete()
            extracted_entities = EntityExtractor.extract_from_pages(page_results)
            for ent in extracted_entities:
                db_entity = Entity(
                    entity_type=ent.entity_type,
                    value=ent.value,
                    confidence=ent.confidence,
                    source_page=ent.source_page,
                    document_id=doc.id,
                )
                db.add(db_entity)

            # 3. Extract Metadata
            metadata = MetadataExtractor.extract_metadata(full_text, extracted_entities)

            # 4. Chunk & Index in Vector Store
            chunker = PageAwareChunker()
            chunks = chunker.create_document_chunks(
                document_id=doc.id,
                user_id=user_id,
                filename=doc.filename,
                category=doc.category.value if doc.category else None,
                pages=page_results,
            )
            try:
                vector_store.delete_document_chunks(doc.id, user_id)
                vector_store.upsert_chunks(chunks)
            except Exception as e:
                logger.warning(f"Vector indexing failed for document {doc.id}: {e}")

            doc.status = DocumentStatus.READY
            doc.updated_at = datetime.now(timezone.utc)

            # Audit log
            audit = AuditLog(
                user_id=user_id,
                action="DOCUMENT_OCR_PROCESSED",
                resource_type="document",
                resource_id=str(doc.id),
                details={
                    "pages": page_count,
                    "confidence": avg_conf,
                    "char_count": len(full_text),
                    "category": doc.category.value if doc.category else None,
                    "entities_count": len(extracted_entities),
                    "chunks_count": len(chunks),
                    "issuer": metadata.issuer,
                },
            )
            db.add(audit)
            db.commit()
            db.refresh(doc)
            return doc, page_results

        except Exception as e:
            doc.status = DocumentStatus.FAILED
            db.commit()
            db.refresh(doc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR text extraction failed: {str(e)}",
            )

