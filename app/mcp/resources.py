# ============================================================
# MCP Resource Handlers
# ============================================================
# Provides read-access to document data through MCP resource URIs.
# Resources follow the pattern:
#   documents://list              — all user documents
#   documents://{id}              — single document detail
#   documents://{id}/entities     — entities for a document
# ============================================================

import logging
import re
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models import Document, Entity

logger = logging.getLogger(__name__)


class ResourceHandlers:
    """Dispatch MCP resource URI reads to the appropriate handler."""

    def handle(self, uri: str, db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Route a resource URI to the correct handler.

        Args:
            uri: The MCP resource URI (e.g. ``documents://list``).
            db: Active SQLAlchemy session.
            user_id: Authenticated user ID for access control.

        Returns:
            A JSON-serializable dict with the resource data.
        """
        # documents://list
        if uri == "documents://list":
            return self._list_documents(db, user_id)

        # documents://{id}/entities
        m = re.match(r"documents://(\d+)/entities", uri)
        if m:
            doc_id = int(m.group(1))
            return self._document_entities(db, user_id, doc_id)

        # documents://{id}
        m = re.match(r"documents://(\d+)$", uri)
        if m:
            doc_id = int(m.group(1))
            return self._document_detail(db, user_id, doc_id)

        return {"error": f"Unknown resource URI: {uri}"}

    # ----------------------------------------------------------
    # Individual handlers
    # ----------------------------------------------------------

    def _list_documents(self, db: Session, user_id: Optional[int]) -> Dict[str, Any]:
        """Return all documents belonging to the user."""
        query = db.query(Document)
        if user_id:
            query = query.filter(Document.user_id == user_id)
        docs = query.order_by(Document.created_at.desc()).all()

        return {
            "total": len(docs),
            "documents": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "category": d.category.value if d.category else None,
                    "status": d.status.value if d.status else None,
                    "mime_type": d.mime_type,
                    "page_count": d.page_count,
                    "version": d.version,
                    "created_at": str(d.created_at),
                }
                for d in docs
            ],
        }

    def _document_detail(self, db: Session, user_id: Optional[int], doc_id: int) -> Dict[str, Any]:
        """Return full details for a single document."""
        query = db.query(Document).filter(Document.id == doc_id)
        if user_id:
            query = query.filter(Document.user_id == user_id)
        doc = query.first()

        if not doc:
            return {"error": f"Document {doc_id} not found or access denied"}

        return {
            "id": doc.id,
            "filename": doc.filename,
            "mime_type": doc.mime_type,
            "file_hash": doc.file_hash,
            "file_size": doc.file_size,
            "category": doc.category.value if doc.category else None,
            "classification_confidence": doc.classification_confidence,
            "classification_method": doc.classification_method,
            "status": doc.status.value if doc.status else None,
            "page_count": doc.page_count,
            "extracted_text": (doc.extracted_text or "")[:2000],
            "version": doc.version,
            "created_at": str(doc.created_at),
            "updated_at": str(doc.updated_at),
        }

    def _document_entities(self, db: Session, user_id: Optional[int], doc_id: int) -> Dict[str, Any]:
        """Return extracted entities for a document."""
        # Verify document access
        query = db.query(Document).filter(Document.id == doc_id)
        if user_id:
            query = query.filter(Document.user_id == user_id)
        doc = query.first()

        if not doc:
            return {"error": f"Document {doc_id} not found or access denied"}

        entities = db.query(Entity).filter(Entity.document_id == doc_id).all()

        return {
            "document_id": doc_id,
            "document_filename": doc.filename,
            "total_entities": len(entities),
            "entities": [
                {
                    "id": e.id,
                    "entity_type": e.entity_type,
                    "value": e.value,
                    "confidence": e.confidence,
                    "source_page": e.source_page,
                }
                for e in entities
            ],
        }


# Singleton instance
_handlers = ResourceHandlers()


def get_resource_handlers() -> ResourceHandlers:
    """Return the singleton ResourceHandlers instance."""
    return _handlers
