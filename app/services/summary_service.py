# ============================================================
# Summary Service — Single & Cross-Document Summarization
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Document, DocumentStatus, Entity, DocumentCategory

logger = logging.getLogger(__name__)


class SummaryService:
    """Provides single-document and cross-document portfolio summarization."""

    @staticmethod
    def summarize_document(
        db: Session,
        user_id: int,
        document_id: int,
    ) -> Dict[str, Any]:
        """Generate structured summary for a specific document."""
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id,
        ).first()

        if not doc:
            return {"error": f"Document {document_id} not found or unauthorized"}

        entities = db.query(Entity).filter(Entity.document_id == document_id).all()
        entity_map = {e.entity_type: e.value for e in entities}

        text = doc.extracted_text or ""
        lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("---")]
        key_points = lines[:6]

        summary_text = (
            f"**{doc.filename}** ({doc.category.value if doc.category else 'Document'})\n"
            + "\n".join(f"- {p}" for p in key_points)
        )

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "category": doc.category.value if doc.category else None,
            "page_count": doc.page_count,
            "summary": summary_text,
            "key_entities": entity_map,
            "status": doc.status.value,
        }

    @staticmethod
    def summarize_category(
        db: Session,
        user_id: int,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate cross-document summary for all documents in a category."""
        query = db.query(Document).filter(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY,
        )

        if category:
            for c in DocumentCategory:
                if c.value.lower() == category.lower():
                    query = query.filter(Document.category == c)
                    break

        docs = query.all()
        if not docs:
            return {
                "category": category or "All",
                "total_documents": 0,
                "summary": "No documents found for this category.",
                "items": [],
            }

        summaries = []
        for d in docs:
            ents = db.query(Entity).filter(Entity.document_id == d.id).all()
            ent_summary = {e.entity_type: e.value for e in ents}
            summaries.append({
                "document_id": d.id,
                "filename": d.filename,
                "category": d.category.value if d.category else None,
                "entities": ent_summary,
            })

        cross_summary = f"Summary of {len(docs)} documents in '{category or 'All'}':\n"
        for s in summaries:
            cross_summary += f"\n- **{s['filename']}**: " + ", ".join(f"{k}: {v}" for k, v in list(s['entities'].items())[:3])

        return {
            "category": category or "All",
            "total_documents": len(docs),
            "summary": cross_summary,
            "items": summaries,
        }


summary_service = SummaryService()
