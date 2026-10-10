# ============================================================
# Insights Service — Portfolio Analytics & Risk Metrics
# ============================================================

import re
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import Document, DocumentStatus, DocumentCategory, Entity

logger = logging.getLogger(__name__)


class InsightsService:
    """Generates portfolio insights, category distributions, and entity analytics."""

    @staticmethod
    def get_portfolio_insights(
        db: Session,
        user_id: int,
    ) -> Dict[str, Any]:
        """Compute holistic document portfolio analytics for user_id."""
        # Treat portfolio health as two complementary views: all files for
        # processing/inventory status, and ready files for extracted content.
        docs = db.query(Document).filter(Document.user_id == user_id).all()
        ready_docs = [doc for doc in docs if doc.status == DocumentStatus.READY]

        total_docs = len(docs)
        total_storage_bytes = sum(d.file_size for d in docs)
        total_pages = sum(d.page_count or 0 for d in ready_docs)

        # Category Breakdown
        category_counts: Dict[str, int] = {}
        for d in docs:
            cat_name = d.category.value if d.category else "Uncategorized"
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        # Status Breakdown
        status_counts: Dict[str, int] = {}
        for d in docs:
            st = d.status.value
            status_counts[st] = status_counts.get(st, 0) + 1

        # Entity Counts
        entities = (
            db.query(Entity.entity_type, func.count(Entity.id))
            .join(Document, Entity.document_id == Document.id)
            .filter(Document.user_id == user_id, Document.status == DocumentStatus.READY)
            .group_by(Entity.entity_type)
            .all()
        )
        entity_breakdown = {ent_type: count for ent_type, count in entities}

        from sqlalchemy import or_

        # Prefer semantically labeled expiration entities. Some extractors only
        # emit a generic Date entity, so confirm those dates have expiry context
        # in the source text rather than treating every date (e.g. DOB) as expiry.
        expiry_entities = (
            db.query(Entity, Document)
            .join(Document, Entity.document_id == Document.id)
            .filter(
                Document.user_id == user_id,
                Document.status == DocumentStatus.READY,
                or_(
                    Entity.entity_type.ilike("%expiry%"),
                    Entity.entity_type.ilike("%date%"),
                    Entity.entity_type.ilike("%valid%"),
                    Entity.entity_type.ilike("%due%"),
                ),
            )
            .all()
        )

        expiring_by_document_and_date = {}
        for ent, doc in expiry_entities:
            entity_type = ent.entity_type.lower()
            if not any(term in entity_type for term in ("expiry", "valid", "due")):
                source_text = doc.extracted_text or ""
                has_expiry_context = False
                for match in re.finditer(re.escape(ent.value), source_text, re.IGNORECASE):
                    before = source_text[max(0, match.start() - 80):match.start()].lower()
                    if re.search(r"expir|valid|due\s*date|renew(?:al)?\s*by", before):
                        has_expiry_context = True
                        break
                if not has_expiry_context:
                    continue
            key = (doc.id, ent.value.strip().lower())
            expiring_by_document_and_date.setdefault(key, {
                "document_id": doc.id,
                "filename": doc.filename,
                "category": doc.category.value if doc.category else None,
                "expiry_date": ent.value,
            })
        expiring_summary = list(expiring_by_document_and_date.values())

        return {
            "total_documents": total_docs,
            "total_pages": total_pages,
            "total_storage_bytes": total_storage_bytes,
            "total_storage_mb": round(total_storage_bytes / (1024 * 1024), 2),
            "category_distribution": category_counts,
            "status_distribution": status_counts,
            "entity_counts": entity_breakdown,
            "expiring_documents_count": len(expiring_summary),
            "expiring_documents": expiring_summary,
        }


insights_service = InsightsService()
