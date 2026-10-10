# ============================================================
# Timeline Service — Chronological Event Aggregation
# ============================================================

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Document, Entity, DocumentCategory

logger = logging.getLogger(__name__)


def _parse_date(date_str: str) -> Optional[datetime]:
    """Attempt to parse various Indian and international date formats."""
    cleaned = date_str.strip()
    formats = [
        "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%d-%m-%Y",
        "%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y", "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    # Try extracting YYYY
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", cleaned)
    if year_match:
        try:
            return datetime(year=int(year_match.group(1)), month=1, day=1)
        except Exception:
            pass

    return None


class TimelineService:
    """Extracts, parses, and chronologically sorts date events across user documents."""

    @staticmethod
    def extract_timeline(
        db: Session,
        user_id: int,
        category: Optional[str] = None,
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        """Extract and sort all date events across documents owned by user_id."""
        query = (
            db.query(Entity, Document)
            .join(Document, Entity.document_id == Document.id)
            .filter(Document.user_id == user_id)
        )

        if category:
            for c in DocumentCategory:
                if c.value.lower() == category.lower():
                    query = query.filter(Document.category == c)
                    break

        results = query.all()
        events = []

        for ent, doc in results:
            ent_type_lower = ent.entity_type.lower()
            if "date" in ent_type_lower or "expiry" in ent_type_lower or "dob" in ent_type_lower:
                parsed_dt = _parse_date(ent.value)
                events.append({
                    "entity_id": ent.id,
                    "date_string": ent.value,
                    "parsed_timestamp": parsed_dt.isoformat() if parsed_dt else None,
                    "sort_key": parsed_dt or datetime(9999, 1, 1),
                    "event_type": ent.entity_type,
                    "confidence": ent.confidence,
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "category": doc.category.value if doc.category else None,
                    "page": ent.source_page or 1,
                })

        # Sort chronologically
        reverse_sort = sort_order.lower() == "desc"
        events.sort(key=lambda x: x["sort_key"], reverse=reverse_sort)

        # Remove internal sort_key before returning
        for ev in events:
            del ev["sort_key"]

        return {
            "total_events": len(events),
            "category_filter": category,
            "events": events,
        }


timeline_service = TimelineService()
