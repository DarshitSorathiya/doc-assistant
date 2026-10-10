# ============================================================
# Duplicates Service — Exact Hash & Semantic Near-Duplicates
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Document, DocumentStatus
from app.vectorstore.chroma import vector_store

logger = logging.getLogger(__name__)


class DuplicatesService:
    """Finds exact SHA-256 duplicates and semantic near-duplicates."""

    @staticmethod
    def find_duplicates(
        db: Session,
        user_id: int,
        threshold: float = 0.88,
    ) -> Dict[str, Any]:
        """Detect all exact and near-duplicate document pairs for user_id."""
        docs = db.query(Document).filter(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY,
        ).all()

        # 1. Exact SHA-256 Hash Duplicates
        hash_groups: Dict[str, List[Dict[str, Any]]] = {}
        for d in docs:
            hash_groups.setdefault(d.file_hash, []).append({
                "document_id": d.id,
                "filename": d.filename,
                "category": d.category.value if d.category else None,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            })

        exact_duplicates = [
            {"file_hash": h, "count": len(items), "documents": items}
            for h, items in hash_groups.items() if len(items) > 1
        ]

        # 2. Near-Duplicate Vector Matches
        near_duplicates = []
        seen_pairs = set()
        for d1 in docs:
            if d1.extracted_text:
                near_hits = vector_store.find_near_duplicates(
                    document_text=d1.extracted_text,
                    user_id=user_id,
                    threshold=threshold,
                    exclude_doc_id=d1.id,
                )
                for hit in near_hits:
                    hit_id = hit.get("document_id")
                    if hit_id is not None and hit_id != d1.id:
                        pair = tuple(sorted([d1.id, int(hit_id)]))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            near_duplicates.append({
                                "document_1": {"id": d1.id, "filename": d1.filename, "category": d1.category.value if d1.category else None},
                                "document_2": {"id": hit["document_id"], "filename": hit["filename"], "category": hit["category"]},
                                "similarity_score": hit["similarity"],
                            })

        return {
            "total_documents_scanned": len(docs),
            "exact_duplicate_groups": len(exact_duplicates),
            "exact_duplicates": exact_duplicates,
            "near_duplicate_pairs": len(near_duplicates),
            "near_duplicates": near_duplicates,
        }


duplicates_service = DuplicatesService()
