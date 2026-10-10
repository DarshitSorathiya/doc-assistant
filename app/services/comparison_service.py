# ============================================================
# Comparison Service — Document & Version Comparison
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Document, Entity, DocumentVersion

logger = logging.getLogger(__name__)


class ComparisonService:
    """Compares two documents or versions and computes structured diffs."""

    @staticmethod
    def compare_documents(
        db: Session,
        user_id: int,
        doc_id_1: int,
        doc_id_2: int,
    ) -> Dict[str, Any]:
        """Compare two documents and highlight entity differences and coverage changes."""
        doc1 = db.query(Document).filter(
            Document.id == doc_id_1,
            Document.user_id == user_id,
        ).first()

        doc2 = db.query(Document).filter(
            Document.id == doc_id_2,
            Document.user_id == user_id,
        ).first()

        if not doc1 or not doc2:
            return {"error": "One or both document IDs not found or unauthorized"}

        if doc1.id == doc2.id:
            return {"error": "Select two different documents to compare."}

        if doc1.status.value != "ready" or doc2.status.value != "ready":
            return {"error": "Both documents must finish processing before they can be compared."}

        ent1_list = db.query(Entity).filter(Entity.document_id == doc_id_1).all()
        ent2_list = db.query(Entity).filter(Entity.document_id == doc_id_2).all()

        ent1_values: Dict[str, List[Entity]] = {}
        ent2_values: Dict[str, List[Entity]] = {}
        for entity in ent1_list:
            ent1_values.setdefault(entity.entity_type, []).append(entity)
        for entity in ent2_list:
            ent2_values.setdefault(entity.entity_type, []).append(entity)

        # Keep every distinct extracted value; a dict keyed only by entity type
        # silently discarded all but the last occurrence (e.g. multiple dates).
        ent1 = {key: sorted({entity.value for entity in values}) for key, values in ent1_values.items()}
        ent2 = {key: sorted({entity.value for entity in values}) for key, values in ent2_values.items()}

        all_types = sorted(list(set(ent1.keys()).union(set(ent2.keys()))))
        differences = []
        identical = []

        fn1_key = f"{doc1.filename} (ID {doc1.id})"
        fn2_key = f"{doc2.filename} (ID {doc2.id})"

        for et in all_types:
            values1 = ent1.get(et, [])
            values2 = ent2.get(et, [])
            v1 = values1[0] if len(values1) == 1 else values1 or None
            v2 = values2[0] if len(values2) == 1 else values2 or None
            if values1 == values2 and values1:
                identical.append({
                    "entity": et,
                    "value": v1,
                    "document_1_pages": sorted({entity.source_page for entity in ent1_values[et] if entity.source_page}),
                    "document_2_pages": sorted({entity.source_page for entity in ent2_values[et] if entity.source_page}),
                })
            else:
                differences.append({
                    "entity": et,
                    fn1_key: v1 or "N/A",
                    fn2_key: v2 or "N/A",
                    "document_1_pages": sorted({entity.source_page for entity in ent1_values.get(et, []) if entity.source_page}),
                    "document_2_pages": sorted({entity.source_page for entity in ent2_values.get(et, []) if entity.source_page}),
                    "document_1_confidence": round(sum(entity.confidence or 0 for entity in ent1_values.get(et, [])) / len(ent1_values[et]), 3) if ent1_values.get(et) else None,
                    "document_2_confidence": round(sum(entity.confidence or 0 for entity in ent2_values.get(et, [])) / len(ent2_values[et]), 3) if ent2_values.get(et) else None,
                })

        return {
            "document_1": {
                "id": doc1.id,
                "filename": doc1.filename,
                "category": doc1.category.value if doc1.category else None,
                "created_at": doc1.created_at.isoformat() if doc1.created_at else None,
            },
            "document_2": {
                "id": doc2.id,
                "filename": doc2.filename,
                "category": doc2.category.value if doc2.category else None,
                "created_at": doc2.created_at.isoformat() if doc2.created_at else None,
            },
            "differences": differences,
            "identical_entities": identical,
            "diff_summary": f"Compared {doc1.filename} with {doc2.filename}: {len(differences)} differences found across {len(all_types)} extracted attributes.",
        }

    @staticmethod
    def compare_versions(
        db: Session,
        user_id: int,
        document_id: int,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """Compare two versions of the same document."""
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id,
        ).first()

        if not doc:
            return {"error": f"Document {document_id} not found or unauthorized"}

        ver_a = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version_a,
        ).first()

        ver_b = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version_b,
        ).first()

        if not ver_a or not ver_b:
            return {"error": "One or both specified version numbers not found"}

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "version_a": {
                "version_number": ver_a.version_number,
                "file_hash": ver_a.file_hash,
                "file_size": ver_a.file_size,
                "created_at": ver_a.created_at.isoformat() if ver_a.created_at else None,
                "changes_summary": ver_a.changes_summary,
            },
            "version_b": {
                "version_number": ver_b.version_number,
                "file_hash": ver_b.file_hash,
                "file_size": ver_b.file_size,
                "created_at": ver_b.created_at.isoformat() if ver_b.created_at else None,
                "changes_summary": ver_b.changes_summary,
            },
            "file_size_difference_bytes": ver_b.file_size - ver_a.file_size,
            "hash_match": ver_a.file_hash == ver_b.file_hash,
        }


comparison_service = ComparisonService()
