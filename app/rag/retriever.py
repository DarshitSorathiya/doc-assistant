# ============================================================
# Document Retriever — Unified Search Orchestrator
# ============================================================
# Coordinates keyword, semantic, hybrid, and entity-level search.
# ============================================================

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.models import Document, Entity, DocumentCategory
from app.api.schemas.search import (
    SearchResultItem,
    EntitySearchResultItem,
)
from app.vectorstore.chroma import vector_store
from app.rag.hybrid_search import hybrid_searcher


class DocumentRetriever:
    """Unified document retriever for all search paradigms."""

    @staticmethod
    def keyword_search(
        db: Session,
        user_id: int,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResultItem]:
        """Execute BM25 keyword search."""
        return hybrid_searcher.bm25_search(
            db=db,
            user_id=user_id,
            query=query,
            category=category,
            top_k=top_k,
        )

    @staticmethod
    def semantic_search(
        user_id: int,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResultItem]:
        """Execute dense vector semantic search."""
        hits = vector_store.query_similar(
            query_text=query,
            user_id=user_id,
            top_k=top_k,
            category=category,
        )
        return [
            SearchResultItem(
                document_id=hit["document_id"],
                filename=hit["filename"],
                chunk_id=hit["chunk_id"],
                page_number=hit["page_number"],
                category=hit["category"],
                text=hit["text"],
                score=hit["score"],
                method="vector",
            )
            for hit in hits
        ]

    @staticmethod
    def hybrid_search(
        db: Session,
        user_id: int,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
        alpha: float = 0.5,
    ) -> List[SearchResultItem]:
        """Execute hybrid search (Vector + BM25 + RRF + Reranker)."""
        return hybrid_searcher.search(
            db=db,
            user_id=user_id,
            query=query,
            category=category,
            top_k=top_k,
            use_reranker=use_reranker,
            alpha=alpha,
        )

    @staticmethod
    def entity_search(
        db: Session,
        user_id: int,
        entity_type: Optional[str] = None,
        entity_value: Optional[str] = None,
        top_k: int = 20,
    ) -> List[EntitySearchResultItem]:
        """Search documents containing specific extracted entities (PAN, Aadhaar, IFSC, etc.)."""
        query = (
            db.query(Entity, Document)
            .join(Document, Entity.document_id == Document.id)
            .filter(Document.user_id == user_id)
        )

        if entity_type:
            query = query.filter(Entity.entity_type.ilike(f"%{entity_type.strip()}%"))

        if entity_value:
            query = query.filter(Entity.value.ilike(f"%{entity_value.strip()}%"))

        results = query.limit(top_k).all()

        items: List[EntitySearchResultItem] = []
        for ent, doc in results:
            items.append(
                EntitySearchResultItem(
                    id=ent.id,
                    document_id=doc.id,
                    filename=doc.filename,
                    category=doc.category.value if doc.category else None,
                    entity_type=ent.entity_type,
                    value=ent.value,
                    confidence=ent.confidence,
                    source_page=ent.source_page or 1,
                )
            )

        return items

    def retrieve(
        self,
        db: Session,
        user_id: int,
        query: str,
        mode: str = "hybrid",
        category: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
        alpha: float = 0.5,
    ) -> List[SearchResultItem]:
        """Dispatch query to requested search mode."""
        if mode == "keyword":
            return self.keyword_search(db=db, user_id=user_id, query=query, category=category, top_k=top_k)
        elif mode == "semantic":
            return self.semantic_search(user_id=user_id, query=query, category=category, top_k=top_k)
        else:
            return self.hybrid_search(
                db=db,
                user_id=user_id,
                query=query,
                category=category,
                top_k=top_k,
                use_reranker=use_reranker,
                alpha=alpha,
            )


# Global retriever instance
retriever = DocumentRetriever()
document_retriever = retriever
