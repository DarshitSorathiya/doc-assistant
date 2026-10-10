# ============================================================
# Hybrid Search Engine (Vector + BM25 + RRF)
# ============================================================
# Combines dense semantic vector retrieval (ChromaDB) with sparse
# keyword search (BM25Okapi) using Reciprocal Rank Fusion (RRF).
# ============================================================

import re
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from rank_bm25 import BM25Plus

from app.models import Document, DocumentStatus, DocumentCategory
from app.api.schemas.search import SearchResultItem
from app.vectorstore.chroma import vector_store
from app.rag.reranker import reranker

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class HybridSearcher:
    """Executes hybrid vector + BM25 retrieval with Reciprocal Rank Fusion."""

    @staticmethod
    def bm25_search(
        db: Session,
        user_id: int,
        query: str,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> List[SearchResultItem]:
        """Execute BM25 keyword search over all documents owned by user_id."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        doc_query = db.query(Document).filter(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY,
            Document.extracted_text.isnot(None),
        )

        if category:
            for c in DocumentCategory:
                if c.value.lower() == category.lower():
                    doc_query = doc_query.filter(Document.category == c)
                    break

        docs = doc_query.all()
        if not docs:
            return []

        # Build corpus chunks from documents
        corpus_items: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []

        for doc in docs:
            paragraphs = [p.strip() for p in doc.extracted_text.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                tokens = _tokenize(para)
                if tokens:
                    tokenized_corpus.append(tokens)
                    corpus_items.append({
                        "document_id": doc.id,
                        "filename": doc.filename,
                        "category": doc.category.value if doc.category else None,
                        "text": para,
                        "chunk_id": f"{doc.id}_bm25_p{p_idx}",
                        "page_number": 1,
                    })

        if not tokenized_corpus:
            return []

        bm25 = BM25Plus(tokenized_corpus)
        doc_scores = bm25.get_scores(query_tokens)

        # Pair items with scores
        scored_items = []
        for idx, score in enumerate(doc_scores):
            # Check if there is any token overlap or positive BM25 score
            has_match = any(qt in tokenized_corpus[idx] for qt in query_tokens)
            if score > 0.0 or has_match:
                item = corpus_items[idx]
                effective_score = max(float(score), 0.01) if has_match else float(score)
                scored_items.append((effective_score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResultItem] = []
        for score, item in scored_items[:top_k]:
            results.append(
                SearchResultItem(
                    document_id=item["document_id"],
                    filename=item["filename"],
                    chunk_id=item["chunk_id"],
                    page_number=item["page_number"],
                    category=item["category"],
                    text=item["text"],
                    score=round(float(score), 3),
                    method="bm25",
                )
            )

        return results

    @classmethod
    def search(
        cls,
        db: Session,
        user_id: int,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
        rrf_k: int = 60,
    ) -> List[SearchResultItem]:
        """Execute full hybrid search (Dense Vector + Sparse BM25 + RRF + Reranker)."""
        # 1. Vector Search
        vector_hits = vector_store.query_similar(
            query_text=query,
            user_id=user_id,
            top_k=top_k * 2,
            category=category,
        )

        vector_items = [
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
            for hit in vector_hits
        ]

        # 2. BM25 Search
        bm25_items = cls.bm25_search(
            db=db,
            user_id=user_id,
            query=query,
            category=category,
            top_k=top_k * 2,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_score(d) = sum( 1 / (k + rank(d)) )
        candidate_map: Dict[str, SearchResultItem] = {}
        rrf_scores: Dict[str, float] = {}

        # Process vector ranks
        for rank, item in enumerate(vector_items):
            key = f"{item.document_id}_{item.text[:60]}"
            candidate_map[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Process BM25 ranks
        for rank, item in enumerate(bm25_items):
            key = f"{item.document_id}_{item.text[:60]}"
            if key not in candidate_map:
                candidate_map[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by RRF score
        fused_items: List[SearchResultItem] = []
        for key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            item = candidate_map[key]
            item.score = round(score, 4)
            item.method = "hybrid"
            fused_items.append(item)

        candidates = fused_items[:top_k * 2]

        # 4. Optional Neural Reranker
        if use_reranker and candidates:
            return reranker.rerank(query=query, items=candidates, top_k=top_k)

        return candidates[:top_k]


# Global hybrid searcher instance
hybrid_searcher = HybridSearcher()
