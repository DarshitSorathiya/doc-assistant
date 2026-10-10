# ============================================================
# Hybrid Search Engine (Vector + BM25 + RRF)
# ============================================================
# Combines dense semantic vector retrieval (ChromaDB) with sparse
# keyword search (BM25Okapi) using Reciprocal Rank Fusion (RRF).
# ============================================================

import re
import logging
import hashlib
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
            matched_category = False
            for c in DocumentCategory:
                if c.value.lower() == category.lower():
                    doc_query = doc_query.filter(Document.category == c)
                    matched_category = True
                    break
            if not matched_category:
                return []

        docs = doc_query.all()
        if not docs:
            return []

        # Build corpus chunks from documents
        corpus_items: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []

        for doc in docs:
            text = doc.extracted_text or ""
            page_sections = re.split(r"(?m)^--- Page (\d+) ---\s*$", text)
            if len(page_sections) > 1:
                pages = [
                    (int(page_sections[index]), page_sections[index + 1])
                    for index in range(1, len(page_sections) - 1, 2)
                ]
            else:
                pages = [(1, text)]

            for page_number, page_text in pages:
                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                for p_idx, para in enumerate(paragraphs):
                    tokens = _tokenize(para)
                    if tokens:
                        tokenized_corpus.append(tokens)
                        corpus_items.append({
                            "document_id": doc.id,
                            "filename": doc.filename,
                            "category": doc.category.value if doc.category else None,
                            "text": para,
                            "chunk_id": f"{doc.id}_bm25_p{page_number}_c{p_idx}",
                            "page_number": page_number,
                        })

        if not tokenized_corpus:
            return []

        bm25 = BM25Plus(tokenized_corpus)
        doc_scores = bm25.get_scores(query_tokens)

        # Pair items with scores
        scored_items = []
        for idx, score in enumerate(doc_scores):
            # BM25Plus has a positive baseline for all documents. Only include
            # passages with actual query-token overlap to prevent unrelated hits.
            has_match = any(qt in tokenized_corpus[idx] for qt in query_tokens)
            if has_match:
                item = corpus_items[idx]
                scored_items.append((float(score), item))

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
        alpha: float = 0.5,
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

        alpha = min(max(alpha, 0.0), 1.0)

        # Process vector ranks, weighted by the caller's dense/sparse preference.
        def candidate_key(item: SearchResultItem) -> str:
            normalized_text = " ".join(item.text.casefold().split())
            text_digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            return f"{item.document_id}_{item.page_number}_{text_digest}"

        for rank, item in enumerate(vector_items):
            key = candidate_key(item)
            candidate_map[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (alpha / (rrf_k + rank + 1))

        # Process BM25 ranks
        for rank, item in enumerate(bm25_items):
            key = candidate_key(item)
            if key not in candidate_map:
                candidate_map[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + ((1.0 - alpha) / (rrf_k + rank + 1))

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
