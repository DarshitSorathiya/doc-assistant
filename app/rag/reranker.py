# ============================================================
# Cross-Encoder & Neural Reranker
# ============================================================
# Re-scores candidate search passages using cross-attention
# between the query and candidate chunk texts.
# ============================================================

import logging
from typing import List, Optional
import numpy as np

from app.api.schemas.search import SearchResultItem
from app.embeddings.embedder import embedder

logger = logging.getLogger(__name__)


class Reranker:
    """Neural Cross-Encoder reranker with token-relevance fallback."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _get_model(self):
        """Lazy load CrossEncoder model."""
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self._initialized = True
            except Exception as e:
                logger.info(f"CrossEncoder '{self.model_name}' not active ({e}). Using semantic fallback reranker.")
                self._model = None
                self._initialized = True
        return self._model

    def _fallback_rescore(self, query: str, items: List[SearchResultItem]) -> List[SearchResultItem]:
        """Heuristic cross-similarity and token-density rescoring."""
        query_words = set(query.lower().split())
        query_emb = embedder.embed_text(query)

        for item in items:
            text_lower = item.text.lower()
            # 1. Exact token match ratio
            matched_words = sum(1 for w in query_words if w in text_lower)
            token_ratio = matched_words / max(len(query_words), 1)

            # 2. Dense semantic similarity
            chunk_emb = embedder.embed_text(item.text)
            dense_sim = embedder.similarity(query_emb, chunk_emb)

            # Combined rerank score: 60% dense + 40% keyword density
            combined_score = (0.6 * dense_sim) + (0.4 * token_ratio)
            item.score = round(combined_score, 3)
            item.method = "reranked"

        items.sort(key=lambda x: x.score, reverse=True)
        return items

    def rerank(self, query: str, items: List[SearchResultItem], top_k: int = 5) -> List[SearchResultItem]:
        """Rerank candidate items given the user query and return top_k."""
        if not items:
            return []

        model = self._get_model()
        if model is not None:
            pairs = [[query, item.text] for item in items]
            try:
                scores = model.predict(pairs)
                for idx, score in enumerate(scores):
                    # Sigmoid scale logits to [0, 1]
                    norm_score = 1.0 / (1.0 + np.exp(-float(score)))
                    items[idx].score = round(norm_score, 3)
                    items[idx].method = "cross-encoder"

                items.sort(key=lambda x: x.score, reverse=True)
                return items[:top_k]
            except Exception as e:
                logger.warning(f"CrossEncoder prediction failed: {e}")

        # Fallback reranker
        rescored = self._fallback_rescore(query, items)
        return rescored[:top_k]


# Global reranker instance
reranker = Reranker()
