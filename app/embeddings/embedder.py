# ============================================================
# Sentence Transformers Embedding Wrapper
# ============================================================
# Generates dense semantic vector embeddings for chunks and queries.
# ============================================================

import math
import hashlib
import logging
from typing import List, Optional
import numpy as np

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vector lists."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class Embedder:
    """Configurable embedding generator using sentence-transformers with graceful fallback."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._initialized = False

    def _get_model(self):
        """Lazy load sentence-transformers model."""
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._initialized = True
            except Exception as e:
                logger.warning(f"Could not load sentence-transformers model '{self.model_name}': {e}. Using fallback embedder.")
                self._model = None
                self._initialized = True
        return self._model

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """Deterministic pseudo-embedding for testing or offline execution without weights."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            # Hash word into bucket indices
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        model = self._get_model()
        if model is not None:
            emb = model.encode(text, normalize_embeddings=True)
            return emb.tolist()
        return self._fallback_embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of strings."""
        if not texts:
            return []
        model = self._get_model()
        if model is not None:
            embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embs.tolist()
        return [self._fallback_embed(t) for t in texts]

    @staticmethod
    def similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        return _cosine_similarity(vec1, vec2)


# Global embedder instance
embedder = Embedder()
