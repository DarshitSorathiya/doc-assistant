# ============================================================
# ChromaDB Vector Store Client Wrapper
# ============================================================
# Manages dense vector indexing and retrieval with strict query-level
# user isolation (user_id metadata filter enforced at all times).
# ============================================================

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.config.settings import get_settings
from app.embeddings.chunker import DocumentChunk
from app.embeddings.embedder import embedder

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB wrapper providing secure user-isolated vector indexing and search."""

    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "personal_documents"):
        settings = get_settings()
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._in_memory_fallback: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self):
        """Initialize ChromaDB client or fallback."""
        try:
            import chromadb
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}. Using in-memory vector fallback.")
            self._client = None
            self._collection = None

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed and upsert document chunks into the vector store."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]
        embeddings = embedder.embed_documents(texts)

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            # In-memory fallback
            for i, chunk_id in enumerate(ids):
                self._in_memory_fallback[chunk_id] = {
                    "text": texts[i],
                    "metadata": metadatas[i],
                    "embedding": embeddings[i],
                }

        return len(chunks)

    def query_similar(
        self,
        query_text: str,
        user_id: int,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query vector store for chunks semantically similar to query_text.

        CRITICAL: Always enforces ``user_id == user_id`` in query filters.
        """
        query_emb = embedder.embed_text(query_text)
        results: List[Dict[str, Any]] = []

        if self._collection is not None:
            # Build where clause enforcing user_id
            if category:
                where_filter = {
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"category": {"$eq": category}},
                    ]
                }
            else:
                where_filter = {"user_id": {"$eq": user_id}}

            # Check collection count first to avoid querying empty collection
            if self._collection.count() == 0:
                return []

            query_res = self._collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, max(self._collection.count(), 1)),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            if query_res and query_res.get("ids") and len(query_res["ids"][0]) > 0:
                for idx in range(len(query_res["ids"][0])):
                    doc_id = query_res["ids"][0][idx]
                    text = query_res["documents"][0][idx]
                    meta = query_res["metadatas"][0][idx]
                    dist = query_res["distances"][0][idx]
                    # Cosine distance to similarity score
                    score = round(max(1.0 - dist, 0.0), 3) if dist is not None else 1.0

                    results.append({
                        "chunk_id": doc_id,
                        "text": text,
                        "score": score,
                        "metadata": meta,
                        "document_id": meta.get("document_id"),
                        "page_number": meta.get("page_number", 1),
                        "category": meta.get("category"),
                        "filename": meta.get("filename"),
                    })
        else:
            # In-memory search with strict user_id filtering
            candidates = []
            for chunk_id, data in self._in_memory_fallback.items():
                meta = data["metadata"]
                if meta.get("user_id") != user_id:
                    continue
                if category and meta.get("category") != category:
                    continue

                sim = embedder.similarity(query_emb, data["embedding"])
                candidates.append((sim, chunk_id, data))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for sim, chunk_id, data in candidates[:top_k]:
                meta = data["metadata"]
                results.append({
                    "chunk_id": chunk_id,
                    "text": data["text"],
                    "score": round(sim, 3),
                    "metadata": meta,
                    "document_id": meta.get("document_id"),
                    "page_number": meta.get("page_number", 1),
                    "category": meta.get("category"),
                    "filename": meta.get("filename"),
                })

        return results

    def delete_document_chunks(self, document_id: int, user_id: int) -> None:
        """Delete all indexed chunks for a specific document owned by user_id."""
        if self._collection is not None:
            where_clause = {
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"user_id": {"$eq": user_id}},
                ]
            }
            try:
                self._collection.delete(where=where_clause)
            except Exception as e:
                logger.warning(f"Failed to delete chunks from ChromaDB: {e}")
        else:
            to_delete = [
                cid for cid, d in self._in_memory_fallback.items()
                if d["metadata"].get("document_id") == document_id and d["metadata"].get("user_id") == user_id
            ]
            for cid in to_delete:
                del self._in_memory_fallback[cid]

    def find_near_duplicates(
        self,
        document_text: str,
        user_id: int,
        threshold: float = 0.88,
        exclude_doc_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find other documents owned by user_id that are semantically near-duplicates."""
        hits = self.query_similar(query_text=document_text[:1500], user_id=user_id, top_k=5)
        duplicates: List[Dict[str, Any]] = []
        seen_doc_ids = set()

        for hit in hits:
            doc_id = hit["document_id"]
            if doc_id == exclude_doc_id:
                continue
            if doc_id in seen_doc_ids:
                continue
            if hit["score"] >= threshold:
                seen_doc_ids.add(doc_id)
                duplicates.append({
                    "document_id": doc_id,
                    "filename": hit["filename"],
                    "similarity": hit["score"],
                    "category": hit["category"],
                })

        return duplicates


# Global vector store instance
vector_store = ChromaVectorStore()
