# ============================================================
# Page-Aware Document Chunker
# ============================================================
# Splits document text into overlapping chunks while preserving
# page boundaries, section coherence, and source metadata.
# ============================================================

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.ocr.base import PageResult


class DocumentChunk(BaseModel):
    """A semantic chunk of a document with rich retrieval metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier: {doc_id}_p{page}_c{idx}")
    text: str = Field(..., description="Text content of the chunk")
    document_id: int = Field(..., description="Parent document database ID")
    user_id: int = Field(..., description="Owner user ID for strict security isolation")
    page_number: int = Field(..., description="1-indexed source page number")
    category: Optional[str] = Field(None, description="Document classification category")
    filename: str = Field(..., description="Original document filename")
    chunk_index: int = Field(default=0, description="0-indexed chunk sequence on the page")

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary suitable for vectorstore (ChromaDB) storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "page_number": self.page_number,
            "category": self.category or "Other",
            "filename": self.filename,
            "chunk_index": self.chunk_index,
        }


class PageAwareChunker:
    """Chunks text on a per-page basis with sliding-window overlap."""

    def __init__(
        self,
        target_chunk_size: int = 600,  # ~120-150 words
        chunk_overlap: int = 100,      # ~20-25 words
        min_chunk_size: int = 40,
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_page_text(self, text: str) -> List[str]:
        """Split a single page's text into overlapping chunks respecting sentence/paragraph boundaries."""
        clean_text = text.strip()
        if not clean_text:
            return []

        if len(clean_text) <= self.target_chunk_size:
            return [clean_text]

        # Split into paragraphs or sentences
        paragraphs = re.split(r"\n\s*\n|\n", clean_text)
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 1 <= self.target_chunk_size:
                current_chunk += "\n" + para
            else:
                chunks.append(current_chunk)
                # Apply overlap from the tail of the current chunk
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = (overlap_text + "\n" + para).strip()

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        elif current_chunk and chunks:
            # Append small trailing remnant to the last chunk
            chunks[-1] = (chunks[-1] + "\n" + current_chunk).strip()
        elif current_chunk:
            chunks.append(current_chunk)

        return chunks

    def create_document_chunks(
        self,
        document_id: int,
        user_id: int,
        filename: str,
        category: Optional[str],
        pages: List[PageResult],
    ) -> List[DocumentChunk]:
        """Convert all pages of a document into tagged DocumentChunk objects."""
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            page_text_chunks = self.chunk_page_text(page.text)
            for idx, chunk_text in enumerate(page_text_chunks):
                chunk_id = f"{document_id}_p{page.page_num}_c{idx}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    document_id=document_id,
                    user_id=user_id,
                    page_number=page.page_num,
                    category=category,
                    filename=filename,
                    chunk_index=idx,
                )
                all_chunks.append(chunk)

        return all_chunks
