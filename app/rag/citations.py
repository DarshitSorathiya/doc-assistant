# ============================================================
# Citations & Source Attribution Engine
# ============================================================
# Tracks, validates, and formats document sources with page-level
# citations for grounded RAG generation.
# ============================================================

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Represents a specific document and page citation."""
    document_id: int
    filename: str
    page_number: int = 1
    snippet: Optional[str] = None
    score: Optional[float] = None


class CitationManager:
    """Manages citation formatting, source extraction, and grounding validation."""

    @staticmethod
    def format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into numbered context blocks for LLM prompt."""
        if not chunks:
            return "No relevant documents found."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("document_id", "Unknown")
            filename = chunk.get("filename", "Unknown")
            page = chunk.get("page_number", 1)
            text = chunk.get("text", "").strip()

            block = (
                f"[Source {idx}]: File: {filename} (ID: {doc_id}), Page: {page}\n"
                f"Content: {text}"
            )
            formatted_blocks.append(block)

        return "\n\n---\n\n".join(formatted_blocks)

    @staticmethod
    def extract_sources(chunks: List[Dict[str, Any]]) -> List[SourceCitation]:
        """Deduplicate and extract structured source citations from retrieved chunks."""
        seen = set()
        citations = []

        for chunk in chunks:
            doc_id = chunk.get("document_id")
            filename = chunk.get("filename", "document")
            page = chunk.get("page_number", 1)
            text = chunk.get("text", "")
            score = chunk.get("score")

            if doc_id is None:
                continue

            key = (doc_id, page)
            if key not in seen:
                seen.add(key)
                snippet = text[:200] + "..." if len(text) > 200 else text
                citations.append(
                    SourceCitation(
                        document_id=doc_id,
                        filename=filename,
                        page_number=page,
                        snippet=snippet,
                        score=score,
                    )
                )

        return citations

    @staticmethod
    def is_insufficient_evidence(
        chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
        min_score: float = 0.05,
    ) -> bool:
        """Check if retrieved evidence is empty, low scoring, or lacks core query keywords."""
        if not chunks:
            return True

        if query:
            stopwords = {
                "what", "is", "the", "of", "my", "in", "for", "on", "at", "to", "a",
                "an", "and", "or", "tell", "me", "about", "how", "much", "does", "do",
                "did", "can", "could", "would", "will", "have", "has", "had", "find",
                "show", "give", "get", "with", "from", "by", "as", "number", "details",
                "information", "please", "any", "which", "where", "who", "when", "why",
                "should", "know", "document", "documents", "file", "files", "based", "using",
            }
            query_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query.lower()) if w not in stopwords]
            
            if query_tokens:
                # User wording often differs from labels used in a document.
                synonym_groups = (
                    {"coverage", "cover", "covered", "covers", "protection", "insured", "insurance"},
                    {"cost", "price", "premium", "fee", "amount", "charge", "payment"},
                    {"salary", "income", "pay", "compensation", "wage"},
                    {"expiry", "expiration", "expires", "validity", "valid", "due"},
                    {"job", "role", "position", "employment", "work"},
                    {"degree", "education", "qualification", "university", "college"},
                )
                expanded_query_tokens = set(query_tokens)
                for group in synonym_groups:
                    if expanded_query_tokens.intersection(group):
                        expanded_query_tokens.update(group)

                all_text = " ".join(c.get("text", "").lower() for c in chunks)
                has_keyword_match = any(
                    bool(re.search(rf"\b{re.escape(token)}\b", all_text))
                    for token in expanded_query_tokens
                )
                if not has_keyword_match:
                    return True
            else:
                has_keyword_match = False
        else:
            has_keyword_match = False

        top_score = chunks[0].get("score", 0.0) if chunks else 0.0
        # RRF scores are naturally smaller than reranker scores; trust a clear
        # concept/keyword match instead of imposing one scale-dependent cutoff.
        if top_score is not None and top_score < min_score and not has_keyword_match:
            return True

        return False


citation_manager = CitationManager()
