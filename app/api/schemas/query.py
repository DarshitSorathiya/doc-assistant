# ============================================================
# Query API Schemas (Phase 8 RAG Chain)
# ============================================================

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceCitationSchema(BaseModel):
    """Pydantic representation of a document citation."""
    document_id: int = Field(..., description="ID of the cited document")
    filename: str = Field(..., description="Filename of the cited document")
    page_number: int = Field(default=1, description="Page number where evidence was found")
    snippet: Optional[str] = Field(None, description="Relevant text snippet")
    score: Optional[float] = Field(None, description="Relevance / similarity score")


class QueryRequest(BaseModel):
    """Request model for conversational RAG query."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    session_id: Optional[str] = Field(None, description="Conversation session ID for context continuity")
    category: Optional[str] = Field(None, description="Optional document category filter")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    use_reranker: bool = Field(default=True, description="Whether to apply neural reranking")
    document_id: Optional[int] = Field(None, description="Optional document to use when resolving an ambiguous document-specific question")


class QueryResponse(BaseModel):
    """Response model for conversational RAG query."""
    query: str
    answer: str
    session_id: str
    sources: List[SourceCitationSchema]
    tools_invoked: List[str] = Field(default_factory=list, description="List of agent tools called")
    confidence: float
    model_used: str
    document_options: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Documents the user can select to disambiguate a document-specific request.",
    )


class SessionClearResponse(BaseModel):
    """Response when clearing conversation memory."""
    session_id: str
    cleared: bool
    message: str
