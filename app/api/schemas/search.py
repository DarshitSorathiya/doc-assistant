# ============================================================
# Search Schemas (Pydantic Models)
# ============================================================

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class KeywordSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Keywords or terms to search")
    category: Optional[str] = Field(None, description="Optional category filter")
    top_k: int = Field(default=5, ge=1, le=50, description="Max number of results")


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language semantic search query")
    category: Optional[str] = Field(None, description="Optional category filter")
    top_k: int = Field(default=5, ge=1, le=50, description="Max number of results")


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    category: Optional[str] = Field(None, description="Optional category filter")
    top_k: int = Field(default=5, ge=1, le=50, description="Max number of results to return")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight between vector (1.0) and BM25 (0.0)")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")


class EntitySearchRequest(BaseModel):
    entity_type: Optional[str] = Field(None, description="Type of entity: PAN, Aadhaar, Policy Number, IFSC, etc.")
    entity_value: Optional[str] = Field(None, description="Entity value or substring")
    top_k: int = Field(default=20, ge=1, le=100, description="Max number of results")


class SearchResultItem(BaseModel):
    document_id: int
    filename: str
    chunk_id: Optional[str] = None
    page_number: int = 1
    category: Optional[str] = None
    text: str
    score: float
    method: str = "hybrid"  # vector | bm25 | hybrid | reranked


class SearchResponse(BaseModel):
    query: str
    total_results: int
    items: List[SearchResultItem]


class EntitySearchResultItem(BaseModel):
    id: int
    document_id: int
    filename: str
    category: Optional[str] = None
    entity_type: str
    value: str
    confidence: Optional[float] = None
    source_page: Optional[int] = 1


class EntitySearchResponse(BaseModel):
    entity_type: Optional[str] = None
    entity_value: Optional[str] = None
    total_results: int
    items: List[EntitySearchResultItem]
