# ============================================================
# Search API Routes — Keyword, Semantic, Hybrid & Entity Search
# ============================================================

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.jwt import get_current_user
from app.rag.retriever import retriever
from app.api.schemas.search import (
    KeywordSearchRequest,
    SemanticSearchRequest,
    HybridSearchRequest,
    EntitySearchRequest,
    SearchResponse,
    EntitySearchResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Search"])


# ------------------------------------------------------------------
# POST /api/v1/search — Keyword Search (BM25)
# ------------------------------------------------------------------
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Keyword search across user documents using BM25",
)
def keyword_search(
    request: KeywordSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search documents by exact keyword matches using BM25 ranking."""
    items = retriever.keyword_search(
        db=db,
        user_id=current_user.id,
        query=request.query,
        category=request.category,
        top_k=request.top_k,
    )
    return SearchResponse(
        query=request.query,
        total_results=len(items),
        items=items,
    )


# ------------------------------------------------------------------
# POST /api/v1/semantic-search — Dense Vector Search
# ------------------------------------------------------------------
@router.post(
    "/semantic-search",
    response_model=SearchResponse,
    summary="Semantic vector search across user documents",
)
def semantic_search(
    request: SemanticSearchRequest,
    current_user: User = Depends(get_current_user),
):
    """Search documents using sentence-transformers embedding similarity."""
    items = retriever.semantic_search(
        user_id=current_user.id,
        query=request.query,
        category=request.category,
        top_k=request.top_k,
    )
    return SearchResponse(
        query=request.query,
        total_results=len(items),
        items=items,
    )


# ------------------------------------------------------------------
# POST /api/v1/hybrid-search — Vector + BM25 + RRF + Reranker
# ------------------------------------------------------------------
@router.post(
    "/hybrid-search",
    response_model=SearchResponse,
    summary="Hybrid search combining Dense Vector, BM25 and Cross-Encoder Reranking",
)
def hybrid_search(
    request: HybridSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Execute state-of-the-art hybrid search with Reciprocal Rank Fusion."""
    items = retriever.hybrid_search(
        db=db,
        user_id=current_user.id,
        query=request.query,
        category=request.category,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
    )
    return SearchResponse(
        query=request.query,
        total_results=len(items),
        items=items,
    )


# ------------------------------------------------------------------
# POST /api/v1/entity-search — Search by Extracted Entity (PAN, IFSC, etc.)
# ------------------------------------------------------------------
@router.post(
    "/entity-search",
    response_model=EntitySearchResponse,
    summary="Search documents by extracted entities (PAN, Aadhaar, Policy No, etc.)",
)
def entity_search(
    request: EntitySearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search for documents containing specific structured identifiers."""
    items = retriever.entity_search(
        db=db,
        user_id=current_user.id,
        entity_type=request.entity_type,
        entity_value=request.entity_value,
        top_k=request.top_k,
    )
    return EntitySearchResponse(
        entity_type=request.entity_type,
        entity_value=request.entity_value,
        total_results=len(items),
        items=items,
    )
