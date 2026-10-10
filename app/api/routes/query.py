# ============================================================
# Conversational Query API Routes (Phase 8 RAG Chain)
# ============================================================

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.jwt import get_current_active_user
from app.api.schemas.query import QueryRequest, QueryResponse, SessionClearResponse, SourceCitationSchema
from app.agent.agent import document_agent
from app.rag.memory import conversation_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Conversational RAG"])


@router.post("/query", response_model=QueryResponse, summary="Execute Conversational RAG Query")
def execute_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Execute conversational agentic query grounded in user's isolated documents.

    - Resolves multi-turn conversational context & pronouns.
    - Selects and executes appropriate tools via Agent Planner.
    - Records tool actions in audit trail.
    - Generates factual answer with page-level citations.
    - Strictly isolates retrieval to documents owned by current_user.
    """
    try:
        agent_resp = document_agent.execute(
            db=db,
            user_id=current_user.id,
            query=request.query,
            session_id=request.session_id,
            category=request.category,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )

        sources_schema = [
            SourceCitationSchema(
                document_id=s.document_id,
                filename=s.filename,
                page_number=s.page_number,
                snippet=s.snippet,
                score=s.score,
            )
            for s in agent_resp.sources
        ]

        return QueryResponse(
            query=agent_resp.query,
            answer=agent_resp.answer,
            session_id=agent_resp.session_id,
            sources=sources_schema,
            tools_invoked=agent_resp.tools_invoked,
            confidence=agent_resp.confidence,
            model_used=agent_resp.model_used,
        )
    except Exception as e:
        logger.error(f"Error executing RAG query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.delete("/query/session/{session_id}", response_model=SessionClearResponse, summary="Clear Session Memory")
def clear_session_memory(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Reset conversational memory for a specific session ID."""
    cleared = conversation_memory.clear_session(user_id=current_user.id, session_id=session_id)
    return SessionClearResponse(
        session_id=session_id,
        cleared=cleared,
        message="Session memory cleared successfully." if cleared else "No active memory found for session.",
    )
