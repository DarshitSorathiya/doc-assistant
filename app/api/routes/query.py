# ============================================================
# Conversational Query API Routes (RAG + Persistent Chat History)
# ============================================================

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, ChatSession, ChatMessageRecord, Document
from app.auth.jwt import get_current_active_user
from app.api.schemas.query import QueryRequest, QueryResponse, SessionClearResponse, SourceCitationSchema
from app.agent.agent import document_agent
from app.rag.memory import ChatMessage, conversation_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Conversational RAG"])


def _to_memory_message(record: ChatMessageRecord) -> ChatMessage:
    """Convert a persisted message row into an in-memory message."""
    return ChatMessage(
        role=record.role,
        content=record.content,
        timestamp=record.created_at.timestamp() if record.created_at else 0,
        sources=record.sources,
        document_options=record.document_options,
    )


@router.post("/query", response_model=QueryResponse, summary="Execute Conversational RAG Query")
def execute_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Answer a query, keeping session history private to its owner and persisted."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        chat = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if chat and chat.user_id != current_user.id:
            # Never read or overwrite another user's chat if its identifier is guessed.
            chat = None
            session_id = str(uuid.uuid4())

        if chat and not conversation_memory.get_history(current_user.id, session_id):
            conversation_memory.set_history(
                current_user.id,
                session_id,
                [_to_memory_message(record) for record in chat.messages],
            )

        response = document_agent.execute(
            db=db,
            user_id=current_user.id,
            query=request.query,
            session_id=session_id,
            category=request.category,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            document_id=request.document_id,
        )

        if chat is None:
            chat = ChatSession(
                id=response.session_id,
                user_id=current_user.id,
                title=request.query.strip()[:200] or "New chat",
            )
            db.add(chat)
            db.flush()
        elif not chat.messages:
            chat.title = request.query.strip()[:200] or "New chat"

        user_message = request.query
        if request.document_id is not None:
            selected_doc = db.query(Document).filter(
                Document.id == request.document_id,
                Document.user_id == current_user.id,
            ).first()
            if selected_doc:
                user_message = f"Selected document: {selected_doc.filename}"

        db.add(ChatMessageRecord(session_id=response.session_id, role="user", content=user_message))
        db.add(ChatMessageRecord(
            session_id=response.session_id,
            role="assistant",
            content=response.answer,
            sources=[source.model_dump() for source in response.sources],
            document_options=response.document_options,
        ))
        chat.updated_at = datetime.now(timezone.utc)
        db.commit()

        return QueryResponse(
            query=response.query,
            answer=response.answer,
            session_id=response.session_id,
            sources=[
                SourceCitationSchema(
                    document_id=source.document_id,
                    filename=source.filename,
                    page_number=source.page_number,
                    snippet=source.snippet,
                    score=source.score,
                )
                for source in response.sources
            ],
            tools_invoked=response.tools_invoked,
            confidence=response.confidence,
            model_used=response.model_used,
            document_options=response.document_options,
        )
    except Exception as exc:
        db.rollback()
        logger.error("Error executing RAG query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(exc)}",
        )


@router.get("/query/sessions", summary="List the authenticated user's saved chats")
def list_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return {
        "sessions": [
            {
                "session_id": chat.id,
                "title": chat.title,
                "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
                "message_count": len(chat.messages),
            }
            for chat in sessions
        ]
    }


@router.get("/query/session/{session_id}", summary="Load a saved chat")
def get_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    chat = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")

    messages = [_to_memory_message(record) for record in chat.messages]
    conversation_memory.set_history(current_user.id, session_id, messages)
    return {
        "session_id": session_id,
        "title": chat.title,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "sources": message.sources or [],
                "document_options": message.document_options or [],
            }
            for message in messages
        ],
    }


@router.delete("/query/session/{session_id}", response_model=SessionClearResponse, summary="Delete a saved chat")
def clear_session_memory(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a persisted chat and clear its in-memory context."""
    chat = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    deleted = chat is not None
    if chat:
        db.delete(chat)
        db.commit()
    cleared = conversation_memory.clear_session(user_id=current_user.id, session_id=session_id)
    return SessionClearResponse(
        session_id=session_id,
        cleared=cleared or deleted,
        message="Chat deleted successfully." if deleted else ("Session memory cleared successfully." if cleared else "No active memory found for session."),
    )
