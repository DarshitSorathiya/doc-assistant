# ============================================================
# Knowledge Graph API Routes (Phase 11)
# ============================================================

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.jwt import get_current_user
from app.knowledge_graph.graph import kg_manager
from app.api.schemas.graph import GraphResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])


@router.get("", response_model=GraphResponse, summary="Query Knowledge Graph")
def get_knowledge_graph(
    entity_type: Optional[str] = Query(None, description="Filter by node entity type (e.g. Person, Policy, TaxIdentifier, BankAccount, Property)"),
    query: Optional[str] = Query(None, description="Search term for node labels or entity values"),
    document_id: Optional[int] = Query(None, description="Filter by specific document node and its 1-hop subgraph"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve filtered knowledge graph partitioned for the authenticated user."""
    try:
        graph_data = kg_manager.query_graph(
            db=db,
            user_id=current_user.id,
            entity_type=entity_type,
            query=query,
            document_id=document_id,
        )
        return graph_data
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying knowledge graph: {str(e)}",
        )
