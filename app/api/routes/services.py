# ============================================================
# Document Services API Routes (Phase 10)
# ============================================================

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.jwt import get_current_user
from app.api.schemas.services import (
    SummaryRequest,
    SummaryResponse,
    CompareRequest,
    CompareResponse,
    TimelineRequest,
    TimelineResponse,
    DuplicatesRequest,
    DuplicatesResponse,
    InsightsResponse,
)
from app.services.summary_service import summary_service
from app.services.comparison_service import comparison_service
from app.services.timeline_service import timeline_service
from app.services.duplicates_service import duplicates_service
from app.services.insights_service import insights_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Document Services"])


# ------------------------------------------------------------
# POST /api/v1/summary
# ------------------------------------------------------------
@router.post("/summary", response_model=SummaryResponse, summary="Summarize Document(s)")
def summarize_documents(
    request: SummaryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate structured summary for a single document or category."""
    if request.document_id:
        result = summary_service.summarize_document(
            db=db,
            user_id=current_user.id,
            document_id=request.document_id,
        )
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["error"])
        return SummaryResponse(**result)
    else:
        result = summary_service.summarize_category(
            db=db,
            user_id=current_user.id,
            category=request.category,
        )
        return SummaryResponse(**result)


# ------------------------------------------------------------
# POST /api/v1/compare
# ------------------------------------------------------------
@router.post("/compare", response_model=CompareResponse, summary="Compare Two Documents")
def compare_documents(
    request: CompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compare two documents or versions and return structured diffs."""
    result = comparison_service.compare_documents(
        db=db,
        user_id=current_user.id,
        doc_id_1=request.doc_id_1,
        doc_id_2=request.doc_id_2,
    )
    if "error" in result:
        detail = result["error"]
        if "not found" in detail.lower() or "unauthorized" in detail.lower():
            code = status.HTTP_404_NOT_FOUND
        elif "finish processing" in detail.lower():
            code = status.HTTP_409_CONFLICT
        else:
            code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=detail)
    return CompareResponse(**result)


# ------------------------------------------------------------
# POST /api/v1/timeline
# ------------------------------------------------------------
@router.post("/timeline", response_model=TimelineResponse, summary="Extract Document Timeline")
def extract_timeline(
    request: TimelineRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Extract chronologically sorted date events across user's documents."""
    result = timeline_service.extract_timeline(
        db=db,
        user_id=current_user.id,
        category=request.category,
        sort_order=request.sort_order,
    )
    return TimelineResponse(**result)


# ------------------------------------------------------------
# POST /api/v1/duplicates
# ------------------------------------------------------------
@router.post("/duplicates", response_model=DuplicatesResponse, summary="Find Duplicate Documents")
def find_duplicates(
    request: DuplicatesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Detect exact hash duplicates and semantic near-duplicates."""
    result = duplicates_service.find_duplicates(
        db=db,
        user_id=current_user.id,
        threshold=request.threshold,
    )
    return DuplicatesResponse(**result)


# ------------------------------------------------------------
# POST /api/v1/insights
# ------------------------------------------------------------
@router.post("/insights", response_model=InsightsResponse, summary="Portfolio Insights")
def get_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compute aggregate analytics, category distribution, and risk metrics."""
    result = insights_service.get_portfolio_insights(
        db=db,
        user_id=current_user.id,
    )
    return InsightsResponse(**result)
