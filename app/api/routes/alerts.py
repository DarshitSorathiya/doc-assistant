# ============================================================
# Proactive Alerts API Routes (Phase 11)
# ============================================================

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.jwt import get_current_user
from app.services.alert_service import alert_service
from app.api.schemas.alerts import AlertsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alerts", tags=["Proactive Intelligence"])


@router.get("", response_model=AlertsResponse, summary="Get Proactive Alerts")
def get_alerts(
    warning_days: int = Query(60, ge=1, le=365, description="Warning window in days for upcoming expirations"),
    alert_type: Optional[str] = Query(None, description="Optional filter by alert type: expiry, missing_document, duplicate, version_modified"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve proactive document alerts for the authenticated user (expiry, missing docs, duplicates, revisions)."""
    try:
        alerts_data = alert_service.get_all_alerts(
            db=db,
            user_id=current_user.id,
            warning_days=warning_days,
        )

        if alert_type:
            filtered = [a for a in alerts_data["alerts"] if a.get("alert_type") == alert_type.lower()]
            return {
                "total_alerts": len(filtered),
                "critical_count": sum(1 for a in filtered if a.get("severity") == "critical"),
                "warning_count": sum(1 for a in filtered if a.get("severity") == "warning"),
                "info_count": sum(1 for a in filtered if a.get("severity") == "info"),
                "alerts": filtered,
            }

        return alerts_data
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating proactive alerts: {str(e)}",
        )
