# ============================================================
# Audit Log API Routes (Phase 12)
# ============================================================
# Admin-only endpoint for querying the immutable audit trail.
# Every API call and tool invocation is recorded here.
# ============================================================

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import User, UserRole, AuditLog
from app.auth.jwt import get_current_user, require_role
from app.api.schemas.audit import AuditLogItem, AuditLogResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])


@router.get(
    "",
    response_model=AuditLogResponse,
    summary="Query Audit Trail (Admin Only)",
)
def get_audit_logs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Entries per page"),
    action: Optional[str] = Query(None, description="Filter by action name (e.g. 'upload', 'query', 'mcp:search_documents')"),
    user_id: Optional[int] = Query(None, description="Filter by specific user ID"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    """Retrieve paginated audit-log entries.

    **Admin only** — Viewers and regular Users receive 403.

    Supports filtering by action type, user, and resource type.
    Results are ordered by timestamp descending (newest first).
    """
    query = db.query(AuditLog)

    # Apply optional filters
    if action:
        query = query.filter(AuditLog.action == action)
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    # Count total matching entries
    total = query.count()

    # Paginate
    offset = (page - 1) * page_size
    entries = (
        query.order_by(desc(AuditLog.timestamp))
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # Build response items — join user email for readability
    items = []
    for entry in entries:
        user_email = None
        if entry.user_id:
            audit_user = db.query(User).filter(User.id == entry.user_id).first()
            user_email = audit_user.email if audit_user else None

        items.append(
            AuditLogItem(
                id=entry.id,
                user_id=entry.user_id,
                user_email=user_email,
                action=entry.action,
                resource_type=entry.resource_type,
                resource_id=entry.resource_id,
                details=entry.details,
                ip_address=entry.ip_address,
                timestamp=entry.timestamp,
            )
        )

    return AuditLogResponse(
        total=total,
        page=page,
        page_size=page_size,
        entries=items,
    )
