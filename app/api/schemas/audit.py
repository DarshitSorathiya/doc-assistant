# ============================================================
# Schemas for Audit Log API (Phase 12)
# ============================================================

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AuditLogItem(BaseModel):
    """Single audit-log entry."""
    id: int
    user_id: Optional[int] = None
    user_email: Optional[str] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    timestamp: datetime


class AuditLogResponse(BaseModel):
    """Paginated response for audit-log queries."""
    total: int = Field(description="Total matching audit entries")
    page: int = Field(description="Current page (1-indexed)")
    page_size: int = Field(description="Entries per page")
    entries: List[AuditLogItem] = Field(description="Audit log entries")
