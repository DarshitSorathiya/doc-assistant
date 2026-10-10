# ============================================================
# Schemas for Proactive Alerts API
# ============================================================

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AlertItem(BaseModel):
    alert_type: str
    severity: str
    title: str
    description: str
    document_id: Optional[int] = None
    filename: Optional[str] = None
    category: Optional[str] = None
    entity_id: Optional[int] = None
    expiry_date: Optional[str] = None
    days_remaining: Optional[int] = None


class AlertsResponse(BaseModel):
    total_alerts: int
    critical_count: int
    warning_count: int
    info_count: int
    alerts: List[AlertItem]
