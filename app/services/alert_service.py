# ============================================================
# Alert Service — Proactive Document Intelligence
# ============================================================
# Generates proactive alerts:
# 1. Expiry alerts (policies, IDs, passports, warranties)
# 2. Missing document heuristics
# 3. Duplicate document alerts (exact and near-duplicate)
# 4. Version modification alerts
# ============================================================

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
from sqlalchemy.orm import Session

from app.models import Document, Entity, DocumentCategory, DocumentVersion, DocumentStatus
from app.services.timeline_service import _parse_date
from app.services.duplicates_service import duplicates_service

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertType(str, Enum):
    EXPIRY = "expiry"
    MISSING_DOCUMENT = "missing_document"
    DUPLICATE = "duplicate"
    VERSION_MODIFIED = "version_modified"


class AlertService:
    """Proactively analyzes user document portfolios to generate actionable intelligence."""

    @staticmethod
    def get_expiry_alerts(
        db: Session,
        user_id: int,
        warning_days: int = 60,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Check all extracted date entities for upcoming or past expirations."""
        current_time = now or datetime.now()
        warning_delta = timedelta(days=warning_days)

        query = (
            db.query(Entity, Document)
            .join(Document, Entity.document_id == Document.id)
            .filter(Document.user_id == user_id, Document.status == DocumentStatus.READY)
        )
        results = query.all()

        alerts = []
        seen_keys = set()

        for ent, doc in results:
            ent_type_lower = ent.entity_type.lower()
            if "expiry" in ent_type_lower or "valid" in ent_type_lower or "due" in ent_type_lower:
                parsed_dt = _parse_date(ent.value)
                if not parsed_dt:
                    continue

                alert_key = (doc.id, ent.value)
                if alert_key in seen_keys:
                    continue
                seen_keys.add(alert_key)

                days_until = (parsed_dt - current_time).days

                if days_until < 0:
                    alerts.append({
                        "alert_type": AlertType.EXPIRY.value,
                        "severity": AlertSeverity.CRITICAL.value,
                        "title": f"Document Expired: {doc.filename}",
                        "description": f"Expired {abs(days_until)} days ago on {parsed_dt.strftime('%d-%b-%Y')} ({ent.entity_type}: {ent.value}).",
                        "document_id": doc.id,
                        "filename": doc.filename,
                        "category": doc.category.value if doc.category else None,
                        "entity_id": ent.id,
                        "expiry_date": parsed_dt.strftime("%Y-%m-%d"),
                        "days_remaining": days_until,
                    })
                elif days_until <= warning_days:
                    alerts.append({
                        "alert_type": AlertType.EXPIRY.value,
                        "severity": AlertSeverity.WARNING.value,
                        "title": f"Upcoming Expiry: {doc.filename}",
                        "description": f"Expires in {days_until} days on {parsed_dt.strftime('%d-%b-%Y')} ({ent.entity_type}: {ent.value}).",
                        "document_id": doc.id,
                        "filename": doc.filename,
                        "category": doc.category.value if doc.category else None,
                        "entity_id": ent.id,
                        "expiry_date": parsed_dt.strftime("%Y-%m-%d"),
                        "days_remaining": days_until,
                    })

        return alerts

    @staticmethod
    def get_missing_document_alerts(
        db: Session,
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """Evaluate heuristics to identify missing essential documents in user portfolio."""
        docs = db.query(Document).filter(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY,
        ).all()

        categories = {d.category for d in docs if d.category}
        all_text = " ".join((d.extracted_text or "").lower() for d in docs)
        alerts = []

        # Heuristic 1: Property docs exist, but no Insurance
        if DocumentCategory.PROPERTY in categories:
            has_insurance = DocumentCategory.INSURANCE in categories or "insurance" in all_text
            if not has_insurance:
                alerts.append({
                    "alert_type": AlertType.MISSING_DOCUMENT.value,
                    "severity": AlertSeverity.WARNING.value,
                    "title": "Missing Property Insurance",
                    "description": "You have registered Property deeds/khata records, but no active Home/Property Insurance policy was detected.",
                    "document_id": None,
                    "filename": None,
                    "category": DocumentCategory.PROPERTY.value,
                })

        # Heuristic 2: Banking/Tax records exist, but no Government ID / PAN
        if DocumentCategory.TAX in categories or DocumentCategory.BANKING in categories or DocumentCategory.FINANCIAL in categories:
            has_pan_or_id = DocumentCategory.GOVERNMENT_ID in categories or "pan" in all_text or "aadhaar" in all_text
            if not has_pan_or_id:
                alerts.append({
                    "alert_type": AlertType.MISSING_DOCUMENT.value,
                    "severity": AlertSeverity.INFO.value,
                    "title": "Missing Government ID (PAN / Aadhaar)",
                    "description": "You have uploaded Tax/Financial statements, but no corresponding PAN Card or Government ID has been indexed.",
                    "document_id": None,
                    "filename": None,
                    "category": DocumentCategory.TAX.value,
                })

        # Heuristic 3: Vehicle mentioned in DL/Docs, but no Motor Insurance
        if "driving licence" in all_text or "driving license" in all_text or "dl" in all_text:
            has_motor_ins = "motor insurance" in all_text or "vehicle insurance" in all_text or "car insurance" in all_text
            if not has_motor_ins:
                alerts.append({
                    "alert_type": AlertType.MISSING_DOCUMENT.value,
                    "severity": AlertSeverity.INFO.value,
                    "title": "Missing Vehicle Insurance Policy",
                    "description": "Driving License records found, but no matching Vehicle / Auto Insurance policy document is present.",
                    "document_id": None,
                    "filename": None,
                    "category": DocumentCategory.GOVERNMENT_ID.value,
                })

        return alerts

    @staticmethod
    def get_duplicate_alerts(
        db: Session,
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve exact and near duplicate document alerts."""
        dup_res = duplicates_service.find_duplicates(db, user_id)
        alerts = []

        for group in dup_res.get("exact_duplicates", []):
            fn_list = ", ".join(d["filename"] for d in group["documents"])
            alerts.append({
                "alert_type": AlertType.DUPLICATE.value,
                "severity": AlertSeverity.WARNING.value,
                "title": "Exact Duplicate Documents Detected",
                "description": f"Identical file content (SHA-256: {group['file_hash'][:10]}...) detected across {group['count']} documents: {fn_list}.",
                "document_id": group["documents"][0]["document_id"],
                "filename": group["documents"][0]["filename"],
                "category": group["documents"][0]["category"],
            })

        for near in dup_res.get("near_duplicates", []):
            d1_name = near["document_1"]["filename"]
            d2_name = near["document_2"]["filename"]
            score = round(near.get("similarity_score", 0.0) * 100, 1)
            alerts.append({
                "alert_type": AlertType.DUPLICATE.value,
                "severity": AlertSeverity.INFO.value,
                "title": "Near-Duplicate Documents Detected",
                "description": f"High content similarity ({score}%) between '{d1_name}' and '{d2_name}'.",
                "document_id": near["document_1"]["id"],
                "filename": d1_name,
                "category": near["document_1"]["category"],
            })

        return alerts

    @staticmethod
    def get_version_modification_alerts(
        db: Session,
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """Alert on documents that have multiple version revisions."""
        docs = db.query(Document).filter(
            Document.user_id == user_id,
            Document.version > 1,
        ).all()

        alerts = []
        for doc in docs:
            alerts.append({
                "alert_type": AlertType.VERSION_MODIFIED.value,
                "severity": AlertSeverity.INFO.value,
                "title": f"Document Modified (v{doc.version}): {doc.filename}",
                "description": f"Document {doc.filename} has been revised to version {doc.version}.",
                "document_id": doc.id,
                "filename": doc.filename,
                "category": doc.category.value if doc.category else None,
            })

        return alerts

    @classmethod
    def get_all_alerts(
        cls,
        db: Session,
        user_id: int,
        warning_days: int = 60,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Aggregate all proactive alerts for a user into a prioritized feed."""
        expiry = cls.get_expiry_alerts(db, user_id, warning_days=warning_days, now=now)
        missing = cls.get_missing_document_alerts(db, user_id)
        duplicates = cls.get_duplicate_alerts(db, user_id)
        versions = cls.get_version_modification_alerts(db, user_id)

        all_alerts = expiry + missing + duplicates + versions

        # Priority ordering: critical > warning > info
        priority_map = {
            AlertSeverity.CRITICAL.value: 0,
            AlertSeverity.WARNING.value: 1,
            AlertSeverity.INFO.value: 2,
        }
        all_alerts.sort(key=lambda x: priority_map.get(x.get("severity"), 3))

        return {
            "total_alerts": len(all_alerts),
            "critical_count": sum(1 for a in all_alerts if a.get("severity") == AlertSeverity.CRITICAL.value),
            "warning_count": sum(1 for a in all_alerts if a.get("severity") == AlertSeverity.WARNING.value),
            "info_count": sum(1 for a in all_alerts if a.get("severity") == AlertSeverity.INFO.value),
            "alerts": all_alerts,
        }


alert_service = AlertService()
