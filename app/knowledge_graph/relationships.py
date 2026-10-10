# ============================================================
# Knowledge Graph Relationships & Edge Definitions
# ============================================================

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Semantic relationship edge types."""
    OWNS = "owns"
    HAS_POLICY = "has_policy"
    HAS_ACCOUNT = "has_account"
    ASSOCIATED_WITH = "associated_with"
    CONTAINS = "contains"
    ISSUED_BY = "issued_by"
    EXPIRES_ON = "expires_on"
    LOCATED_AT = "located_at"


class GraphEdge(BaseModel):
    """Represents a directed relationship between two nodes."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation: RelationType = Field(..., description="Relationship predicate")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge metadata")
