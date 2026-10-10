# ============================================================
# Knowledge Graph Entities & Node Definitions
# ============================================================

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Supported entity node types in the document knowledge graph."""
    DOCUMENT = "Document"
    PERSON = "Person"
    POLICY = "Policy"
    BANK_ACCOUNT = "BankAccount"
    PROPERTY = "Property"
    TAX_ID = "TaxIdentifier"
    ORGANIZATION = "Organization"
    DATE_EVENT = "DateEvent"


class GraphNode(BaseModel):
    """Represents a vertex in the Knowledge Graph."""
    id: str = Field(..., description="Unique node ID (e.g., 'doc_1', 'person_John Doe', 'policy_POL-123')")
    label: str = Field(..., description="Human-readable node label")
    node_type: NodeType = Field(..., description="Entity category/type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Metadata key-values")
