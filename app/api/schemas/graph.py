# ============================================================
# Schemas for Knowledge Graph API
# ============================================================

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GraphNodeSchema(BaseModel):
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdgeSchema(BaseModel):
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    total_nodes: int
    total_edges: int
    nodes: List[GraphNodeSchema]
    edges: List[GraphEdgeSchema]
