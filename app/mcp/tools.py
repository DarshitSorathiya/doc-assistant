# ============================================================
# MCP Tools — 11 Document Intelligence Tools over MCP Protocol
# ============================================================
# Each tool wraps the Phase 9 ToolRegistry with MCP-compatible
# calling conventions. User isolation is enforced via the
# user_id passed in the MCP session context.
# ============================================================

import logging
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.agent.tools import tool_registry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# MCP Tool Handler interface
# Each function takes user_id + db + **kwargs and returns a dict.
# ------------------------------------------------------------------

def mcp_search_documents(user_id: int, db: Session, query: str, category: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    """Keyword BM25 search over user's documents."""
    return tool_registry.execute_tool("search_documents", user_id=user_id, db=db, kwargs={"query": query, "category": category, "top_k": top_k})


def mcp_semantic_search(user_id: int, db: Session, query: str, category: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    """Dense vector semantic search."""
    return tool_registry.execute_tool("semantic_search", user_id=user_id, db=db, kwargs={"query": query, "category": category, "top_k": top_k})


def mcp_hybrid_search(user_id: int, db: Session, query: str, category: Optional[str] = None, top_k: int = 5, use_reranker: bool = True) -> Dict[str, Any]:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion."""
    return tool_registry.execute_tool("hybrid_search", user_id=user_id, db=db, kwargs={"query": query, "category": category, "top_k": top_k, "use_reranker": use_reranker})


def mcp_get_document(user_id: int, db: Session, document_id: int) -> Dict[str, Any]:
    """Fetch full document info and text preview."""
    return tool_registry.execute_tool("get_document", user_id=user_id, db=db, kwargs={"document_id": document_id})


def mcp_get_document_metadata(user_id: int, db: Session, document_id: int) -> Dict[str, Any]:
    """Get extracted entities and metadata for a document."""
    return tool_registry.execute_tool("get_document_metadata", user_id=user_id, db=db, kwargs={"document_id": document_id})


def mcp_search_by_entity(user_id: int, db: Session, entity_type: Optional[str] = None, value: Optional[str] = None) -> Dict[str, Any]:
    """Search documents by entity type or value (PAN, Aadhaar, IFSC, etc.)."""
    return tool_registry.execute_tool("search_by_entity", user_id=user_id, db=db, kwargs={"entity_type": entity_type, "value": value})


def mcp_summarize_document(user_id: int, db: Session, document_id: int) -> Dict[str, Any]:
    """Summarize a single document."""
    return tool_registry.execute_tool("summarize_document", user_id=user_id, db=db, kwargs={"document_id": document_id})


def mcp_compare_documents(user_id: int, db: Session, doc_id_1: int, doc_id_2: int) -> Dict[str, Any]:
    """Compare two documents and return structured diff."""
    return tool_registry.execute_tool("compare_documents", user_id=user_id, db=db, kwargs={"doc_id_1": doc_id_1, "doc_id_2": doc_id_2})


def mcp_extract_timeline(user_id: int, db: Session, category: Optional[str] = None) -> Dict[str, Any]:
    """Extract chronological date events across user documents."""
    return tool_registry.execute_tool("extract_timeline", user_id=user_id, db=db, kwargs={"category": category})


def mcp_find_duplicates(user_id: int, db: Session) -> Dict[str, Any]:
    """Detect exact and near-duplicate documents."""
    return tool_registry.execute_tool("find_duplicates", user_id=user_id, db=db, kwargs={})


def mcp_get_expiring_documents(user_id: int, db: Session, days: int = 30) -> Dict[str, Any]:
    """Retrieve documents with upcoming expiration dates."""
    return tool_registry.execute_tool("get_expiring_documents", user_id=user_id, db=db, kwargs={"days": days})


# ------------------------------------------------------------------
# Tool registry — maps MCP tool name → callable
# ------------------------------------------------------------------
MCP_TOOL_MAP: Dict[str, Any] = {
    "search_documents": mcp_search_documents,
    "semantic_search": mcp_semantic_search,
    "hybrid_search": mcp_hybrid_search,
    "get_document": mcp_get_document,
    "get_document_metadata": mcp_get_document_metadata,
    "search_by_entity": mcp_search_by_entity,
    "summarize_document": mcp_summarize_document,
    "compare_documents": mcp_compare_documents,
    "extract_timeline": mcp_extract_timeline,
    "find_duplicates": mcp_find_duplicates,
    "get_expiring_documents": mcp_get_expiring_documents,
}


def call_mcp_tool(tool_name: str, user_id: int, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch an MCP tool call by name, enforcing user isolation."""
    handler = MCP_TOOL_MAP.get(tool_name)
    if not handler:
        return {"error": f"MCP tool '{tool_name}' not found. Available: {list(MCP_TOOL_MAP.keys())}"}
    try:
        return handler(user_id=user_id, db=db, **params)
    except Exception as e:
        logger.error(f"MCP tool '{tool_name}' error: {e}", exc_info=True)
        return {"error": f"Tool execution failed: {str(e)}"}
