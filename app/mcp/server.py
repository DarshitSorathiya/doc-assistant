# ============================================================
# MCP Server — Personal Document Finder
# ============================================================
# Exposes the same 11 document-intelligence tools available via
# the REST API over the Model Context Protocol (MCP).
#
# Uses the FastMCP high-level API for tool/resource registration.
#
# Entry point:  python -m app.mcp.server
# ============================================================

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from app.config.settings import get_settings
from app.database import SessionLocal
from app.models import AuditLog, User
from app.mcp.tools import call_mcp_tool
from app.mcp.resources import get_resource_handlers

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Create the FastMCP server
# ------------------------------------------------------------------
mcp = FastMCP(
    "personal-document-finder",
    instructions="Personal Document Finder — AI-powered document intelligence. "
    "Upload, OCR, classify, search, and query personal documents. "
    "Pass authentication via _meta.jwt_token or _meta.user_id parameter.",
)


# ------------------------------------------------------------------
# Helper: authenticate user from metadata
# ------------------------------------------------------------------
def _get_user_from_meta(db, meta: Optional[Dict[str, Any]] = None) -> Optional[User]:
    """Extract and validate user from tool call metadata."""
    if not meta:
        return None

    # Option 1: JWT token
    token = meta.get("jwt_token")
    if token:
        from jose import JWTError, jwt as jose_jwt
        settings = get_settings()
        try:
            payload = jose_jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
            email = payload.get("sub")
            if email:
                user = db.query(User).filter(User.email == email).first()
                if user and user.is_active:
                    return user
        except JWTError:
            pass
        return None

    # Option 2: direct user_id (trusted callers only)
    user_id = meta.get("user_id")
    if user_id:
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user and user.is_active:
            return user
    return None


def _record_audit(db, user_id: int, action: str, details: Optional[dict] = None):
    """Write an audit-log row for every MCP tool call."""
    try:
        entry = AuditLog(
            user_id=user_id,
            action=f"mcp:{action}",
            resource_type="mcp_tool",
            details=details or {},
            timestamp=datetime.now(timezone.utc),
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to write MCP audit log: {e}")
        db.rollback()


def _run_tool(tool_name: str, params: dict) -> str:
    """Common handler for all MCP tools: auth → execute → audit → return JSON."""
    db = SessionLocal()
    try:
        meta = params.pop("_meta", None) or params.pop("_metadata", None) or {}
        user = _get_user_from_meta(db, meta)
        if user is None:
            return json.dumps({"error": "Authentication required. Pass jwt_token or user_id in _meta."})

        result = call_mcp_tool(tool_name, user_id=user.id, db=db, params=params)
        _record_audit(db, user.id, tool_name, {"params": {k: str(v)[:200] for k, v in params.items()}})
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"MCP tool '{tool_name}' failed: {e}", exc_info=True)
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
    finally:
        db.close()


# ------------------------------------------------------------------
# Tool registrations — 11 document intelligence tools
# ------------------------------------------------------------------

@mcp.tool()
def search_documents(query: str, category: str = None, top_k: int = 5, _meta: dict = None) -> str:
    """Keyword BM25 search over the user's uploaded documents."""
    params = {"query": query, "category": category, "top_k": top_k}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("search_documents", params)


@mcp.tool()
def semantic_search(query: str, category: str = None, top_k: int = 5, _meta: dict = None) -> str:
    """Dense vector semantic search across the user's documents."""
    params = {"query": query, "category": category, "top_k": top_k}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("semantic_search", params)


@mcp.tool()
def hybrid_search(query: str, category: str = None, top_k: int = 5, use_reranker: bool = True, _meta: dict = None) -> str:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion and optional reranking."""
    params = {"query": query, "category": category, "top_k": top_k, "use_reranker": use_reranker}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("hybrid_search", params)


@mcp.tool()
def get_document(document_id: int, _meta: dict = None) -> str:
    """Fetch full document info including text preview for a given document ID."""
    params = {"document_id": document_id}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("get_document", params)


@mcp.tool()
def get_document_metadata(document_id: int, _meta: dict = None) -> str:
    """Get extracted entities and metadata for a specific document."""
    params = {"document_id": document_id}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("get_document_metadata", params)


@mcp.tool()
def search_by_entity(entity_type: str = None, value: str = None, _meta: dict = None) -> str:
    """Search documents by entity type (PAN, Aadhaar, IFSC, etc.) or value."""
    params = {"entity_type": entity_type, "value": value}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("search_by_entity", params)


@mcp.tool()
def summarize_document(document_id: int, _meta: dict = None) -> str:
    """Generate a structured summary of a single document."""
    params = {"document_id": document_id}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("summarize_document", params)


@mcp.tool()
def compare_documents(doc_id_1: int, doc_id_2: int, _meta: dict = None) -> str:
    """Compare two documents and return structured diffs (added/removed/changed info)."""
    params = {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("compare_documents", params)


@mcp.tool()
def extract_timeline(category: str = None, _meta: dict = None) -> str:
    """Extract chronological date events across the user's documents."""
    params = {"category": category}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("extract_timeline", params)


@mcp.tool()
def find_duplicates(_meta: dict = None) -> str:
    """Detect exact hash duplicates and semantic near-duplicates in the user's document collection."""
    params = {}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("find_duplicates", params)


@mcp.tool()
def get_expiring_documents(days: int = 30, _meta: dict = None) -> str:
    """Retrieve documents with upcoming expiration dates within a configurable window."""
    params = {"days": days}
    if _meta:
        params["_meta"] = _meta
    return _run_tool("get_expiring_documents", params)


# ------------------------------------------------------------------
# Resource registrations
# ------------------------------------------------------------------

@mcp.resource("documents://list")
def list_user_documents() -> str:
    """List all documents belonging to the authenticated user."""
    db = SessionLocal()
    try:
        handlers = get_resource_handlers()
        result = handlers.handle("documents://list", db)
        return json.dumps(result, default=str)
    finally:
        db.close()


# ------------------------------------------------------------------
# Convenience: create_mcp_server for programmatic use
# ------------------------------------------------------------------
def create_mcp_server() -> FastMCP:
    """Return the configured MCP server instance."""
    return mcp


# ------------------------------------------------------------------
# Entry point:  python -m app.mcp.server
# ------------------------------------------------------------------
async def main():
    """Run the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger.info("Starting Personal Document Finder MCP server …")
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
