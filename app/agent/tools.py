# ============================================================
# Agent Tools Registry (11 Core Capabilities)
# ============================================================
# Strict schemas & implementations for agentic tool execution.
# Every tool enforces multi-tenant user_id isolation.
# ============================================================

import re
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from sqlalchemy.orm import Session

from app.models import Document, DocumentStatus, Entity, DocumentCategory
from app.rag.retriever import retriever
from app.vectorstore.chroma import vector_store

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Represents a callable agent tool with metadata and JSON schema."""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_schema(self) -> Dict[str, Any]:
        """Convert to standard function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry of all 11 core document intelligence tools."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_all_tools()

    def register(self, tool: ToolDefinition):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self.tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, user_id: int, db: Session, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with user isolation and error handling."""
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found."}
        try:
            return tool.func(user_id=user_id, db=db, **kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
            return {"error": f"Execution of '{name}' failed: {str(e)}"}

    # ----------------------------------------------------------
    # Tool Implementations
    # ----------------------------------------------------------
    @staticmethod
    def tool_search_documents(user_id: int, db: Session, query: str, category: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Execute BM25 keyword search over user's documents."""
        results = retriever.keyword_search(db=db, user_id=user_id, query=query, category=category, top_k=top_k)
        return {
            "query": query,
            "count": len(results),
            "results": [r.model_dump() for r in results],
        }

    @staticmethod
    def tool_semantic_search(user_id: int, db: Session, query: str, category: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Execute dense vector semantic search over user's documents."""
        results = retriever.semantic_search(user_id=user_id, query=query, category=category, top_k=top_k)
        return {
            "query": query,
            "count": len(results),
            "results": [r.model_dump() for r in results],
        }

    @staticmethod
    def tool_hybrid_search(user_id: int, db: Session, query: str, category: Optional[str] = None, top_k: int = 5, use_reranker: bool = True) -> Dict[str, Any]:
        """Execute hybrid search combining BM25 and vector search with RRF."""
        results = retriever.hybrid_search(db=db, user_id=user_id, query=query, category=category, top_k=top_k, use_reranker=use_reranker)
        return {
            "query": query,
            "count": len(results),
            "results": [r.model_dump() for r in results],
        }

    @staticmethod
    def tool_get_document(user_id: int, db: Session, document_id: int) -> Dict[str, Any]:
        """Get full details and text preview of a specific document."""
        doc = db.query(Document).filter(Document.id == document_id, Document.user_id == user_id).first()
        if not doc:
            return {"error": f"Document ID {document_id} not found or unauthorized."}
        return {
            "id": doc.id,
            "filename": doc.filename,
            "category": doc.category.value if doc.category else None,
            "status": doc.status.value,
            "page_count": doc.page_count,
            "file_size": doc.file_size,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "extracted_text": doc.extracted_text[:2000] if doc.extracted_text else "",
        }

    @staticmethod
    def tool_get_document_metadata(user_id: int, db: Session, document_id: int) -> Dict[str, Any]:
        """Get extracted structured metadata, confidence scores, and entities."""
        doc = db.query(Document).filter(Document.id == document_id, Document.user_id == user_id).first()
        if not doc:
            return {"error": f"Document ID {document_id} not found or unauthorized."}
        entities = db.query(Entity).filter(Entity.document_id == document_id).all()
        return {
            "id": doc.id,
            "filename": doc.filename,
            "category": doc.category.value if doc.category else None,
            "classification_confidence": doc.classification_confidence,
            "classification_method": doc.classification_method,
            "page_count": doc.page_count,
            "entities": [
                {
                    "type": e.entity_type,
                    "value": e.value,
                    "confidence": e.confidence,
                    "page": e.source_page,
                }
                for e in entities
            ],
        }

    @staticmethod
    def tool_search_by_entity(user_id: int, db: Session, entity_type: Optional[str] = None, value: Optional[str] = None) -> Dict[str, Any]:
        """Search documents containing specific entity types or values (PAN, Aadhaar, IFSC, etc.)."""
        results = retriever.entity_search(db=db, user_id=user_id, entity_type=entity_type, entity_value=value)
        return {
            "entity_type": entity_type,
            "value": value,
            "count": len(results),
            "matches": [r.model_dump() for r in results],
        }

    @staticmethod
    def tool_summarize_document(user_id: int, db: Session, document_id: int) -> Dict[str, Any]:
        """Summarize a document with structured highlights and entities."""
        doc = db.query(Document).filter(Document.id == document_id, Document.user_id == user_id).first()
        if not doc:
            return {"error": f"Document ID {document_id} not found or unauthorized."}
        entities = db.query(Entity).filter(Entity.document_id == document_id).all()
        entity_summary = {e.entity_type: e.value for e in entities}

        text = doc.extracted_text or ""
        lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("---")]
        key_points = lines[:5]

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "category": doc.category.value if doc.category else "Uncategorized",
            "summary": f"Summary of {doc.filename} ({doc.category.value if doc.category else 'Document'}):\n" + "\n".join(f"- {p}" for p in key_points),
            "key_entities": entity_summary,
            "page_count": doc.page_count,
        }

    @staticmethod
    def tool_compare_documents(user_id: int, db: Session, doc_id_1: int, doc_id_2: int) -> Dict[str, Any]:
        """Compare two documents or versions and produce structured diff."""
        doc1 = db.query(Document).filter(Document.id == doc_id_1, Document.user_id == user_id).first()
        doc2 = db.query(Document).filter(Document.id == doc_id_2, Document.user_id == user_id).first()

        if not doc1 or not doc2:
            return {"error": "One or both document IDs not found or unauthorized."}

        ent1 = {e.entity_type: e.value for e in db.query(Entity).filter(Entity.document_id == doc_id_1).all()}
        ent2 = {e.entity_type: e.value for e in db.query(Entity).filter(Entity.document_id == doc_id_2).all()}

        all_types = set(ent1.keys()).union(set(ent2.keys()))
        differences = []
        common = []

        for et in all_types:
            v1 = ent1.get(et)
            v2 = ent2.get(et)
            if v1 == v2:
                common.append({"entity": et, "value": v1})
            else:
                differences.append({
                    "entity": et,
                    f"{doc1.filename} (ID {doc1.id})": v1 or "N/A",
                    f"{doc2.filename} (ID {doc2.id})": v2 or "N/A",
                })

        return {
            "doc1": {"id": doc1.id, "filename": doc1.filename, "category": doc1.category.value if doc1.category else None},
            "doc2": {"id": doc2.id, "filename": doc2.filename, "category": doc2.category.value if doc2.category else None},
            "differences": differences,
            "common_entities": common,
        }

    @staticmethod
    def tool_extract_timeline(user_id: int, db: Session, category: Optional[str] = None) -> Dict[str, Any]:
        """Extract chronologically sorted date events across all user documents."""
        query = (
            db.query(Entity, Document)
            .join(Document, Entity.document_id == Document.id)
            .filter(Document.user_id == user_id)
        )
        if category:
            for c in DocumentCategory:
                if c.value.lower() == category.lower():
                    query = query.filter(Document.category == c)
                    break

        # Filter entity types containing date/expiry
        results = query.all()
        events = []

        for ent, doc in results:
            if "date" in ent.entity_type.lower() or "expiry" in ent.entity_type.lower():
                events.append({
                    "date_str": ent.value,
                    "event_type": ent.entity_type,
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "page": ent.source_page or 1,
                })

        return {
            "total_events": len(events),
            "timeline": events,
        }

    @staticmethod
    def tool_find_duplicates(user_id: int, db: Session) -> Dict[str, Any]:
        """Detect exact hash duplicates and near-duplicate vector matches."""
        from app.services.duplicates_service import duplicates_service
        return duplicates_service.find_duplicates(db=db, user_id=user_id)

    @staticmethod
    def tool_get_expiring_documents(user_id: int, db: Session, days: int = 30) -> Dict[str, Any]:
        """Find documents with expiry dates, sorted by upcoming expiration."""
        entities = (
            db.query(Entity, Document)
            .join(Document, Entity.document_id == Document.id)
            .filter(
                Document.user_id == user_id,
                Entity.entity_type.ilike("%expiry%"),
            )
            .all()
        )

        expiring_list = []
        for ent, doc in entities:
            expiring_list.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "category": doc.category.value if doc.category else None,
                "expiry_date": ent.value,
                "confidence": ent.confidence,
            })

        return {
            "warning_window_days": days,
            "expiring_count": len(expiring_list),
            "expiring_documents": expiring_list,
        }

    def _register_all_tools(self):
        """Register the 11 core tool schemas."""
        self.register(ToolDefinition(
            name="search_documents",
            description="Keyword search using BM25 for exact terms, names, and account numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms or keywords"},
                    "category": {"type": "string", "description": "Optional category filter"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            func=self.tool_search_documents,
        ))

        self.register(ToolDefinition(
            name="semantic_search",
            description="Dense vector semantic search to match conceptual intent and synonyms.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "category": {"type": "string", "description": "Optional category filter"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            func=self.tool_semantic_search,
        ))

        self.register(ToolDefinition(
            name="hybrid_search",
            description="Hybrid search combining dense vector + BM25 with Reciprocal Rank Fusion.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Optional category filter"},
                    "top_k": {"type": "integer", "default": 5},
                    "use_reranker": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
            func=self.tool_hybrid_search,
        ))

        self.register(ToolDefinition(
            name="get_document",
            description="Fetch full document information and text preview by document ID.",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "ID of document"},
                },
                "required": ["document_id"],
            },
            func=self.tool_get_document,
        ))

        self.register(ToolDefinition(
            name="get_document_metadata",
            description="Get structured metadata and all extracted entities for a document.",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "ID of document"},
                },
                "required": ["document_id"],
            },
            func=self.tool_get_document_metadata,
        ))

        self.register(ToolDefinition(
            name="search_by_entity",
            description="Search documents containing a specific entity type or value (PAN, Aadhaar, IFSC, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "description": "e.g. PAN, Aadhaar, Policy Number, IFSC"},
                    "value": {"type": "string", "description": "Entity value substring"},
                },
            },
            func=self.tool_search_by_entity,
        ))

        self.register(ToolDefinition(
            name="summarize_document",
            description="Summarize a document with structured highlights and key entities.",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "ID of document to summarize"},
                },
                "required": ["document_id"],
            },
            func=self.tool_summarize_document,
        ))

        self.register(ToolDefinition(
            name="compare_documents",
            description="Compare two documents and highlight differing entities and values.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_id_1": {"type": "integer", "description": "First document ID"},
                    "doc_id_2": {"type": "integer", "description": "Second document ID"},
                },
                "required": ["doc_id_1", "doc_id_2"],
            },
            func=self.tool_compare_documents,
        ))

        self.register(ToolDefinition(
            name="extract_timeline",
            description="Extract chronologically sorted date events across all user documents.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Optional category filter"},
                },
            },
            func=self.tool_extract_timeline,
        ))

        self.register(ToolDefinition(
            name="find_duplicates",
            description="Find duplicate and near-duplicate documents in user's repository.",
            parameters={"type": "object", "properties": {}},
            func=self.tool_find_duplicates,
        ))

        self.register(ToolDefinition(
            name="get_expiring_documents",
            description="Get documents expiring within N days or sorted by expiry date.",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 30, "description": "Warning window in days"},
                },
            },
            func=self.tool_get_expiring_documents,
        ))


tool_registry = ToolRegistry()
