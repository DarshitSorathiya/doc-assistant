# ============================================================
# Document Intelligence Agent Orchestrator
# ============================================================
# Coordinates planner, tool execution, audit logging, and
# grounded answer synthesis with page citations.
# ============================================================

import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import AuditLog
from app.rag.chain import rag_chain, RAGResponse
from app.rag.citations import citation_manager, SourceCitation
from app.rag.memory import conversation_memory
from app.agent.tools import tool_registry
from app.agent.planner import agent_planner, AgentPlanStep

logger = logging.getLogger(__name__)


class AgentResponse:
    """Standardized response from the Agent."""
    def __init__(
        self,
        query: str,
        answer: str,
        session_id: str,
        tools_invoked: List[str],
        sources: List[SourceCitation],
        confidence: float = 1.0,
        model_used: str = "agentic-react-engine",
    ):
        self.query = query
        self.answer = answer
        self.session_id = session_id
        self.tools_invoked = tools_invoked
        self.sources = sources
        self.confidence = confidence
        self.model_used = model_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "session_id": self.session_id,
            "tools_invoked": self.tools_invoked,
            "sources": [s.model_dump() for s in self.sources],
            "confidence": self.confidence,
            "model_used": self.model_used,
        }


class DocumentAgent:
    """Agentic document intelligence executor."""

    def __init__(self):
        self.registry = tool_registry
        self.planner = agent_planner

    def execute(
        self,
        db: Session,
        user_id: int,
        query: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> AgentResponse:
        """Run the agentic reasoning and tool execution loop."""
        if not session_id:
            session_id = str(uuid.uuid4())

        # 1. Contextualize query with conversational memory
        effective_query = rag_chain.contextualize_query(user_id, session_id, query)

        # 2. Plan tool steps
        plan_steps = self.planner.plan_rule_based(effective_query, user_id, db)
        tools_invoked = []
        tool_outputs = []
        collected_sources: List[SourceCitation] = []

        # 3. Execute tools
        for step in plan_steps:
            tools_invoked.append(step.tool_name)
            result = self.registry.execute_tool(
                name=step.tool_name,
                user_id=user_id,
                db=db,
                kwargs=step.arguments,
            )
            tool_outputs.append({"tool": step.tool_name, "output": result})

            # Record in DB audit log
            try:
                audit_entry = AuditLog(
                    user_id=user_id,
                    action=f"agent_tool_call:{step.tool_name}",
                    resource_type="agent_tool",
                    resource_id=f"query:{query[:90]}",
                    details={"arguments": step.arguments, "summary": str(result)[:300]},
                )
                db.add(audit_entry)
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"Failed to record audit log: {e}")

            # Collect source citations if search or document tool
            if "results" in result and isinstance(result["results"], list):
                for r in result["results"]:
                    doc_id = r.get("document_id")
                    fn = r.get("filename", "document")
                    pg = r.get("page_number", 1)
                    if doc_id:
                        collected_sources.append(
                            SourceCitation(
                                document_id=doc_id,
                                filename=fn,
                                page_number=pg,
                                snippet=r.get("text", "")[:200],
                                score=r.get("score"),
                            )
                        )
            elif "matches" in result and isinstance(result["matches"], list):
                for m in result["matches"]:
                    doc_id = m.get("document_id")
                    fn = m.get("filename", "document")
                    pg = m.get("source_page", 1)
                    if doc_id:
                        collected_sources.append(
                            SourceCitation(
                                document_id=doc_id,
                                filename=fn,
                                page_number=pg,
                                snippet=f"{m.get('entity_type')}: {m.get('value')}",
                                score=1.0,
                            )
                        )

        # Check for search tool grounding
        search_tools = {"hybrid_search", "search_documents", "semantic_search"}
        only_search = all(t in search_tools for t in tools_invoked)
        
        search_chunks = []
        for item in tool_outputs:
            if item["tool"] in search_tools:
                search_chunks.extend(item["output"].get("results", []))

        if only_search and citation_manager.is_insufficient_evidence(search_chunks, query=effective_query):
            refusal_answer = "I cannot find sufficient information in your documents to answer this question."
            conversation_memory.add_user_message(user_id, session_id, query)
            conversation_memory.add_assistant_message(user_id, session_id, refusal_answer, sources=[])
            return AgentResponse(
                query=query,
                answer=refusal_answer,
                session_id=session_id,
                tools_invoked=tools_invoked,
                sources=[],
                confidence=0.0,
                model_used="agentic-guardrail",
            )

        # 4. Synthesize structured answer
        answer = self._synthesize_answer(effective_query, tool_outputs)

        # Deduplicate sources
        unique_sources = []
        seen = set()
        for s in collected_sources:
            k = (s.document_id, s.page_number)
            if k not in seen:
                seen.add(k)
                unique_sources.append(s)

        # 5. Persist dialogue in memory
        conversation_memory.add_user_message(user_id, session_id, query)
        conversation_memory.add_assistant_message(
            user_id, session_id, answer, sources=[s.model_dump() for s in unique_sources]
        )

        return AgentResponse(
            query=query,
            answer=answer,
            session_id=session_id,
            tools_invoked=tools_invoked,
            sources=unique_sources,
            confidence=1.0 if unique_sources or "summary" in answer.lower() or "comparison" in answer.lower() else 0.8,
            model_used="agentic-react-engine",
        )

    def _synthesize_answer(self, query: str, tool_outputs: List[Dict[str, Any]]) -> str:
        """Synthesize tool execution results into a comprehensive, grounded answer."""
        if not tool_outputs:
            return "I cannot find sufficient information in your documents to answer this question."

        parts = []
        for item in tool_outputs:
            tool_name = item["tool"]
            output = item["output"]

            if tool_name == "compare_documents":
                doc1 = output.get("doc1", {})
                doc2 = output.get("doc2", {})
                diffs = output.get("differences", [])
                parts.append(f"### Comparison: {doc1.get('filename')} vs {doc2.get('filename')}\n")
                if diffs:
                    fn1 = doc1.get("filename", "Doc 1")
                    id1 = doc1.get("id", 1)
                    fn2 = doc2.get("filename", "Doc 2")
                    id2 = doc2.get("id", 2)
                    k1 = f"{fn1} (ID {id1})"
                    k2 = f"{fn2} (ID {id2})"
                    for d in diffs:
                        ent = d.get("entity")
                        v1 = d.get(k1, "N/A")
                        v2 = d.get(k2, "N/A")
                        parts.append(f"- **{ent}**: {fn1} = {v1}, {fn2} = {v2}")
                else:
                    parts.append("No major differences detected across key entities.")

            elif tool_name == "get_expiring_documents":
                exp = output.get("expiring_documents", [])
                if exp:
                    parts.append("### Upcoming Expirations:\n")
                    for e in exp:
                        parts.append(f"- **{e.get('filename')}** ({e.get('category')}): Expiry Date **{e.get('expiry_date')}** [Doc ID: {e.get('document_id')}]")
                else:
                    parts.append("No expiring documents found within the requested timeframe.")

            elif tool_name == "extract_timeline":
                events = output.get("timeline", [])
                if events:
                    parts.append("### Chronological Timeline:\n")
                    for ev in events:
                        parts.append(f"- **{ev.get('date_str')}** — {ev.get('event_type')} ({ev.get('filename')}, Page {ev.get('page')})")
                else:
                    parts.append("No dated timeline events found.")

            elif tool_name == "summarize_document":
                summary = output.get("summary", "")
                parts.append(summary)

            elif tool_name == "search_by_entity":
                matches = output.get("matches", [])
                if matches:
                    parts.append(f"### Matches for {output.get('entity_type')} {output.get('value') or ''}:\n")
                    for m in matches:
                        parts.append(f"- Found in **{m.get('filename')}** (Page {m.get('source_page', 1)}) [Doc ID: {m.get('document_id')}]")
                else:
                    parts.append("No matching documents found for this entity.")

            elif tool_name in ["hybrid_search", "search_documents", "semantic_search"]:
                results = output.get("results", [])
                if results:
                    ans = rag_chain.generate_answer_local(query, results)
                    parts.append(ans)
                else:
                    parts.append("I cannot find sufficient information in your documents to answer this question.")

        return "\n\n".join(parts)


document_agent = DocumentAgent()
