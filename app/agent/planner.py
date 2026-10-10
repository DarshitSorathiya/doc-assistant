# ============================================================
# Agent Planner — ReAct Reasoning & Tool Orchestration
# ============================================================
# Decides which tools to invoke and sequences multi-step execution.
# Supports LLM Function Calling + Deterministic Rule-Based Planner.
# ============================================================

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.agent.tools import tool_registry

logger = logging.getLogger(__name__)


class AgentPlanStep:
    """Represents a planned tool call."""
    def __init__(self, tool_name: str, arguments: Dict[str, Any]):
        self.tool_name = tool_name
        self.arguments = arguments

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "arguments": self.arguments}


class AgentPlanner:
    """Plans tool execution sequence based on query intent."""

    def __init__(self):
        self.settings = get_settings()

    def plan_rule_based(self, query: str, user_id: int, db: Session) -> List[AgentPlanStep]:
        """Deterministic intent analysis and tool sequencing when external LLM is offline."""
        q_lower = query.lower()
        steps: List[AgentPlanStep] = []

        # 1. Compare Intent
        if "compare" in q_lower or "difference" in q_lower or "versus" in q_lower or " vs " in q_lower:
            # First search for relevant candidate documents
            cat = "Insurance" if "insurance" in q_lower or "policy" in q_lower else None
            search_res = tool_registry.execute_tool(
                "hybrid_search",
                user_id=user_id,
                db=db,
                kwargs={"query": query, "category": cat, "top_k": 5},
            )
            doc_ids = []
            for r in search_res.get("results", []):
                doc_id = r.get("document_id")
                if doc_id and doc_id not in doc_ids:
                    doc_ids.append(doc_id)

            if len(doc_ids) >= 2:
                steps.append(AgentPlanStep("compare_documents", {"doc_id_1": doc_ids[0], "doc_id_2": doc_ids[1]}))
                if "expir" in q_lower:
                    steps.append(AgentPlanStep("get_expiring_documents", {"days": 365}))
                return steps

        # 2. Expiry Intent
        if "expir" in q_lower or "due date" in q_lower or "valid till" in q_lower:
            steps.append(AgentPlanStep("get_expiring_documents", {"days": 365}))
            steps.append(AgentPlanStep("hybrid_search", {"query": query, "top_k": 5}))
            return steps

        # 3. Timeline Intent
        if "timeline" in q_lower or "chronolog" in q_lower or "schedule of events" in q_lower:
            steps.append(AgentPlanStep("extract_timeline", {}))
            return steps

        # 4. Duplicate Intent
        if "duplicate" in q_lower or "near duplicate" in q_lower or "copies" in q_lower:
            steps.append(AgentPlanStep("find_duplicates", {}))
            return steps

        # 5. Entity Search Intent (PAN, Aadhaar, IFSC, etc.)
        pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", query, re.IGNORECASE)
        if pan_match:
            steps.append(AgentPlanStep("search_by_entity", {"entity_type": "PAN", "value": pan_match.group(0).upper()}))
            return steps

        if "aadhaar" in q_lower:
            steps.append(AgentPlanStep("search_by_entity", {"entity_type": "Aadhaar"}))
            return steps

        # 6. Summarize Intent
        if "summarize" in q_lower or "summary" in q_lower:
            search_res = tool_registry.execute_tool(
                "hybrid_search",
                user_id=user_id,
                db=db,
                kwargs={"query": query, "top_k": 1},
            )
            results = search_res.get("results", [])
            if results:
                doc_id = results[0].get("document_id")
                steps.append(AgentPlanStep("summarize_document", {"document_id": doc_id}))
                return steps

        # Default fallback: Hybrid Search
        steps.append(AgentPlanStep("hybrid_search", {"query": query, "top_k": 5}))
        return steps


agent_planner = AgentPlanner()
