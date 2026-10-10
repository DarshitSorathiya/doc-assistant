# ============================================================
# Knowledge Graph Manager (NetworkX In-Process Engine)
# ============================================================
# Connects Persons, Policies, Bank Accounts, Properties, and Documents
# with semantic relationships, enforcing strict user_id isolation.
# ============================================================

import logging
import re
import networkx as nx
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from app.models import Document, Entity, DocumentStatus
from app.knowledge_graph.entities import NodeType, GraphNode
from app.knowledge_graph.relationships import RelationType, GraphEdge

logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """Builds, manages, and queries in-process knowledge graphs partitioned by user."""

    def __init__(self):
        # Maps user_id -> nx.MultiDiGraph
        self._user_graphs: Dict[int, nx.MultiDiGraph] = {}

    def get_or_build_graph(self, db: Session, user_id: int) -> nx.MultiDiGraph:
        """Retrieve cached graph for user or build fresh from database."""
        G = nx.MultiDiGraph()

        docs = db.query(Document).filter(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY,
        ).all()

        doc_ids = [doc.id for doc in docs]
        entities_by_document: Dict[int, List[Entity]] = {}
        if doc_ids:
            all_entities = db.query(Entity).filter(Entity.document_id.in_(doc_ids)).all()
            for entity in all_entities:
                entities_by_document.setdefault(entity.document_id, []).append(entity)

        for doc in docs:
            doc_node_id = f"doc_{doc.id}"
            G.add_node(
                doc_node_id,
                label=doc.filename,
                node_type=NodeType.DOCUMENT.value,
                document_id=doc.id,
                category=doc.category.value if doc.category else "Uncategorized",
                page_count=doc.page_count,
            )

            for ent in entities_by_document.get(doc.id, []):
                ent_type_lower = ent.entity_type.lower()
                ent_val = ent.value.strip()
                canonical_value = re.sub(r"\s+", " ", ent_val).casefold()

                if not ent_val:
                    continue

                if "policy" in ent_type_lower:
                    node_id = f"policy_{canonical_value}"
                    G.add_node(node_id, label=f"Policy: {ent_val}", node_type=NodeType.POLICY.value, value=ent_val)
                    G.add_edge(doc_node_id, node_id, relation=RelationType.HAS_POLICY.value)

                elif "account" in ent_type_lower or "ifsc" in ent_type_lower:
                    node_id = f"bank_{canonical_value}"
                    G.add_node(node_id, label=f"Bank Account: {ent_val}", node_type=NodeType.BANK_ACCOUNT.value, value=ent_val)
                    G.add_edge(doc_node_id, node_id, relation=RelationType.HAS_ACCOUNT.value)

                elif "property" in ent_type_lower or "khata" in ent_type_lower or "survey" in ent_type_lower:
                    node_id = f"property_{canonical_value}"
                    G.add_node(node_id, label=f"Property: {ent_val[:80]}", node_type=NodeType.PROPERTY.value, value=ent_val)
                    G.add_edge(doc_node_id, node_id, relation=RelationType.LOCATED_AT.value)

                elif "pan" in ent_type_lower or "aadhaar" in ent_type_lower or "passport" in ent_type_lower or "license" in ent_type_lower:
                    node_id = f"tax_{canonical_value}"
                    G.add_node(node_id, label=f"{ent.entity_type}: {ent_val}", node_type=NodeType.TAX_ID.value, value=ent_val)
                    G.add_edge(doc_node_id, node_id, relation=RelationType.CONTAINS.value)

                elif "name" in ent_type_lower or "holder" in ent_type_lower or "purchaser" in ent_type_lower:
                    node_id = f"person_{canonical_value}"
                    G.add_node(node_id, label=ent_val, node_type=NodeType.PERSON.value, value=ent_val)
                    G.add_edge(node_id, doc_node_id, relation=RelationType.OWNS.value)

                elif "issuer" in ent_type_lower or "bank" in ent_type_lower or "organization" in ent_type_lower:
                    node_id = f"org_{canonical_value}"
                    G.add_node(node_id, label=ent_val, node_type=NodeType.ORGANIZATION.value, value=ent_val)
                    G.add_edge(node_id, doc_node_id, relation=RelationType.ISSUED_BY.value)

        self._user_graphs[user_id] = G
        return G

    def query_graph(
        self,
        db: Session,
        user_id: int,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        document_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query user's knowledge graph and return nodes and edges."""
        G = self.get_or_build_graph(db, user_id)

        candidate_nodes: Set[str] = set(G.nodes())
        if document_id is not None:
            doc_node_id = f"doc_{document_id}"
            if doc_node_id not in G:
                return {"total_nodes": 0, "total_edges": 0, "nodes": [], "edges": []}
            candidate_nodes = {doc_node_id}
            candidate_nodes.update(G.neighbors(doc_node_id))
            candidate_nodes.update(G.predecessors(doc_node_id))

        q_lower = (query or "").strip().casefold()
        et_lower = (entity_type or "").strip().casefold()
        if entity_type or query:
            matched_nodes = set()
            for node_id in candidate_nodes:
                data = G.nodes[node_id]
                node_type = str(data.get("node_type", "")).casefold()
                label = str(data.get("label", "")).casefold()
                value = str(data.get("value", "")).casefold()
                type_match = not et_lower or et_lower in node_type
                text_match = not q_lower or q_lower in label or q_lower in value
                if type_match and text_match:
                    matched_nodes.add(node_id)
            candidate_nodes = matched_nodes

        target_nodes = set(candidate_nodes)
        if entity_type or query:
            for node_id in candidate_nodes:
                target_nodes.update(G.neighbors(node_id))
                target_nodes.update(G.predecessors(node_id))

        # Construct response
        nodes_res: List[GraphNode] = []
        for nid in sorted(target_nodes):
            if nid in G.nodes:
                data = G.nodes[nid]
                nt_str = data.get("node_type", NodeType.DOCUMENT.value)
                try:
                    nt_enum = NodeType(nt_str)
                except ValueError:
                    nt_enum = NodeType.DOCUMENT

                nodes_res.append(
                    GraphNode(
                        id=nid,
                        label=data.get("label", nid),
                        node_type=nt_enum,
                        properties={k: v for k, v in data.items() if k not in ["label", "node_type"]},
                    )
                )

        edges_res: List[GraphEdge] = []
        for u, v, k, data in G.edges(keys=True, data=True):
            if u in target_nodes and v in target_nodes:
                rel_str = data.get("relation", RelationType.ASSOCIATED_WITH.value)
                try:
                    rel_enum = RelationType(rel_str)
                except ValueError:
                    rel_enum = RelationType.ASSOCIATED_WITH

                edges_res.append(
                    GraphEdge(
                        source=u,
                        target=v,
                        relation=rel_enum,
                        properties={k: v for k, v in data.items() if k != "relation"},
                    )
                )

        return {
            "total_nodes": len(nodes_res),
            "total_edges": len(edges_res),
            "nodes": [n.model_dump() for n in nodes_res],
            "edges": [e.model_dump() for e in edges_res],
        }

    def get_connected_documents(self, db: Session, user_id: int, entity_value: str) -> List[Dict[str, Any]]:
        """Find all documents connected to a specific entity value."""
        G = self.get_or_build_graph(db, user_id)
        val_lower = entity_value.lower()

        matched_entity_nodes = [
            n for n, d in G.nodes(data=True)
            if val_lower in d.get("label", "").lower() or val_lower in str(d.get("value", "")).lower()
        ]

        doc_nodes = set()
        for enode in matched_entity_nodes:
            # Check predecessors & successors for Document nodes
            for neighbor in list(G.neighbors(enode)) + list(G.predecessors(enode)):
                if G.nodes[neighbor].get("node_type") == NodeType.DOCUMENT.value:
                    doc_nodes.add(neighbor)

        connected_docs = []
        for dnode in doc_nodes:
            d_data = G.nodes[dnode]
            connected_docs.append({
                "document_id": d_data.get("document_id"),
                "filename": d_data.get("label"),
                "category": d_data.get("category"),
            })

        return connected_docs


kg_manager = KnowledgeGraphManager()
