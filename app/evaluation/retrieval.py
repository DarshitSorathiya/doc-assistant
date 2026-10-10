# ============================================================
# Retrieval Evaluation — Precision@K, Recall@K, MRR, Hit Rate, NDCG
# ============================================================
# Measures retrieval quality against a labeled evaluation set.
# Each item in the eval set: { query, relevant_doc_ids, relevant_pages }
# ============================================================

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[int], relevant_ids: set, k: int) -> float:
    """Fraction of top-K results that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[int], relevant_ids: set, k: int) -> float:
    """Fraction of relevant documents captured in top-K results.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Recall@K in [0, 1].
    """
    if not relevant_ids:
        return 1.0  # vacuously true
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: List[int], relevant_ids: set, k: int) -> float:
    """Binary: 1.0 if any relevant document appears in top-K, else 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: set) -> float:
    """Reciprocal of the rank of the first relevant result.

    Returns:
        1/rank of first hit, or 0.0 if no relevant doc found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: 1 if in relevant set, 0 otherwise.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        NDCG@K in [0, 1].
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because ranks are 1-indexed

    # Ideal DCG — all relevant docs at the top
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ------------------------------------------------------------------
# Evaluation runner
# ------------------------------------------------------------------

class RetrievalEvaluator:
    """Run retrieval evaluation against a labeled dataset."""

    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate_single(
        self,
        retrieved_ids: List[int],
        relevant_ids: set,
    ) -> Dict[str, Any]:
        """Evaluate a single query's retrieval results.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.

        Returns:
            Dict with metrics at each K value plus MRR.
        """
        results = {"mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids)}

        for k in self.k_values:
            results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            results[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
            results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

        return results

    def evaluate_batch(
        self,
        eval_set: List[Dict[str, Any]],
        retrieval_fn,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run evaluation across an entire labeled eval set.

        Args:
            eval_set: List of dicts, each with:
                - "query": str
                - "relevant_doc_ids": List[int]
            retrieval_fn: Callable(query, db, user_id) -> List[int] of doc IDs.
            db: SQLAlchemy session.
            user_id: User ID for scoped retrieval.

        Returns:
            Dict with per-query results and aggregate averages.
        """
        per_query_results = []

        for item in eval_set:
            query = item["query"]
            relevant = set(item["relevant_doc_ids"])

            try:
                retrieved = retrieval_fn(query, db, user_id)
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query}': {e}")
                retrieved = []

            metrics = self.evaluate_single(retrieved, relevant)
            metrics["query"] = query
            per_query_results.append(metrics)

        # Compute averages
        averages = self._compute_averages(per_query_results)

        return {
            "num_queries": len(eval_set),
            "k_values": self.k_values,
            "per_query": per_query_results,
            "averages": averages,
        }

    def _compute_averages(self, per_query_results: List[Dict]) -> Dict[str, float]:
        """Average each metric across all queries."""
        if not per_query_results:
            return {}

        # Collect all numeric metric keys
        metric_keys = [k for k in per_query_results[0] if k != "query"]
        averages = {}
        for key in metric_keys:
            values = [r[key] for r in per_query_results if isinstance(r.get(key), (int, float))]
            if values:
                averages[f"avg_{key}"] = sum(values) / len(values)

        return averages


# Singleton
retrieval_evaluator = RetrievalEvaluator()
