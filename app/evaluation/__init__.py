"""RAG evaluation — retrieval and generation quality metrics (Phase 14).

Entry point:  python -m app.evaluation
"""

from app.evaluation.retrieval import (
    retrieval_evaluator,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from app.evaluation.generation import (
    generation_evaluator,
    citation_correctness,
    context_coverage,
    answer_length_check,
    hedging_detection,
)
