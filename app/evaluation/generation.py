# ============================================================
# Generation Evaluation — Faithfulness, Relevance, Citations
# ============================================================
# Evaluates the quality of generated RAG answers using
# LLM-as-judge or heuristic checks.
#
# Metrics:
#   - Answer relevance: Does the answer address the question?
#   - Faithfulness: Is the answer grounded in the context?
#   - Context relevance: Are the retrieved contexts useful?
#   - Citation correctness: Do cited sources actually support claims?
#   - Hallucination rate: Fraction of statements not in context.
# ============================================================

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Heuristic / rule-based evaluators (no LLM required)
# ------------------------------------------------------------------

def citation_correctness(
    answer: str,
    cited_sources: List[Dict[str, Any]],
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check whether cited sources actually appear in the context.

    Args:
        answer: Generated answer text.
        cited_sources: List of source dicts with ``document`` and ``page`` keys.
        context_chunks: List of chunk dicts with ``document_id`` / ``filename``
                        and ``page`` keys that were actually retrieved.

    Returns:
        Dict with:
          - ``total_citations``: Number of citations in the answer.
          - ``valid_citations``: Citations that match a retrieved chunk.
          - ``correctness_score``: valid / total (or 1.0 if none cited).
          - ``invalid_citations``: List of citations not found in context.
    """
    if not cited_sources:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "correctness_score": 1.0,  # vacuously correct
            "invalid_citations": [],
        }

    # Build a set of (filename-or-id, page) from context
    context_keys = set()
    for chunk in context_chunks:
        fname = chunk.get("filename") or str(chunk.get("document_id", ""))
        page = chunk.get("page")
        context_keys.add((fname.lower(), page))
        # Also add without page for looser matching
        context_keys.add((fname.lower(), None))

    valid = 0
    invalid = []
    for src in cited_sources:
        src_doc = (src.get("document", "") or "").lower()
        src_page = src.get("page")
        if (src_doc, src_page) in context_keys or (src_doc, None) in context_keys:
            valid += 1
        else:
            invalid.append(src)

    total = len(cited_sources)
    return {
        "total_citations": total,
        "valid_citations": valid,
        "correctness_score": valid / total if total > 0 else 1.0,
        "invalid_citations": invalid,
    }


def answer_has_sources(answer: str, sources: List[Dict]) -> bool:
    """Quick check: does the answer come with at least one source?"""
    return len(sources) > 0


def context_coverage(
    answer: str,
    context_chunks: List[Dict[str, Any]],
) -> float:
    """Estimate how much of the context was actually used in the answer.

    Uses simple word-overlap heuristic. Returns a score in [0, 1].
    """
    if not context_chunks or not answer:
        return 0.0

    answer_words = set(re.findall(r"\w+", answer.lower()))
    if not answer_words:
        return 0.0

    chunks_used = 0
    for chunk in context_chunks:
        chunk_text = chunk.get("text", chunk.get("chunk_text", ""))
        chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
        # If significant overlap, count this chunk as used
        if chunk_words:
            overlap = len(answer_words & chunk_words) / len(chunk_words)
            if overlap > 0.05:
                chunks_used += 1

    return chunks_used / len(context_chunks) if context_chunks else 0.0


def answer_length_check(answer: str, min_words: int = 5, max_words: int = 500) -> Dict[str, Any]:
    """Check if the answer length is within reasonable bounds."""
    words = answer.split()
    word_count = len(words)
    return {
        "word_count": word_count,
        "too_short": word_count < min_words,
        "too_long": word_count > max_words,
        "acceptable": min_words <= word_count <= max_words,
    }


def hedging_detection(answer: str) -> Dict[str, Any]:
    """Detect excessive hedging or refusal to answer.

    Looks for phrases like 'I don't know', 'I'm not sure',
    'insufficient information', etc.
    """
    hedging_patterns = [
        r"i don'?t know",
        r"i'?m not sure",
        r"insufficient information",
        r"cannot determine",
        r"no relevant",
        r"unable to find",
        r"not enough context",
        r"i cannot answer",
    ]
    answer_lower = answer.lower()
    hedges_found = []
    for pattern in hedging_patterns:
        if re.search(pattern, answer_lower):
            hedges_found.append(pattern)

    return {
        "is_hedging": len(hedges_found) > 0,
        "hedging_phrases": hedges_found,
        "hedging_score": len(hedges_found) / len(hedging_patterns),
    }


# ------------------------------------------------------------------
# LLM-as-judge evaluators
# ------------------------------------------------------------------

# Prompts used for LLM-based evaluation (documented per Phase 14 spec)
FAITHFULNESS_PROMPT = """You are an evaluation judge. Given a CONTEXT and an ANSWER, determine if every claim in the ANSWER is supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Rate the faithfulness on a scale of 1-5:
1 = Completely unfaithful / hallucinated
2 = Mostly unfaithful, some claims unsupported
3 = Mixed — some claims supported, some not
4 = Mostly faithful, minor unsupported details
5 = Completely faithful to the context

Output ONLY a JSON object: {{"score": <int>, "explanation": "<brief reason>"}}"""

RELEVANCE_PROMPT = """You are an evaluation judge. Given a QUESTION and an ANSWER, determine if the ANSWER addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rate the answer relevance on a scale of 1-5:
1 = Completely irrelevant
2 = Mostly irrelevant
3 = Partially relevant
4 = Mostly relevant
5 = Perfectly addresses the question

Output ONLY a JSON object: {{"score": <int>, "explanation": "<brief reason>"}}"""

CONTEXT_RELEVANCE_PROMPT = """You are an evaluation judge. Given a QUESTION and CONTEXT chunks, determine if the CONTEXT is useful for answering the QUESTION.

QUESTION:
{question}

CONTEXT:
{context}

Rate the context relevance on a scale of 1-5:
1 = Completely irrelevant context
2 = Mostly irrelevant
3 = Partially relevant
4 = Mostly relevant
5 = Highly relevant context

Output ONLY a JSON object: {{"score": <int>, "explanation": "<brief reason>"}}"""


def _call_llm_judge(prompt: str, llm_fn=None) -> Dict[str, Any]:
    """Call LLM with a judge prompt and parse the response.

    Args:
        prompt: The formatted evaluation prompt.
        llm_fn: Optional callable(prompt) -> str. If None, returns a
                 placeholder indicating LLM is not configured.
    """
    if llm_fn is None:
        return {
            "score": None,
            "explanation": "LLM judge not configured — pass llm_fn to enable.",
        }

    try:
        response_text = llm_fn(prompt)
        # Try to extract JSON
        import json
        # Find JSON in response
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"score": None, "explanation": response_text[:500]}
    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        return {"score": None, "explanation": f"Error: {str(e)}"}


def evaluate_faithfulness(
    answer: str,
    context: str,
    llm_fn=None,
) -> Dict[str, Any]:
    """Evaluate whether the answer is faithful to the context."""
    prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:1000])
    return _call_llm_judge(prompt, llm_fn)


def evaluate_relevance(
    question: str,
    answer: str,
    llm_fn=None,
) -> Dict[str, Any]:
    """Evaluate whether the answer is relevant to the question."""
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer[:1000])
    return _call_llm_judge(prompt, llm_fn)


def evaluate_context_relevance(
    question: str,
    context: str,
    llm_fn=None,
) -> Dict[str, Any]:
    """Evaluate whether the retrieved context is relevant to the question."""
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context[:3000])
    return _call_llm_judge(prompt, llm_fn)


# ------------------------------------------------------------------
# Batch evaluator
# ------------------------------------------------------------------

class GenerationEvaluator:
    """Run generation evaluation over a batch of Q&A items."""

    def __init__(self, llm_fn=None):
        """
        Args:
            llm_fn: Optional callable(prompt: str) -> str for LLM-based evaluation.
        """
        self.llm_fn = llm_fn

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        cited_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a single Q&A pair.

        Returns a dict of all metric scores.
        """
        context_text = "\n---\n".join(
            c.get("text", c.get("chunk_text", "")) for c in context_chunks
        )

        results = {
            "question": question,
            "answer_length": answer_length_check(answer),
            "hedging": hedging_detection(answer),
            "citation_correctness": citation_correctness(answer, cited_sources, context_chunks),
            "has_sources": answer_has_sources(answer, cited_sources),
            "context_coverage": context_coverage(answer, context_chunks),
        }

        # LLM-based metrics (if llm_fn available)
        results["faithfulness"] = evaluate_faithfulness(answer, context_text, self.llm_fn)
        results["relevance"] = evaluate_relevance(question, answer, self.llm_fn)
        results["context_relevance"] = evaluate_context_relevance(
            question, context_text, self.llm_fn
        )

        return results

    def evaluate_batch(
        self,
        eval_set: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run evaluation across a batch.

        Args:
            eval_set: List of dicts, each with:
                - "question": str
                - "answer": str
                - "context_chunks": List[Dict]
                - "cited_sources": List[Dict]

        Returns:
            Dict with per-item results and aggregate summaries.
        """
        per_item = []
        for item in eval_set:
            result = self.evaluate_single(
                question=item["question"],
                answer=item["answer"],
                context_chunks=item.get("context_chunks", []),
                cited_sources=item.get("cited_sources", []),
            )
            per_item.append(result)

        # Aggregate
        summary = self._compute_summary(per_item)
        return {
            "num_items": len(eval_set),
            "per_item": per_item,
            "summary": summary,
        }

    def _compute_summary(self, per_item: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics across all items."""
        if not per_item:
            return {}

        n = len(per_item)
        summary = {
            "avg_citation_correctness": sum(
                r["citation_correctness"]["correctness_score"] for r in per_item
            ) / n,
            "avg_context_coverage": sum(r["context_coverage"] for r in per_item) / n,
            "sources_present_rate": sum(1 for r in per_item if r["has_sources"]) / n,
            "hedging_rate": sum(1 for r in per_item if r["hedging"]["is_hedging"]) / n,
            "avg_word_count": sum(r["answer_length"]["word_count"] for r in per_item) / n,
        }

        # LLM scores (if available)
        faith_scores = [
            r["faithfulness"]["score"] for r in per_item
            if r["faithfulness"].get("score") is not None
        ]
        if faith_scores:
            summary["avg_faithfulness"] = sum(faith_scores) / len(faith_scores)

        rel_scores = [
            r["relevance"]["score"] for r in per_item
            if r["relevance"].get("score") is not None
        ]
        if rel_scores:
            summary["avg_relevance"] = sum(rel_scores) / len(rel_scores)

        return summary


# Singleton
generation_evaluator = GenerationEvaluator()
