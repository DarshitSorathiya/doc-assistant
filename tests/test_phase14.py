# ============================================================
# Phase 14 — RAG Evaluation & Full Test Suite Tests
# ============================================================

import pytest
from app.config.settings import Settings
from app.evaluation.retrieval import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    RetrievalEvaluator,
)
from app.evaluation.generation import (
    citation_correctness,
    context_coverage,
    answer_length_check,
    hedging_detection,
    answer_has_sources,
    GenerationEvaluator,
)


# ==================================================================
# Settings & LLM Provider Tests
# ==================================================================

class TestSettingsProviders:
    def test_groq_provider_is_supported(self):
        settings = Settings(
            LLM_PROVIDER="groq",
            GROQ_API_KEY="test-key",
            GROQ_MODEL="llama-3.1-8b-instant",
        )

        assert settings.LLM_PROVIDER == "groq"
        assert settings.GROQ_API_KEY == "test-key"
        assert settings.GROQ_MODEL == "llama-3.1-8b-instant"


# ==================================================================
# Retrieval Metric Unit Tests
# ==================================================================

class TestPrecisionAtK:
    def test_all_relevant(self):
        """All top-K results are relevant."""
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_none_relevant(self):
        """No top-K results are relevant."""
        assert precision_at_k([4, 5, 6], {1, 2, 3}, 3) == 0.0

    def test_partial_relevant(self):
        """Some top-K results are relevant."""
        assert precision_at_k([1, 4, 2], {1, 2, 3}, 3) == pytest.approx(2 / 3)

    def test_k_larger_than_results(self):
        """K exceeds the number of retrieved results."""
        assert precision_at_k([1, 2], {1, 2, 3}, 5) == pytest.approx(2 / 5)

    def test_k_zero(self):
        assert precision_at_k([1, 2, 3], {1}, 0) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        """All relevant docs are in top-K."""
        assert recall_at_k([1, 2, 3, 4], {1, 2}, 4) == 1.0

    def test_partial_recall(self):
        """Only some relevant docs in top-K."""
        assert recall_at_k([1, 4, 5], {1, 2, 3}, 3) == pytest.approx(1 / 3)

    def test_no_recall(self):
        assert recall_at_k([4, 5, 6], {1, 2, 3}, 3) == 0.0

    def test_empty_relevant(self):
        """Vacuously true when no relevant docs exist."""
        assert recall_at_k([1, 2], set(), 3) == 1.0


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k([3, 4, 1], {1}, 3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k([3, 4, 5], {1}, 3) == 0.0

    def test_hit_at_position_1(self):
        assert hit_rate_at_k([1, 2, 3], {1}, 1) == 1.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank([1, 2, 3], {1}) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank([3, 1, 2], {1}) == 0.5

    def test_third_position(self):
        assert mean_reciprocal_rank([3, 4, 1], {1}) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mean_reciprocal_rank([3, 4, 5], {1}) == 0.0

    def test_multiple_relevant(self):
        """MRR only cares about the first relevant result."""
        assert mean_reciprocal_rank([3, 1, 2], {1, 2}) == 0.5


class TestNDCG:
    def test_perfect_ranking(self):
        """All relevant docs at the top."""
        assert ndcg_at_k([1, 2, 3, 4], {1, 2}, 4) == 1.0

    def test_worst_ranking(self):
        """Relevant docs pushed to the end."""
        result = ndcg_at_k([5, 6, 7, 1], {1}, 4)
        assert 0.0 < result < 1.0

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k([1, 2], {1}, 0) == 0.0


class TestRetrievalEvaluator:
    def test_evaluate_single(self):
        """evaluate_single returns all expected metrics."""
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
        result = evaluator.evaluate_single([1, 3, 5, 7, 9], {1, 5})

        assert "mrr" in result
        assert result["mrr"] == 1.0
        assert "precision@1" in result
        assert "recall@3" in result
        assert "hit_rate@5" in result
        assert "ndcg@5" in result

    def test_evaluate_batch(self):
        """evaluate_batch returns averages and per-query results."""
        evaluator = RetrievalEvaluator(k_values=[3])

        eval_set = [
            {"query": "q1", "relevant_doc_ids": [1]},
            {"query": "q2", "relevant_doc_ids": [2, 3]},
        ]

        def mock_retrieve(query, db, user_id):
            return [1, 2, 3, 4, 5]

        result = evaluator.evaluate_batch(eval_set, mock_retrieve)
        assert result["num_queries"] == 2
        assert len(result["per_query"]) == 2
        assert "averages" in result
        assert "avg_mrr" in result["averages"]

    def test_evaluate_batch_with_failing_retrieval(self):
        """Handles retrieval failures gracefully."""
        evaluator = RetrievalEvaluator(k_values=[3])

        def failing_retrieve(query, db, user_id):
            raise RuntimeError("Search failed")

        result = evaluator.evaluate_batch(
            [{"query": "fail_q", "relevant_doc_ids": [1]}],
            failing_retrieve,
        )
        assert result["num_queries"] == 1
        # Should still produce metrics (with empty retrieval)
        assert result["per_query"][0]["mrr"] == 0.0


# ==================================================================
# Generation Metric Unit Tests
# ==================================================================

class TestCitationCorrectness:
    def test_all_citations_valid(self):
        """All cited sources are in context."""
        result = citation_correctness(
            answer="The policy number is POL-123.",
            cited_sources=[{"document": "policy.pdf", "page": 1}],
            context_chunks=[{"filename": "policy.pdf", "page": 1}],
        )
        assert result["correctness_score"] == 1.0
        assert result["valid_citations"] == 1

    def test_no_citations(self):
        """No citations — vacuously correct."""
        result = citation_correctness("Answer.", [], [])
        assert result["correctness_score"] == 1.0
        assert result["total_citations"] == 0

    def test_invalid_citation(self):
        """Cited source not in context."""
        result = citation_correctness(
            answer="...",
            cited_sources=[{"document": "unknown.pdf", "page": 5}],
            context_chunks=[{"filename": "policy.pdf", "page": 1}],
        )
        assert result["correctness_score"] == 0.0
        assert len(result["invalid_citations"]) == 1

    def test_mixed_citations(self):
        """Some valid, some invalid."""
        result = citation_correctness(
            answer="...",
            cited_sources=[
                {"document": "policy.pdf", "page": 1},
                {"document": "mystery.pdf", "page": 3},
            ],
            context_chunks=[{"filename": "policy.pdf", "page": 1}],
        )
        assert result["correctness_score"] == 0.5


class TestContextCoverage:
    def test_high_coverage(self):
        """Answer uses words from context chunks."""
        score = context_coverage(
            "The insurance policy covers hospitalization expenses",
            [{"text": "Insurance policy covers hospitalization expenses up to 5 lakhs"}],
        )
        assert score > 0.0

    def test_empty_answer(self):
        assert context_coverage("", [{"text": "some context"}]) == 0.0

    def test_empty_context(self):
        assert context_coverage("some answer", []) == 0.0


class TestAnswerLengthCheck:
    def test_acceptable_length(self):
        result = answer_length_check("This is a reasonable length answer with enough words.")
        assert result["acceptable"] is True
        assert result["too_short"] is False
        assert result["too_long"] is False

    def test_too_short(self):
        result = answer_length_check("Yes.", min_words=5)
        assert result["too_short"] is True

    def test_too_long(self):
        result = answer_length_check(" ".join(["word"] * 600), max_words=500)
        assert result["too_long"] is True


class TestHedgingDetection:
    def test_hedging_detected(self):
        result = hedging_detection("I'm not sure about the answer and cannot determine the details.")
        assert result["is_hedging"] is True
        assert len(result["hedging_phrases"]) >= 1

    def test_no_hedging(self):
        result = hedging_detection("Your policy number is POL-998877.")
        assert result["is_hedging"] is False

    def test_insufficient_info_hedge(self):
        result = hedging_detection("There is insufficient information to answer this question.")
        assert result["is_hedging"] is True


class TestAnswerHasSources:
    def test_has_sources(self):
        assert answer_has_sources("answer", [{"doc": "a.pdf"}]) is True

    def test_no_sources(self):
        assert answer_has_sources("answer", []) is False


class TestGenerationEvaluator:
    def test_evaluate_single(self):
        """evaluate_single returns all expected keys."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate_single(
            question="What is my policy number?",
            answer="Your policy is POL-123.",
            context_chunks=[{"text": "Policy Number: POL-123", "filename": "p.pdf", "page": 1}],
            cited_sources=[{"document": "p.pdf", "page": 1}],
        )
        assert "answer_length" in result
        assert "hedging" in result
        assert "citation_correctness" in result
        assert "faithfulness" in result
        assert "relevance" in result

    def test_evaluate_batch(self):
        """Batch evaluation returns summary stats."""
        evaluator = GenerationEvaluator()
        eval_set = [
            {
                "question": "Q1?",
                "answer": "Answer to Q1 with some details.",
                "context_chunks": [{"text": "Q1 context", "filename": "a.pdf", "page": 1}],
                "cited_sources": [{"document": "a.pdf", "page": 1}],
            },
            {
                "question": "Q2?",
                "answer": "I don't know the answer.",
                "context_chunks": [],
                "cited_sources": [],
            },
        ]
        result = evaluator.evaluate_batch(eval_set)
        assert result["num_items"] == 2
        assert "summary" in result
        assert "avg_citation_correctness" in result["summary"]
        assert "hedging_rate" in result["summary"]
        # One hedging answer out of two
        assert result["summary"]["hedging_rate"] == 0.5


# ==================================================================
# Integration: Evaluation runner smoke test
# ==================================================================

class TestEvaluationRunner:
    def test_runner_imports(self):
        """The evaluation runner module imports successfully."""
        from app.evaluation.__main__ import run_evaluation
        assert callable(run_evaluation)

    def test_runner_executes(self, tmp_path, monkeypatch):
        """The evaluation runner completes without error."""
        monkeypatch.chdir(tmp_path)
        from app.evaluation.__main__ import run_evaluation
        results = run_evaluation()
        assert "retrieval" in results
        assert "generation" in results
        assert results["retrieval"]["num_queries"] > 0
        assert results["generation"]["num_items"] > 0
