"""RAG Evaluation Runner — python -m app.evaluation

Runs retrieval and generation evaluation against a sample dataset
and writes results to evaluation_results.json.
"""

import json
import logging
import sys
from pathlib import Path

from app.evaluation.retrieval import retrieval_evaluator
from app.evaluation.generation import generation_evaluator

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Sample evaluation dataset (fixture data for smoke testing)
# ------------------------------------------------------------------
SAMPLE_RETRIEVAL_EVAL_SET = [
    {
        "query": "What is my health insurance policy number?",
        "relevant_doc_ids": [1, 3],
    },
    {
        "query": "Show documents related to my property",
        "relevant_doc_ids": [2, 5],
    },
    {
        "query": "Find my PAN card",
        "relevant_doc_ids": [4],
    },
]

SAMPLE_GENERATION_EVAL_SET = [
    {
        "question": "What is my policy number?",
        "answer": "Your health insurance policy number is POL-998877, issued by Star Health Insurance.",
        "context_chunks": [
            {"text": "Policy Number: POL-998877. Issuer: Star Health Insurance.", "filename": "star_health.pdf", "page": 1},
        ],
        "cited_sources": [
            {"document": "star_health.pdf", "page": 1},
        ],
    },
    {
        "question": "When does my passport expire?",
        "answer": "I don't have enough information to determine your passport expiry date.",
        "context_chunks": [],
        "cited_sources": [],
    },
    {
        "question": "Compare my two insurance policies",
        "answer": "Policy A (Star Health) covers up to 5 lakhs with a premium of 12000/yr. Policy B (HDFC Ergo) covers up to 10 lakhs with a premium of 18001/yr.",
        "context_chunks": [
            {"text": "Star Health: Coverage 5L, Premium 12000", "filename": "star_health.pdf", "page": 1},
            {"text": "HDFC Ergo: Coverage 10L, Premium 18001", "filename": "hdfc_ergo.pdf", "page": 2},
        ],
        "cited_sources": [
            {"document": "star_health.pdf", "page": 1},
            {"document": "hdfc_ergo.pdf", "page": 2},
        ],
    },
]


def run_evaluation():
    """Run both retrieval and generation evaluation and save results."""
    results = {}

    # ------ Retrieval evaluation ------
    logger.info("Running retrieval evaluation...")

    # Since we don't have a live retrieval function in this context,
    # we simulate with mock retrieved IDs
    def mock_retrieval_fn(query, db, user_id):
        """Simulate retrieval — returns some IDs for testing."""
        mock_results = {
            "What is my health insurance policy number?": [1, 7, 3, 8, 2],
            "Show documents related to my property": [5, 2, 9, 1, 3],
            "Find my PAN card": [4, 6, 1, 2, 3],
        }
        return mock_results.get(query, [1, 2, 3])

    retrieval_results = retrieval_evaluator.evaluate_batch(
        eval_set=SAMPLE_RETRIEVAL_EVAL_SET,
        retrieval_fn=mock_retrieval_fn,
    )
    results["retrieval"] = retrieval_results

    logger.info(f"Retrieval evaluation complete — {retrieval_results['num_queries']} queries evaluated.")
    logger.info(f"  Averages: {json.dumps(retrieval_results['averages'], indent=2)}")

    # ------ Generation evaluation ------
    logger.info("Running generation evaluation...")

    generation_results = generation_evaluator.evaluate_batch(
        eval_set=SAMPLE_GENERATION_EVAL_SET,
    )
    results["generation"] = generation_results

    logger.info(f"Generation evaluation complete — {generation_results['num_items']} items evaluated.")
    logger.info(f"  Summary: {json.dumps(generation_results['summary'], indent=2)}")

    # ------ Save results ------
    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path.resolve()}")

    return results


if __name__ == "__main__":
    run_evaluation()
