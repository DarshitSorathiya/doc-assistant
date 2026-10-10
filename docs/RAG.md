# RAG Pipeline — Personal Document Finder

## Overview

The RAG (Retrieval Augmented Generation) pipeline powers the document Q&A system. It combines vector-based semantic search with keyword-based BM25 search, fuses results, applies reranking, and generates grounded answers with citations.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Understanding │  ← Intent detection, entity extraction
│  & Intent Detection  │
└────────┬────────────┘
         │
    ┌────▼────┐
    │ Router  │ ← Decides: simple retrieval vs. agentic tool calling
    └────┬────┘
         │
    ┌────▼──────────────────────────┐
    │       Hybrid Retrieval        │
    │  ┌──────────┐  ┌───────────┐ │
    │  │ ChromaDB │  │  BM25     │ │
    │  │ (Vector) │  │ (Keyword) │ │
    │  └────┬─────┘  └─────┬─────┘ │
    │       └──────┬───────┘       │
    │       Reciprocal Rank Fusion  │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │         Reranker              │
    │  Cross-encoder / LLM-based   │
    │  Top-20 → Top-5              │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │     Context Construction      │
    │  + Source attribution          │
    │  + User-scoped filtering       │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │        LLM Generation         │
    │  OpenAI / Ollama (local)      │
    │  + Grounded answer            │
    │  + Citation insertion         │
    └──────────────┬───────────────┘
                   │
                   ▼
         Answer + Sources[]
```

## Components

### 1. Hybrid Search (`app/rag/hybrid_search.py`)

- **Vector search**: ChromaDB similarity search with sentence-transformers embeddings (`all-MiniLM-L6-v2`).
- **Keyword search**: BM25 ranking using `rank_bm25` over indexed document text.
- **Fusion**: Reciprocal Rank Fusion (RRF) merges results from both retrieval paths.
- **Metadata filters**: Category, date range, entity type/value filters applied at query time.
- **User isolation**: `user_id` filter enforced unconditionally at the vector store layer.

### 2. Reranker (`app/rag/reranker.py`)

- Cross-encoder reranking (when available) or LLM-based reranking fallback.
- Reduces candidate set from top-20 to top-5 for context construction.
- Configurable via settings.

### 3. RAG Chain (`app/rag/chain.py`)

- **Query understanding**: Extracts intent and entities from the user query.
- **Context construction**: Assembles top-K chunks with metadata.
- **LLM call**: OpenAI or Ollama backend (configurable via `LLM_PROVIDER`).
- **Citation insertion**: Every answer includes `sources: [{document, page}]`.
- **Insufficient evidence handling**: Chain explicitly states when evidence is insufficient rather than hallucinating.

### 4. Citations (`app/rag/citations.py`)

- Every generated answer carries source attribution.
- Sources include document filename and page number.
- Citations are verified against actually-retrieved chunks.

### 5. Conversational Memory (`app/rag/memory.py`)

- Session-scoped conversation history (in-memory or DB-backed).
- Follow-up questions resolve pronouns and context from prior turns.
- Sessions scoped by `user_id` — no cross-user context leakage.

## Agentic RAG (Phase 9)

The agentic layer (`app/agent/`) wraps the RAG chain with dynamic tool-calling:

- **Planner** (`app/agent/planner.py`): Decides which tools to call and in what order using LLM function-calling or ReAct-style reasoning.
- **Tool Registry** (`app/agent/tools.py`): 11 callable tools — search, compare, summarize, timeline, entity search, etc.
- **Agent** (`app/agent/agent.py`): Orchestrates planner → tool execution → answer synthesis with citations. Logs every tool call to the audit trail.
- **Max iteration cap and timeout**: Prevents infinite loops in agent reasoning.

## Evaluation (Phase 14)

### Retrieval Metrics (`app/evaluation/retrieval.py`)
- Precision@K, Recall@K, MRR, Hit Rate, NDCG@K
- Evaluated against a labeled dataset of queries → relevant documents

### Generation Metrics (`app/evaluation/generation.py`)
- **Heuristic**: Citation correctness, context coverage, answer length, hedging detection
- **LLM-as-judge**: Faithfulness, answer relevance, context relevance (prompts documented in source)

### Running Evaluation
```bash
python -m app.evaluation
```
Results are written to `evaluation_results.json`.

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `LLM_PROVIDER` | `openai` or `ollama` | `openai` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` |
| `VECTOR_DB` | Vector database backend | `chroma` |
