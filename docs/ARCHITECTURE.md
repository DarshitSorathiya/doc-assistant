# Architecture — Personal Document Finder

> This document is updated incrementally as each phase is built.

## Overview

Personal Document Finder is a local-first, AI-powered document intelligence platform.

## High-Level Architecture (Phase 6)

```
┌──────────────┐     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Streamlit   │────▶│                                           FastAPI REST API                                             │
│     UI       │     │                                            (app/main.py)                                               │
└──────────────┘     └───────┬──────────────┬───────────────┬───────────────┬─────────────────┬───────────────┬─────────────┬─┘
                             │              │               │               │                 │               │             │
                     ┌───────▼──────┐ ┌─────▼───────┐ ┌─────▼───────┐ ┌─────▼───────┐ ┌───────▼───────┐ ┌─────▼───────┐ ┌───▼───────┐
                     │ Pydantic     │ │ Auth / RBAC │ │ Document    │ │ OCR Pipeline│ │ Extraction    │ │ Vector Store│ │SQLAlchemy │
                     │ Config       │ │ (JWT/bcrypt)│ │ Service     │ │ (app/ocr/)  │ │ (Classify/    │ │ (ChromaDB & │ │ORM Layer  │
                     │ (app/config/)│ │ (app/auth/) │ │(app/services│ │             │ │  Entities)    │ │  Embeddings)│ │(app/models│
                     └──────────────┘ └─────────────┘ └──────┬──────┘ └──────┬──────┘ └───────┬───────┘ └──────┬──────┘ └───┬───────┘
                                                             │               │                │                │            │
                                                      ┌──────▼──────┐ ┌──────▼──────┐         │         ┌──────▼──────┐ ┌───▼───────┐
                                                      │ User Disk   │ │ PaddleOCR / │         └────────▶│ ChromaDB    │ │SQLite / PG│
                                                      │ Storage     │ │ Tesseract / │                   │ (Isolated)  │ │ (WAL Mode)│
                                                      │ (/storage/) │ │ PyMuPDF     │                   └─────────────┘ └───────────┘
                                                      └─────────────┘ └─────────────┘
```

## Embeddings, Vector Store & Chunking (Phase 6)

- **Page-Aware Chunking**: Splits document pages into overlapping spans (~600 characters, ~100 character overlap) while strictly preserving page numbers (`page_number`), `document_id`, `category`, and `user_id`.
- **Dense Embedding Model**: Sentence-transformers (`all-MiniLM-L6-v2`) generating 384-dimensional normalized vector embeddings.
- **ChromaDB with Hard User-Level Isolation**:
  - All chunk upserts include `user_id` in metadata.
  - All similarity queries enforce `{"user_id": user_id}` filter unconditionally at the storage client boundary.
- **Semantic Near-Duplicate Detection**: Whole-document cosine similarity compares incoming documents against user's indexed corpus (flags similarities $\ge 0.88$).

## Classification & Entity Extraction (Phase 5)

- **12 Categories**: Insurance, Banking, Government ID, Property, Tax, Medical, Education, Employment, Legal, Financial, Utility, Other.
- **Hybrid Classification**: Weighted keyword matching (>0.65 confidence) + zero-shot LLM fallback.
- **Structured Indian Identifier Extraction**:
  - **PAN**: Format `[A-Z]{5}[0-9]{4}[A-Z]`.
  - **Aadhaar**: 12-digit pattern with Verhoeff checksum validation.
  - **IFSC**: 11-character bank branch routing codes.
  - **Passport & Driving License**: Regex and format matching.
  - **Policy Numbers & Bank Accounts**: Contextual account string extractors.
  - **Expiry Dates & Amounts**: Identifies upcoming expiry deadlines and financial values.
- **Database Persistence**: Extracted entities are stored in the relational `entities` table linked to documents for structured queries.

## OCR & Text Extraction Pipeline (Phase 4)

- **Digital-First Text Extraction**: Born-digital PDFs bypass OCR for fast extraction of vector-ready text, bounding boxes, and structured 2D tables via `PyMuPDF` and `pdfplumber`.
- **PaddleOCR Engine (Primary)**: High-accuracy deep learning OCR with layout analysis and text line detection.
- **Tesseract Engine (Fallback)**: Open-source fallback with image preprocessing and word-level confidence scoring.
- **Per-Page Indexing**: Stores per-page text, confidence metrics, and table structures in `PageResult` for grounded citations.

## Storage Layout & User Isolation (Phase 3)

Document binaries are isolated on the filesystem by user ID and document ID:
```
storage/
└── {user_id}/
    └── {document_id}/
        ├── v1_{filename}
        └── v2_{filename}
```

- **File Validation**: Mime-type and extension checking, empty file rejection, and `MAX_UPLOAD_SIZE_MB` enforcement.
- **SHA-256 Deduplication**: Exact file hashes detect duplicate uploads per user.
- **Version Tracking**: Explicit re-uploads create `DocumentVersion` rows with historical file paths and change summaries.
- **Security**: Strict authorization checks verify `document.user_id == current_user.id` on all read/update/delete operations.

## Data Models (Phase 2 & 3)

- **`User`**: Account info, bcrypt hashed password, role enum (`admin`, `user`, `viewer`), active status, timestamps.
- **`Document`**: Metadata (filename, mime type, sha256 hash, size, file path), status (`uploaded`, `processing`, `ready`, `failed`), classification category & confidence, extracted text, ownership relation.
- **`DocumentVersion`**: Document re-upload version snapshots.
- **`Entity`**: Extracted entities (`PAN`, `Aadhaar`, `Date`, `Amount`, etc.) with confidence and page numbers.
- **`Tag`**: Flexible tagging system for categorization and search.
- **`AuditLog`**: Immutable event trail capturing all user actions, resource modifications, and IP addresses.

## Hybrid Search & Reranking (Phase 7)

- **Dual retrieval**: Vector (ChromaDB) and keyword (BM25) searches run in parallel.
- **Reciprocal Rank Fusion**: Merges ranked lists from both retrievers.
- **Metadata filtering**: Category, date range, entity type/value filters applied at query time.
- **Reranker**: Cross-encoder or LLM-based reranking cuts top-20 → top-5 candidates.
- **User isolation**: All queries enforce `user_id` filtering — never delegated to the LLM.

## RAG Chain & Conversational Memory (Phase 8)

- **Pipeline**: Query understanding → intent detection → hybrid retrieval → reranking → context construction → LLM generation → grounded answer.
- **Citations**: Every answer carries `sources: [{document, page}]`.
- **Insufficient evidence**: Chain explicitly says "I don't have enough information" rather than hallucinating.
- **Conversational memory**: Session-scoped history resolves follow-up questions and pronouns.

## Agentic RAG (Phase 9)

- **11 tools**: search_documents, semantic_search, hybrid_search, get_document, get_document_metadata, search_by_entity, summarize_document, compare_documents, extract_timeline, find_duplicates, get_expiring_documents.
- **Planner**: LLM function-calling / ReAct loop with max iteration cap and timeout.
- **Agent**: Orchestrates planner → tool execution → answer synthesis with audit logging.

## Document Services (Phase 10)

- **Summary**: Per-document and cross-document (by category) summarization.
- **Comparison**: Structured diff of two documents — detects added/removed/changed values.
- **Timeline**: Chronologically sorted date events across all user documents.
- **Duplicates**: Exact-hash and semantic near-duplicate detection.
- **Insights**: Category distribution, risk metrics, and portfolio analytics.

## Knowledge Graph (Phase 11)

- **Graph engine**: NetworkX in-process, persisted as node/edge structures.
- **Node types**: Person, Policy, BankAccount, Property, TaxIdentifier, Document.
- **Edges**: `owns`, `has`, `associated_with` — populated from extracted entities during ingestion.
- **Queries**: Filter by entity type, search node labels, 1-hop subgraph from a document.

## Proactive Alerts (Phase 11)

- **Expiry alerts**: Documents with upcoming expiration dates (configurable warning window).
- **Missing document detection**: Heuristics (e.g., property docs without insurance).
- **Duplicate alerts**: Flags exact and near-duplicate documents.
- **Version change alerts**: Notifies when document versions differ.
- **Severity levels**: Critical, warning, info.

## MCP Server (Phase 12)

- **Protocol**: Model Context Protocol over stdio.
- **Tools**: Same 11 tools as the agentic layer, with MCP-compatible schemas.
- **Resources**: Document listing, document detail, entity listing via URI patterns.
- **Auth**: JWT or user_id in session metadata; same isolation as REST API.
- **Audit**: Every tool call logged with `mcp:` prefix.

## Audit Logging (Phase 12)

- **Scope**: Every API call, tool invocation, and MCP call is logged.
- **Data**: User ID, action, resource type/ID, details JSON, IP address, timestamp.
- **Access**: Admin-only `GET /api/v1/audit` endpoint with pagination and filtering.
- **Immutability**: `audit_logs` table is append-only.

## Streamlit UI (Phase 13)

- **7 screens**: Login/Register, Documents, Chat Q&A, Search, Insights & Alerts, Timeline, Knowledge Graph, Compare.
- **API-driven**: All UI calls go through the REST API with JWT — no direct DB access.
- **Design**: Glassmorphism, gradient backgrounds, animated cards, responsive layout.

## RAG Evaluation (Phase 14)

- **Retrieval**: Precision@K, Recall@K, MRR, Hit Rate, NDCG@K.
- **Generation**: Citation correctness, context coverage, hedging detection, answer length, faithfulness (LLM-as-judge), answer relevance, context relevance.
- **Entry point**: `python -m app.evaluation` → outputs `evaluation_results.json`.

## Security Architecture

- **Authentication**: JWT with bcrypt password hashing.
- **RBAC**: Admin > User > Viewer role hierarchy enforced per route.
- **Data isolation**: User-scoped queries at every layer (DB, vector store, MCP).
- **File validation**: MIME type, extension, size limit enforcement.
- **Secret management**: All secrets via environment variables, never logged.
- **Audit trail**: Complete, immutable logging of all actions.

