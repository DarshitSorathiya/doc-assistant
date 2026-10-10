# API Reference — Personal Document Finder

> Updated incrementally per phase.

## Phase 1 Endpoints

### `GET /health`
- **Auth**: None
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "service": "Personal Document Finder",
  "version": "0.1.0"
}
```

## Phase 2 Endpoints (Authentication & RBAC)

### `POST /api/v1/auth/register`
- **Auth**: None
- **Request Body**:
```json
{
  "email": "user@example.com",
  "password": "strongpassword123",
  "role": "user"
}
```
- **Response**: `201 Created`
```json
{
  "id": 1,
  "email": "user@example.com",
  "role": "user",
  "is_active": true,
  "created_at": "2026-09-28T20:00:00Z"
}
```

### `POST /api/v1/auth/login`
- **Auth**: None
- **Request Body**:
```json
{
  "email": "user@example.com",
  "password": "strongpassword123"
}
```
- **Response**: `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "role": "user",
    "is_active": true,
    "created_at": "2026-09-28T20:00:00Z"
  }
}
```

### `GET /api/v1/auth/me`
- **Auth**: `Bearer <token>` in `Authorization` header
- **Response**: `200 OK`
```json
{
  "id": 1,
  "email": "user@example.com",
  "role": "user",
  "is_active": true,
  "created_at": "2026-09-28T20:00:00Z"
}
```

## Phase 3 Endpoints (Document Storage & Metadata)

### `POST /api/v1/documents/upload`
- **Auth**: `Bearer <token>`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: File upload (PDF, PNG, JPG, JPEG, TIFF)
  - `document_id` *(optional)*: Re-upload target document ID for versioning
  - `category` *(optional)*: Category string (e.g., `Insurance`, `Banking`, `Government ID`)
  - `tags` *(optional)*: Comma-separated tag list (e.g. `policy,health,2026`)
- **Response**: `201 Created`
```json
{
  "document": {
    "id": 1,
    "filename": "policy.pdf",
    "mime_type": "application/pdf",
    "file_hash": "a1b2c3d4e5...",
    "file_size": 1048576,
    "category": "Insurance",
    "status": "uploaded",
    "version": 1,
    "current_version": true,
    "user_id": 1,
    "created_at": "2026-09-28T21:00:00Z",
    "updated_at": "2026-09-28T21:00:00Z",
    "tags": [{"id": 1, "name": "health"}]
  },
  "is_duplicate": false,
  "message": "Document uploaded successfully"
}
```

### `GET /api/v1/documents`
- **Auth**: `Bearer <token>`
- **Query Parameters**:
  - `category`: Filter by category
  - `status`: Filter by document status
  - `search`: Case-insensitive substring search in filename
  - `page`: Page number (default: 1)
  - `page_size`: Page limit (default: 20, max: 100)
- **Response**: `200 OK`
```json
{
  "items": [ ... ],
  "total": 5,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

### `GET /api/v1/documents/{id}`
- **Auth**: `Bearer <token>`
- **Response**: `200 OK`
```json
{
  "id": 1,
  "filename": "policy.pdf",
  "mime_type": "application/pdf",
  "file_hash": "a1b2...",
  "file_size": 1048576,
  "category": "Insurance",
  "status": "uploaded",
  "version": 2,
  "current_version": true,
  "user_id": 1,
  "created_at": "2026-09-28T21:00:00Z",
  "updated_at": "2026-09-28T21:05:00Z",
  "tags": [],
  "versions": [
    {
      "id": 1,
      "document_id": 1,
      "version_number": 1,
      "file_path": "storage/1/1/v1_policy.pdf",
      "file_hash": "...",
      "file_size": 1040000,
      "changes_summary": "Initial upload",
      "created_at": "2026-09-28T21:00:00Z"
    },
    {
      "id": 2,
      "document_id": 1,
      "version_number": 2,
      "file_path": "storage/1/1/v2_policy.pdf",
      "file_hash": "...",
      "file_size": 1048576,
      "changes_summary": "Uploaded version 2",
      "created_at": "2026-09-28T21:05:00Z"
    }
  ]
}
```

### `DELETE /api/v1/documents/{id}`
- **Auth**: `Bearer <token>`
- **Response**: `204 No Content`

### `GET /api/v1/documents/{id}/download`
- **Auth**: `Bearer <token>`
- **Response**: Raw file stream

## Phase 4 & 5 Endpoints (OCR, Classification & Entity Extraction)

### `POST /api/v1/documents/{id}/process`
- **Auth**: `Bearer <token>`
- **Description**: Triggers OCR, automatic classification into one of 12 categories, and Indian identifier entity extraction (PAN, Aadhaar, IFSC, Passport, Driving License, Policy Number, Expiry Dates, Amounts).
- **Response**: `200 OK`
```json
{
  "id": 1,
  "filename": "lic_policy.pdf",
  "mime_type": "application/pdf",
  "file_hash": "a1b2...",
  "file_size": 1048576,
  "category": "Insurance",
  "classification_confidence": 0.88,
  "classification_method": "rule",
  "status": "ready",
  "page_count": 2,
  "version": 1,
  "current_version": true,
  "user_id": 1,
  "created_at": "2026-09-28T21:00:00Z",
  "updated_at": "2026-09-28T21:30:00Z",
  "tags": [],
  "versions": [ ... ],
  "entities": [
    {
      "id": 1,
      "entity_type": "Policy Number",
      "value": "POL-99881122",
      "confidence": 0.90,
      "source_page": 1,
      "created_at": "2026-09-28T22:00:00Z"
    },
    {
      "id": 2,
      "entity_type": "PAN",
      "value": "ABCDE1234F",
      "confidence": 0.99,
      "source_page": 1,
      "created_at": "2026-09-28T22:00:00Z"
    },
    {
      "id": 3,
      "entity_type": "Expiry Date",
      "value": "15-Aug-2035",
      "confidence": 0.95,
      "source_page": 1,
      "created_at": "2026-09-28T22:00:00Z"
    }
  ],
  "extracted_text_preview": "--- Page 1 ---\nLife Insurance Corporation of India (LIC)\nPolicy Schedule: POL-99881122..."
}
```

## Phase 6 & 7 Endpoints (Vector Embeddings, Hybrid Search & Entity Lookup)

### `POST /api/v1/search`
- **Auth**: `Bearer <token>`
- **Description**: Standard keyword search using BM25 with exact matching & frequency ranking.
- **Request Body**:
```json
{
  "query": "HDFC0001234 account statement",
  "category": "Banking",
  "top_k": 5
}
```
- **Response**: `200 OK`
```json
{
  "query": "HDFC0001234 account statement",
  "method": "keyword",
  "total_results": 1,
  "results": [
    {
      "document_id": 1,
      "filename": "bank_statement.pdf",
      "chunk_id": "1_bm25_p0",
      "page_number": 1,
      "category": "Banking",
      "text": "HDFC Bank Statement A/C: 1234567890 IFSC: HDFC0001234",
      "score": 0.85,
      "method": "bm25"
    }
  ]
}
```

### `POST /api/v1/semantic-search`
- **Auth**: `Bearer <token>`
- **Description**: Dense vector search using `all-MiniLM-L6-v2` embeddings in ChromaDB with strict per-user multi-tenant metadata filtering.
- **Request Body**:
```json
{
  "query": "health hospitalization insurance coverage",
  "category": "Insurance",
  "top_k": 5
}
```
- **Response**: `200 OK`
```json
{
  "query": "health hospitalization insurance coverage",
  "method": "semantic",
  "total_results": 1,
  "results": [
    {
      "document_id": 2,
      "filename": "mediclaim.pdf",
      "chunk_id": "2_p1_c0",
      "page_number": 1,
      "category": "Insurance",
      "text": "Medical reimbursement coverage details for in-patient hospital admission.",
      "score": 0.82,
      "method": "vector"
    }
  ]
}
```

### `POST /api/v1/hybrid-search`
- **Auth**: `Bearer <token>`
- **Description**: Combines Dense Vector + BM25 with Reciprocal Rank Fusion (RRF, $k=60$) and Cross-Encoder neural reranking.
- **Request Body**:
```json
{
  "query": "income tax acknowledgment refund",
  "category": null,
  "top_k": 5,
  "use_reranker": true,
  "rrf_k": 60
}
```
- **Response**: `200 OK`
```json
{
  "query": "income tax acknowledgment refund",
  "method": "hybrid",
  "total_results": 2,
  "results": [ ... ]
}
```

### `POST /api/v1/entity-search`
- **Auth**: `Bearer <token>`
- **Description**: Structured entity query across extracted metadata and structured Indian IDs.
- **Request Body**:
```json
{
  "entity_type": "PAN",
  "value": "ABCDE1234F"
}
```
- **Response**: `200 OK`
```json
{
  "query": "PAN: ABCDE1234F",
  "method": "entity",
  "total_results": 1,
  "results": [
    {
      "document_id": 3,
      "filename": "pan_card.pdf",
      "chunk_id": "3_entity_1",
      "page_number": 1,
      "category": "Government ID",
      "text": "Document contains PAN: ABCDE1234F",
      "score": 1.0,
      "method": "entity"
    }
  ]
}
```

## Phase 8 & 9 Endpoints (Agentic RAG, Tool Calling & Memory)

### `POST /api/v1/query`
- **Auth**: `Bearer <token>`
- **Description**: Natural language question answering and agentic execution. Automatically plans and invokes tools (`compare_documents`, `get_expiring_documents`, `extract_timeline`, `search_by_entity`, `hybrid_search`, etc.), audit logs executions, and returns grounded answers with citations.
- **Request Body**:
```json
{
  "query": "Compare my two health insurance policies and tell me which expires first.",
  "session_id": "session-xyz-123",
  "category": "Insurance",
  "top_k": 5,
  "use_reranker": true
}
```
- **Response**: `200 OK`
```json
{
  "query": "Compare my two health insurance policies and tell me which expires first.",
  "answer": "### Comparison: hdfc_health_policy.pdf vs star_health_policy.pdf\n\n- **Sum Insured**: hdfc = Rs. 5,00,000, star = Rs. 10,00,000\n- **Premium**: hdfc = Rs. 15,000, star = Rs. 22,000\n\n### Upcoming Expirations:\n- **star_health_policy.pdf**: Expiry Date **20-Dec-2026** (Expires first)\n- **hdfc_health_policy.pdf**: Expiry Date **15-Jun-2027**",
  "session_id": "session-xyz-123",
  "sources": [
    {
      "document_id": 1,
      "filename": "hdfc_health_policy.pdf",
      "page_number": 1,
      "snippet": "HDFC ERGO Health Insurance...",
      "score": 0.88
    },
    {
      "document_id": 2,
      "filename": "star_health_policy.pdf",
      "page_number": 1,
      "snippet": "Star Health Comprehensive Insurance...",
      "score": 0.85
    }
  ],
  "tools_invoked": [
    "compare_documents",
    "get_expiring_documents"
  ],
  "confidence": 1.0,
  "model_used": "agentic-react-engine"
}
```

### `DELETE /api/v1/query/session/{session_id}`
- **Auth**: `Bearer <token>`
- **Description**: Clears dialogue history and conversational memory for a specific session.
- **Response**: `200 OK`
```json
{
  "session_id": "session-xyz-123",
  "cleared": true,
  "message": "Session memory cleared successfully."
}
```

## Phase 10 Endpoints (Document Intelligence Services)

### `POST /api/v1/summary`
- **Auth**: `Bearer <token>`
- **Description**: Summarizes a single document by ID or performs a cross-document category summary.
- **Request Body**:
```json
{
  "document_id": 1,
  "category": null
}
```
- **Response**: `200 OK`
```json
{
  "document_id": 1,
  "filename": "hdfc_health_2026.pdf",
  "category": "Insurance",
  "summary": "**hdfc_health_2026.pdf** (Insurance)\n- HDFC ERGO Health Insurance Policy\n- Sum Insured: Rs. 5,00,000...",
  "key_entities": {
    "Policy Number": "POL-HDFC-101",
    "Expiry Date": "15-Jun-2026"
  }
}
```

### `POST /api/v1/compare`
- **Auth**: `Bearer <token>`
- **Description**: Compares two documents or versions and returns structured diffs across key entities and clauses.
- **Request Body**:
```json
{
  "doc_id_1": 1,
  "doc_id_2": 2
}
```
- **Response**: `200 OK`
```json
{
  "document_1": {"id": 1, "filename": "hdfc_health_2026.pdf", "category": "Insurance"},
  "document_2": {"id": 2, "filename": "hdfc_health_2027.pdf", "category": "Insurance"},
  "differences": [
    {
      "entity": "Sum Insured",
      "hdfc_health_2026.pdf (ID 1)": "Rs. 5,00,000",
      "hdfc_health_2027.pdf (ID 2)": "Rs. 10,00,000"
    }
  ],
  "identical_entities": [
    {"entity": "Policy Number", "value": "POL-HDFC-101"}
  ],
  "diff_summary": "Compared hdfc_health_2026.pdf with hdfc_health_2027.pdf: 1 differences found."
}
```

### `POST /api/v1/timeline`
- **Auth**: `Bearer <token>`
- **Description**: Aggregates and chronologically sorts all date and expiry events across user documents.
- **Request Body**:
```json
{
  "category": null,
  "sort_order": "asc"
}
```
- **Response**: `200 OK`
```json
{
  "total_events": 2,
  "events": [
    {
      "entity_id": 1,
      "date_string": "15-Jun-2026",
      "event_type": "Expiry Date",
      "document_id": 1,
      "filename": "hdfc_health_2026.pdf",
      "page": 1
    }
  ]
}
```

### `POST /api/v1/duplicates`
- **Auth**: `Bearer <token>`
- **Description**: Detects exact SHA-256 hash duplicates and semantic near-duplicates.
- **Request Body**:
```json
{
  "threshold": 0.88
}
```
- **Response**: `200 OK`
```json
{
  "total_documents_scanned": 3,
  "exact_duplicate_groups": 0,
  "near_duplicate_pairs": 1,
  "near_duplicates": [
    {
      "document_1": {"id": 1, "filename": "hdfc_health_2026.pdf"},
      "document_2": {"id": 2, "filename": "hdfc_health_2027.pdf"},
      "similarity_score": 0.94
    }
  ]
}
```

### `POST /api/v1/insights`
- **Auth**: `Bearer <token>`
- **Description**: Generates portfolio statistics, category breakdowns, entity analytics, and expiry counts.
- **Response**: `200 OK`
```json
{
  "total_documents": 3,
  "total_pages": 3,
  "total_storage_bytes": 1048576,
  "total_storage_mb": 1.0,
  "category_distribution": {"Insurance": 2, "Banking": 1},
  "status_distribution": {"ready": 3},
  "entity_counts": {"Policy Number": 2, "Expiry Date": 2},
  "expiring_documents_count": 2,
  "expiring_documents": [ ... ]
}
```

## Phase 11 Endpoints (Knowledge Graph & Proactive Intelligence)

### `GET /api/v1/graph`
- **Auth**: `Bearer <token>`
- **Description**: Returns knowledge graph nodes (Person, Policy, BankAccount, Property, TaxIdentifier, Organization, Document) and edges (owns, has_policy, has_account, contains, issued_by, located_at), scoped to the authenticated user.
- **Query Parameters**:
  - `entity_type` *(optional)*: Filter by entity type (e.g. `Person`, `Policy`, `BankAccount`, `TaxIdentifier`, `Property`)
  - `query` *(optional)*: Match search string in node label or value
  - `document_id` *(optional)*: Return 1-hop subgraph centered on a specific document
- **Response**: `200 OK`
```json
{
  "total_nodes": 4,
  "total_edges": 3,
  "nodes": [
    {
      "id": "doc_1",
      "label": "star_health_policy.pdf",
      "node_type": "Document",
      "properties": {"category": "Insurance", "page_count": 2}
    },
    {
      "id": "person_rahul sharma",
      "label": "Rahul Sharma",
      "node_type": "Person",
      "properties": {"value": "Rahul Sharma"}
    },
    {
      "id": "policy_POL-998877",
      "label": "Policy: POL-998877",
      "node_type": "Policy",
      "properties": {"value": "POL-998877"}
    }
  ],
  "edges": [
    {
      "source": "person_rahul sharma",
      "target": "doc_1",
      "relation": "owns",
      "properties": {}
    },
    {
      "source": "doc_1",
      "target": "policy_POL-998877",
      "relation": "has_policy",
      "properties": {}
    }
  ]
}
```

### `GET /api/v1/alerts`
- **Auth**: `Bearer <token>`
- **Description**: Generates prioritized proactive alerts across 4 intelligence dimensions:
  1. `expiry`: Upcoming policy / passport / warranty expirations within warning window (default 60 days) or critical past expirations.
  2. `missing_document`: Heuristic detection of missing essential documents (e.g. Property without Insurance, Tax/Banking without PAN/ID).
  3. `duplicate`: Exact hash matches and high semantic similarity near-duplicate pairs.
  4. `version_modified`: Document revision tracking.
- **Query Parameters**:
  - `warning_days` *(optional, default=60)*: Lookahead window in days for expirations.
  - `alert_type` *(optional)*: Filter by `expiry`, `missing_document`, `duplicate`, or `version_modified`.
- **Response**: `200 OK`
```json
{
  "total_alerts": 3,
  "critical_count": 1,
  "warning_count": 1,
  "info_count": 1,
  "alerts": [
    {
      "alert_type": "expiry",
      "severity": "critical",
      "title": "Document Expired: expired_passport.pdf",
      "description": "Expired 256 days ago on 15-Jan-2026 (expiry_date: 15-Jan-2026).",
      "document_id": 1,
      "filename": "expired_passport.pdf",
      "category": "Government ID",
      "expiry_date": "2026-01-15",
      "days_remaining": -256
    },
    {
      "alert_type": "expiry",
      "severity": "warning",
      "title": "Upcoming Expiry: motor_insurance.pdf",
      "description": "Expires in 20 days on 19-Oct-2026 (policy_expiry: 19-Oct-2026).",
      "document_id": 2,
      "filename": "motor_insurance.pdf",
      "category": "Insurance",
      "expiry_date": "2026-10-19",
      "days_remaining": 20
    },
    {
      "alert_type": "missing_document",
      "severity": "warning",
      "title": "Missing Property Insurance",
      "description": "You have registered Property deeds/khata records, but no active Home/Property Insurance policy was detected.",
      "category": "Property"
    }
  ]
}
```

---

## Phase 12 Endpoints (Audit & MCP)

### `GET /api/v1/audit`
- **Auth**: Admin only (JWT + `admin` role required)
- **Query Parameters**:
  - `page` (int, default 1): Page number
  - `page_size` (int, default 50, max 200): Entries per page
  - `action` (string, optional): Filter by action name (e.g., `upload`, `query`, `mcp:search_documents`)
  - `user_id` (int, optional): Filter by specific user ID
  - `resource_type` (string, optional): Filter by resource type
- **Response**: `200 OK`
```json
{
  "total": 127,
  "page": 1,
  "page_size": 50,
  "entries": [
    {
      "id": 127,
      "user_id": 1,
      "user_email": "admin@example.com",
      "action": "mcp:search_documents",
      "resource_type": "mcp_tool",
      "resource_id": null,
      "details": {"params": {"query": "insurance policy"}},
      "ip_address": null,
      "timestamp": "2026-10-03T14:00:00Z"
    }
  ]
}
```
- **Error Responses**:
  - `401 Unauthorized` — No/invalid JWT
  - `403 Forbidden` — Non-admin role

---

## Complete API Surface

| Method | Endpoint | Phase | Auth |
|--------|----------|-------|------|
| GET | `/health` | 1 | None |
| POST | `/api/v1/auth/register` | 2 | None |
| POST | `/api/v1/auth/login` | 2 | None |
| POST | `/api/v1/documents/upload` | 3 | User+ |
| GET | `/api/v1/documents` | 3 | User+ |
| GET | `/api/v1/documents/{id}` | 3 | User+ |
| POST | `/api/v1/query` | 8/9 | User+ |
| POST | `/api/v1/search` | 7 | User+ |
| POST | `/api/v1/semantic-search` | 7 | User+ |
| POST | `/api/v1/hybrid-search` | 7 | User+ |
| POST | `/api/v1/entity-search` | 7 | User+ |
| POST | `/api/v1/summary` | 10 | User+ |
| POST | `/api/v1/compare` | 10 | User+ |
| POST | `/api/v1/timeline` | 10 | User+ |
| POST | `/api/v1/duplicates` | 10 | User+ |
| POST | `/api/v1/insights` | 10 | User+ |
| GET | `/api/v1/alerts` | 11 | User+ |
| GET | `/api/v1/graph` | 11 | User+ |
| GET | `/api/v1/audit` | 12 | Admin |

