# MCP Server — Personal Document Finder

## Overview

The MCP (Model Context Protocol) server exposes the same 11 document-intelligence tools available via the REST API over the MCP protocol. This allows external MCP-compatible AI clients (e.g., Claude Desktop, other LLM agents) to interact with your documents programmatically.

## Entry Point

```bash
python -m app.mcp.server
# or
python -m app.mcp
```

The server communicates over **stdio** (stdin/stdout), following the MCP specification.

## Authentication

Every MCP tool call must include authentication in the `_meta` field:

```json
{
  "tool_name": "search_documents",
  "arguments": {
    "query": "health insurance",
    "_meta": {
      "jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
  }
}
```

Alternatively, for trusted internal callers:
```json
{
  "_meta": { "user_id": 1 }
}
```

## Available Tools (11)

| Tool | Description | Required Parameters |
|------|-------------|---------------------|
| `search_documents` | BM25 keyword search | `query` |
| `semantic_search` | Dense vector semantic search | `query` |
| `hybrid_search` | BM25 + vector with RRF | `query` |
| `get_document` | Fetch document info and text | `document_id` |
| `get_document_metadata` | Get entities and metadata | `document_id` |
| `search_by_entity` | Search by entity type/value | `entity_type` or `value` |
| `summarize_document` | Generate document summary | `document_id` |
| `compare_documents` | Structured diff of two docs | `doc_id_1`, `doc_id_2` |
| `extract_timeline` | Chronological date events | *(none)* |
| `find_duplicates` | Detect duplicate documents | *(none)* |
| `get_expiring_documents` | Upcoming expirations | *(optional: `days`)* |

## Available Resources

### Static Resources
| URI | Description |
|-----|-------------|
| `documents://list` | List all user documents |

### Resource Templates
| URI Pattern | Description |
|-------------|-------------|
| `documents://{document_id}` | Document detail |
| `documents://{document_id}/entities` | Extracted entities |

## User Isolation

All tool calls and resource reads enforce per-user data isolation:
- The JWT token is decoded server-side to resolve the user
- All database and vector store queries are scoped to that user's documents
- Cross-user data access is prevented at the query layer

## Audit Logging

Every MCP tool invocation is recorded in the `audit_logs` table with:
- User ID
- Action name (prefixed with `mcp:`)
- Parameters (truncated for storage)
- Timestamp

## Client Configuration

### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "personal-document-finder": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/document-assistant"
    }
  }
}
```

### Generic MCP Client
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["-m", "app.mcp.server"],
    cwd="/path/to/document-assistant"
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("search_documents", {
            "query": "insurance policy",
            "_meta": {"jwt_token": "your-jwt-token"}
        })
```
