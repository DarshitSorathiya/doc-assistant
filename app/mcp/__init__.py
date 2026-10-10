"""MCP server — exposes tools/resources over the MCP protocol (Phase 12).

Usage:
    python -m app.mcp.server
    python -m app.mcp
"""

from app.mcp.tools import MCP_TOOL_MAP, call_mcp_tool
from app.mcp.resources import get_resource_handlers
from app.mcp.server import create_mcp_server
