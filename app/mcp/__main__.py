"""Allow running MCP server as:  python -m app.mcp"""

import asyncio
from app.mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())
