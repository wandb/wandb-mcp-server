"""Proxy search against the official W&B docs MCP at docs.wandb.ai/mcp.

When enabled (default), this tool lets the agent search W&B documentation
from within a single MCP connection. Disable via WANDB_MCP_PROXY_DOCS=false
if the docs MCP is connected separately.
"""

from __future__ import annotations

import json
import os

import httpx

from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DOCS_MCP_URL = "https://docs.wandb.ai/mcp"
DOCS_SEARCH_TIMEOUT = 30

SEARCH_WANDB_DOCS_TOOL_DESCRIPTION = """Search official W&B documentation for API usage, code examples, and guides.

<when_to_use>
Call when you need to know HOW to use a W&B/Weave feature or API. Searches docs.wandb.ai.
</when_to_use>

Parameters
----------
query : str
    Natural language search query about W&B documentation.

Returns
-------
str
    Relevant documentation snippets matching the query.
"""


def is_docs_proxy_enabled() -> bool:
    """Check if the docs proxy is enabled via environment variable."""
    return os.environ.get("WANDB_MCP_PROXY_DOCS", "true").lower() != "false"


async def search_wandb_docs(query: str) -> str:
    """Proxy a search request to the W&B docs MCP server."""
    try:
        log_tool_call("search_wandb_docs", "n/a", {"query": query})
    except Exception:
        logger.debug("analytics emit failed", exc_info=True)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                DOCS_MCP_URL,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "search_weights_biases_documentation",
                        "arguments": {"query": query},
                    },
                    "id": 1,
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=DOCS_SEARCH_TIMEOUT,
            )
            resp.raise_for_status()

            # The docs MCP returns SSE format: "event: message\ndata: {...}\n"
            body = resp.text
            result = None
            for line in body.split("\n"):
                if line.startswith("data: "):
                    try:
                        result = json.loads(line[6:])
                        break
                    except json.JSONDecodeError:
                        continue

            if not result:
                result = resp.json()

            if "error" in result:
                return json.dumps(
                    {"error": result["error"].get("message", "Unknown docs search error"), "query": query}
                )

            content_list = result.get("result", {}).get("content", [])
            if content_list and isinstance(content_list, list):
                texts = [item.get("text", "") for item in content_list if item.get("text")]
                if texts:
                    return "\n\n---\n\n".join(texts)

            return json.dumps({"error": "No documentation results found for this query.", "query": query})

    except httpx.TimeoutException:
        logger.warning(f"Docs MCP proxy timed out for query: {query}")
        return json.dumps(
            {
                "error": "Documentation search timed out. The docs server may be temporarily unavailable.",
                "query": query,
            }
        )
    except httpx.HTTPStatusError as e:
        logger.warning(f"Docs MCP proxy HTTP error: {e.response.status_code}")
        return json.dumps(
            {
                "error": f"Documentation search returned HTTP {e.response.status_code}.",
                "query": query,
            }
        )
    except Exception as e:
        logger.warning(f"Docs MCP proxy error: {e}")
        return json.dumps(
            {
                "error": f"Documentation search failed: {str(e)}",
                "query": query,
            }
        )
