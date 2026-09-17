"""
Weave MCP Server

A Model Context Protocol server for Weave traces.
"""

__version__ = "0.4.0"

__all__ = [
    "cli",
    "query_paginated_weave_traces",
    "add_to_client_cli",
]


def __getattr__(name: str):
    """Preserve public imports without constructing the server on package import."""
    if name == "cli":
        from .entrypoint import cli

        return cli
    if name == "add_to_client_cli":
        from .add_to_client import add_to_client_cli

        return add_to_client_cli
    if name == "query_paginated_weave_traces":
        from .mcp_tools.query_weave import query_paginated_weave_traces

        return query_paginated_weave_traces
    raise AttributeError(name)
