"""
Weave MCP Server

A Model Context Protocol server for Weave traces.
"""

__version__ = "0.1.0"

# Import the functions we want to expose
from .server import cli
from .add_to_client import add_to_client_cli

# Import the raw HTTP-based implementation
from .mcp_tools.query_weave import query_paginated_weave_traces

# Define what gets imported with "from weave_mcp_server import *"
__all__ = [
    "cli",
    "query_paginated_weave_traces",
    "add_to_client_cli",
]
