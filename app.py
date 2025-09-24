#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

Using the correct FastMCP mounting pattern with streamable_http_app().
"""

import os
import sys
import logging
import contextlib
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure W&B directories for HF Spaces (must be done before importing wandb)
os.environ["WANDB_CACHE_DIR"] = "/tmp/.wandb_cache"
os.environ["WANDB_CONFIG_DIR"] = "/tmp/.wandb_config"
os.environ["WANDB_DATA_DIR"] = "/tmp/.wandb_data"
os.environ["HOME"] = "/tmp"
os.environ["WANDB_SILENT"] = "True"
os.environ["WEAVE_SILENT"] = "True"

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
import base64

# Import W&B setup functions
from wandb_mcp_server.server import (
    validate_and_get_api_key,
    setup_wandb_login,
    configure_wandb_logging,
    initialize_weave_tracing,
    register_tools,
    ServerMCPArgs
)

# Import authentication
from wandb_mcp_server.auth import mcp_auth_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wandb-mcp-server")

# Read the index.html file content
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
with open(INDEX_HTML_PATH, "r") as f:
    INDEX_HTML_CONTENT = f.read()

# W&B Logo Favicon - Exact copy from wandb.ai/site
# This is the official favicon PNG (32x32) used on https://wandb.ai
# Downloaded from: https://cdn.wandb.ai/production/ff061fe17/favicon.png
WANDB_FAVICON_BASE64 = """iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAUVBMVEUAAAD/zzD/zzD/zzD/zjH/yzD/zDP/zDP/zTL/zDP/zTL/yzL/yzL/zDL/zDL/zDP/zDP/zDP/zDP/yzL/yzP/zDL/zDL/zDL/zDL/zDP/zDNs+ITNAAAAGnRSTlMAECAwP0BQX2BvcICPkJ+gr7C/wM/Q3+Dv8ORN9PUAAAEOSURBVBgZfcEJkpswAADBEVphB0EwzmJg/v/QcKbKC3E3FI/xN5fa8VEAjRq5ENUGaNXIhai2QBrsOJTf3yWHziHxw6AvPpl04pOsmXehfvksOYTAoXz6qgONi8hJdNEwuMicZBcvXGVOsit6FxWboq4LNpWLntLZFNj0+s0mTM5KSLmpAjtn7ELV5MQPnXZ8VJacxFvgUrhFZnc1cCGod6BTE7t7Xd/YJbUDKjWw6Zw92AS1AsK9SWyiq4JNau6BN8lV4n+Sq8Sb8PXri93gbOBNGtUnm6Kbpq7gUDDrXFRc6B0TuMqcJbWFyUXmLKoNtC4SmzyOmUMztAUUf9TMbtKRk8g/gw58UvZ9yZu/MeoYEFwSwuAAAAAASUVORK5CYII=""".strip()

# Use the official favicon directly
FAVICON_BASE64 = WANDB_FAVICON_BASE64

# Initialize W&B
logger.info("Initializing W&B configuration...")
configure_wandb_logging()

args = ServerMCPArgs(
    transport="http",
    host="0.0.0.0",
    port=7860,
    wandb_api_key=os.environ.get("WANDB_API_KEY")
)

wandb_configured = False
api_key = validate_and_get_api_key(args)
if api_key:
    try:
        setup_wandb_login(api_key)
        initialize_weave_tracing()
        wandb_configured = True
        logger.info("Server W&B API key configured successfully")
    except Exception as e:
        logger.warning(f"Failed to configure server W&B API key: {e}")
else:
    logger.info("No server W&B API key configured - clients will provide their own")

# Create the MCP server
logger.info("Creating W&B MCP server...")
mcp = FastMCP("wandb-mcp-server")

# Register all W&B tools
register_tools(mcp)

# Create lifespan context manager for session management
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP session lifecycle."""
    async with mcp.session_manager.run():
        logger.info("MCP session manager started")
        yield
        logger.info("MCP session manager stopped")

# Create the main FastAPI app with lifespan
app = FastAPI(
    title="Weights & Biases MCP Server",
    description="Model Context Protocol server for W&B",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware for MCP endpoints
@app.middleware("http")
async def auth_middleware(request, call_next):
    """Add OAuth 2.1 Bearer token authentication for MCP endpoints."""
    return await mcp_auth_middleware(request, call_next)

# Add custom routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the landing page."""
    return INDEX_HTML_CONTENT

@app.get("/favicon.ico")
async def favicon():
    """Serve the official W&B logo favicon (exact copy from wandb.ai)."""
    return Response(
        content=base64.b64decode(FAVICON_BASE64),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
            "Content-Type": "image/x-icon"  # Standard favicon content type
        }
    )

# Removed OAuth endpoints - only API key authentication is supported
# See AUTH_README.md for details on why full OAuth isn't feasible

@app.get("/health")
async def health():
    """Health check endpoint."""
    # list_tools is async, so we need to handle it properly
    try:
        tools = await mcp.list_tools()
        tool_count = len(tools)
    except:
        tool_count = 0
    
    auth_status = "disabled" if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true" else "enabled"
    
    return {
        "status": "healthy",
        "service": "wandb-mcp-server",
        "wandb_configured": wandb_configured,
        "tools_registered": tool_count,
        "authentication": auth_status
    }

# Mount the MCP streamable HTTP app
# Note: streamable_http_app() creates internal routes at /mcp
# So we mount at root to avoid /mcp/mcp double path
mcp_app = mcp.streamable_http_app()
logger.info("Mounting MCP streamable HTTP app")
# Mount at root, so MCP endpoint will be at /mcp (not /mcp/mcp)
app.mount("/", mcp_app)

# Port for HF Spaces
PORT = int(os.environ.get("PORT", "7860"))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    logger.info("Landing page: /")
    logger.info("Health check: /health")
    logger.info("MCP endpoint: /mcp")
    uvicorn.run(app, host="0.0.0.0", port=PORT)