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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

# Import W&B setup functions
from wandb_mcp_server.server import (
    validate_and_get_api_key,
    setup_wandb_login,
    configure_wandb_logging,
    initialize_weave_tracing,
    register_tools,
    ServerMCPArgs
)

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
try:
    api_key = validate_and_get_api_key(args)
    setup_wandb_login(api_key)
    initialize_weave_tracing()
    wandb_configured = True
    logger.info("W&B API configured successfully")
except ValueError as e:
    logger.warning(f"W&B API key not configured: {e}")
    logger.warning("Server will start but W&B operations will fail")

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

# Add custom routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the landing page."""
    return INDEX_HTML_CONTENT

@app.get("/health")
async def health():
    """Health check endpoint."""
    # list_tools is async, so we need to handle it properly
    try:
        tools = await mcp.list_tools()
        tool_count = len(tools)
    except:
        tool_count = 0
    
    return {
        "status": "healthy",
        "service": "wandb-mcp-server",
        "wandb_configured": wandb_configured,
        "tools_registered": tool_count
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