#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

This script runs the MCP server with streamable HTTP transport for HF Spaces.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

# Import and configure the MCP server components
from wandb_mcp_server.server import (
    validate_and_get_api_key, 
    setup_wandb_login,
    configure_wandb_logging,
    initialize_weave_tracing,
    register_tools,
    ServerMCPArgs
)
from wandb_mcp_server.utils import get_rich_logger

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_rich_logger("huggingface-spaces-app")

# Read the index.html file content
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
with open(INDEX_HTML_PATH, "r") as f:
    INDEX_HTML_CONTENT = f.read()

def main():
    """Main entry point for HuggingFace Spaces."""
    logger.info("Starting Weights & Biases MCP Server on HuggingFace Spaces")
    
    # Configure W&B logging behavior
    configure_wandb_logging()
    
    # Create minimal args for HF Spaces (always use HTTP transport)
    args = ServerMCPArgs(
        transport="http",
        host="0.0.0.0",
        port=7860,
        wandb_api_key=os.environ.get("WANDB_API_KEY")
    )
    
    # Validate and get API key
    try:
        api_key = validate_and_get_api_key(args)
        
        # Perform W&B login
        setup_wandb_login(api_key)
        
        # Initialize Weave tracing for MCP tool calls
        weave_initialized = initialize_weave_tracing()
        
        logger.info("W&B API configured successfully")
        logger.info(f"Weave tracing: {'Enabled' if weave_initialized else 'Disabled'}")
    except ValueError as e:
        logger.warning(f"W&B API key not configured: {e}")
        logger.warning("MCP server will start but operations will fail without a valid API key")
        logger.warning("Please set WANDB_API_KEY in the Space's environment variables")
    
    # Create the MCP server
    logger.info("Creating MCP server for HTTP transport")
    mcp = FastMCP(
        "wandb-mcp-server",
        host="0.0.0.0",
        port=7860,
        stateless_http=True
    )
    
    # Register all W&B tools
    register_tools(mcp)
    
    # Add custom routes to the MCP server
    # FastMCP creates routes using decorators, we can add our own
    @mcp.get("/")
    async def index():
        """Serve the landing page."""
        return HTMLResponse(content=INDEX_HTML_CONTENT)
    
    @mcp.get("/health")
    async def health():
        """Health check endpoint."""
        wandb_configured = bool(os.environ.get("WANDB_API_KEY"))
        return {
            "status": "healthy",
            "service": "wandb-mcp-server",
            "wandb_configured": wandb_configured
        }
    
    logger.info("Landing page available at: /")
    logger.info("Health check available at: /health")
    logger.info("MCP endpoint (Streamable HTTP) available at: /mcp")
    logger.info(f"Starting server on 0.0.0.0:7860")
    
    # Run the MCP server with streamable-http transport
    # This will start the uvicorn server internally
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()