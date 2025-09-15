#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

This script creates a FastAPI application with a landing page and mounts
the MCP server using the Streamable HTTP transport at /mcp for Hugging Face Spaces deployment.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import and configure the MCP server
from wandb_mcp_server.server import (
    validate_and_get_api_key, 
    setup_wandb_login,
    configure_wandb_logging,
    initialize_weave_tracing,
    create_mcp_server,
    ServerMCPArgs
)
from wandb_mcp_server.utils import get_rich_logger

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_rich_logger("huggingface-spaces-app")

# Create the main FastAPI app
app = FastAPI(
    title="Weights & Biases MCP Server",
    description="Model Context Protocol server for querying W&B data on Hugging Face Spaces",
    version="0.1.0"
)

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read the index.html file content
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
with open(INDEX_HTML_PATH, "r") as f:
    INDEX_HTML_CONTENT = f.read()

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the landing page."""
    return INDEX_HTML_CONTENT

@app.get("/health")
async def health():
    """Health check endpoint."""
    wandb_configured = bool(os.environ.get("WANDB_API_KEY"))
    return {
        "status": "healthy",
        "service": "wandb-mcp-server",
        "wandb_configured": wandb_configured
    }

# Initialize W&B and MCP server on app startup
@app.on_event("startup")
async def startup_event():
    """Initialize W&B and MCP server on startup."""
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
    
    # Create and mount the MCP server
    logger.info("Creating MCP server for HTTP transport")
    mcp_server = create_mcp_server("http", "0.0.0.0", 7860)
    
    # Mount the MCP server to the /mcp path
    mcp_server.run(
        transport="streamable-http",
        http_app=app,
        http_path="/mcp"
    )
    
    logger.info("MCP server mounted at /mcp")
    logger.info("Landing page available at: /")
    logger.info("MCP endpoint (Streamable HTTP) available at: /mcp")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown."""
    logger.info("Shutting down Weights & Biases MCP Server")

def main():
    """Main entry point for HuggingFace Spaces."""
    # Force specific settings for HF Spaces
    port = 7860  # HF Spaces expects port 7860
    host = "0.0.0.0"  # HF Spaces requires binding to all interfaces
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=False  # Disable reload in production
    )

if __name__ == "__main__":
    main()