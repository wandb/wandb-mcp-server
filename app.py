#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

This script creates a FastAPI application with a landing page and mounts
the MCP server from the existing server module.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the existing MCP server instance and utilities
from wandb_mcp_server.server import mcp
from wandb_mcp_server.utils import get_rich_logger

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_rich_logger("huggingface-spaces-app")

# Lifecycle manager for the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Weights & Biases MCP Server on HuggingFace Spaces")
    
    # Check for required environment variables
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("WANDB_API_KEY environment variable is not set!")
        logger.warning("Please set your Weights & Biases API key in the Space's environment variables.")
        logger.warning("You can get your API key from: https://wandb.ai/authorize")
    else:
        logger.info("WANDB_API_KEY configured: Yes")
    
    logger.info(f"Starting HTTP server on port {os.environ.get('PORT', '7860')}")
    logger.info("Landing page available at: /")
    logger.info("MCP endpoint available at: /mcp/sse")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Weights & Biases MCP Server")

# Create FastAPI app with lifecycle management
app = FastAPI(
    title="Weights & Biases MCP Server",
    description="Model Context Protocol server for querying W&B data",
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

# Mount static files (create directory if it doesn't exist)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the landing page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "wandb-mcp-server"}

# Mount the existing MCP server at /mcp path
# The MCP server from server.py already has all tools registered
mcp.run(
    transport="streamable-http",
    http_app=app,
    http_path="/mcp"
)

def main():
    """Main entry point for HuggingFace Spaces."""
    # Run the FastAPI app with Uvicorn
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()