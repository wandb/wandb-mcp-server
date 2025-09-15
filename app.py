#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

Simplified approach for HF Spaces deployment.
"""

import os
import sys
import logging
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
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("huggingface-spaces-app")

# Read the index.html file content
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
with open(INDEX_HTML_PATH, "r") as f:
    INDEX_HTML_CONTENT = f.read()

# Create FastAPI app
app = FastAPI(
    title="Weights & Biases MCP Server",
    description="Model Context Protocol server for querying W&B data",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Import and setup MCP functionality after basic app setup
try:
    from wandb_mcp_server.server import (
        validate_and_get_api_key,
        setup_wandb_login,
        configure_wandb_logging,
        initialize_weave_tracing,
        create_mcp_server,
        ServerMCPArgs
    )
    
    # Setup W&B on import
    logger.info("Initializing W&B configuration...")
    configure_wandb_logging()
    
    args = ServerMCPArgs(
        transport="http",
        host="0.0.0.0",
        port=7860,
        wandb_api_key=os.environ.get("WANDB_API_KEY")
    )
    
    try:
        api_key = validate_and_get_api_key(args)
        setup_wandb_login(api_key)
        logger.info("W&B API configured successfully")
    except ValueError as e:
        logger.warning(f"W&B API key not configured: {e}")
        logger.warning("Server will start but W&B operations will fail")
    
    # Create MCP server instance
    mcp_server = create_mcp_server("http", "0.0.0.0", 7860)
    
    # Try to extract the FastAPI app from FastMCP and mount it
    if hasattr(mcp_server, 'app'):
        # Mount MCP app routes under /mcp
        from fastapi import APIRouter
        mcp_router = APIRouter(prefix="/mcp")
        
        # Copy routes from MCP app to our router
        if hasattr(mcp_server.app, 'routes'):
            for route in mcp_server.app.routes:
                if hasattr(route, 'endpoint') and hasattr(route, 'path'):
                    mcp_router.add_api_route(
                        path=route.path,
                        endpoint=route.endpoint,
                        methods=route.methods if hasattr(route, 'methods') else ["POST", "GET"],
                    )
        
        app.include_router(mcp_router)
        logger.info("MCP routes mounted at /mcp")
    else:
        # Fallback: Create simple MCP endpoint
        @app.post("/mcp")
        async def mcp_endpoint():
            return {"message": "MCP server is running", "status": "ready"}
        
        @app.get("/mcp")
        async def mcp_sse():
            from fastapi.responses import StreamingResponse
            import json
            
            async def event_stream():
                yield f"data: {json.dumps({'status': 'connected'})}\n\n"
            
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        logger.warning("Using fallback MCP endpoints")
    
except Exception as e:
    logger.error(f"Error setting up MCP server: {e}")
    
    # Provide fallback endpoints even if MCP setup fails
    @app.post("/mcp")
    async def mcp_fallback():
        return {"error": "MCP server initialization failed", "details": str(e)}
    
    @app.get("/mcp")
    async def mcp_sse_fallback():
        return {"error": "MCP SSE not available", "details": str(e)}

def main():
    """Main entry point for HuggingFace Spaces."""
    port = 7860
    host = "0.0.0.0"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("Landing page: /")
    logger.info("Health check: /health")
    logger.info("MCP endpoint: /mcp")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()