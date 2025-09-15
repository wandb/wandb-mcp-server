#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

This implements MCP streamable HTTP transport directly in FastAPI.
"""

import os
import sys
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure W&B directories for HF Spaces (must be done before importing wandb)
os.environ["WANDB_CACHE_DIR"] = "/tmp/.wandb_cache"
os.environ["WANDB_CONFIG_DIR"] = "/tmp/.wandb_config"
os.environ["WANDB_DATA_DIR"] = "/tmp/.wandb_data"
os.environ["HOME"] = "/tmp"
os.environ["WANDB_SILENT"] = "True"
os.environ["WEAVE_SILENT"] = "True"

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

# MCP server state
mcp_initialized = False
mcp_tools = {}
wandb_configured = False

@app.on_event("startup")
async def startup_event():
    """Initialize W&B and MCP tools on startup."""
    global mcp_initialized, mcp_tools, wandb_configured
    
    logger.info("Starting Weights & Biases MCP Server on HuggingFace Spaces")
    
    try:
        # Import W&B components
        from wandb_mcp_server.server import (
            validate_and_get_api_key,
            setup_wandb_login,
            configure_wandb_logging,
            initialize_weave_tracing,
            ServerMCPArgs
        )
        
        # Configure W&B
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
            initialize_weave_tracing()
            wandb_configured = True
            logger.info("W&B API configured successfully")
        except ValueError as e:
            logger.warning(f"W&B API key not configured: {e}")
            logger.warning("Server will start but W&B operations will fail")
        
        # Import MCP tools
        from wandb_mcp_server.mcp_tools.query_weave import (
            QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
            query_paginated_weave_traces
        )
        from wandb_mcp_server.mcp_tools.count_traces import (
            COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
            count_traces
        )
        from wandb_mcp_server.mcp_tools.query_wandb_gql import (
            QUERY_WANDB_GQL_TOOL_DESCRIPTION,
            query_paginated_wandb_gql
        )
        from wandb_mcp_server.mcp_tools.create_report import (
            CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
            create_report
        )
        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import (
            LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
            list_entity_projects
        )
        from wandb_mcp_server.mcp_tools.query_wandbot import (
            WANDBOT_TOOL_DESCRIPTION,
            query_wandbot_api
        )
        
        # Register tools with their descriptions
        mcp_tools = {
            "query_weave_traces_tool": {
                "function": query_paginated_weave_traces,
                "description": QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
                "async": True
            },
            "count_weave_traces_tool": {
                "function": count_traces,
                "description": COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
                "async": False
            },
            "query_wandb_tool": {
                "function": query_paginated_wandb_gql,
                "description": QUERY_WANDB_GQL_TOOL_DESCRIPTION,
                "async": False
            },
            "create_wandb_report_tool": {
                "function": create_report,
                "description": CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
                "async": False
            },
            "query_wandb_entity_projects": {
                "function": list_entity_projects,
                "description": LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
                "async": False
            },
            "query_wandb_support_bot": {
                "function": query_wandbot_api,
                "description": WANDBOT_TOOL_DESCRIPTION,
                "async": False
            }
        }
        
        mcp_initialized = True
        logger.info(f"MCP server initialized with {len(mcp_tools)} tools")
        
    except Exception as e:
        logger.error(f"Error initializing MCP server: {e}")
        mcp_initialized = False

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the landing page."""
    return INDEX_HTML_CONTENT

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "wandb-mcp-server",
        "wandb_configured": wandb_configured,
        "mcp_initialized": mcp_initialized,
        "tools_count": len(mcp_tools)
    }

# MCP Streamable HTTP Implementation
@app.post("/mcp")
async def handle_mcp_post(request: Request):
    """
    Handle MCP POST requests following the streamable HTTP transport protocol.
    """
    if not mcp_initialized:
        return JSONResponse(
            status_code=503,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "MCP server not initialized"
                }
            }
        )
    
    try:
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        request_id = body.get("id")
        
        # Handle different MCP methods
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "prompts": {"listChanged": False},
                        "resources": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "wandb-mcp-server",
                        "version": "0.1.0"
                    }
                }
            }
        
        elif method == "tools/list":
            tools_list = []
            for tool_name, tool_info in mcp_tools.items():
                # Extract a shorter description (first line)
                desc_lines = tool_info["description"].split('\n')
                short_desc = desc_lines[0] if desc_lines else f"W&B tool: {tool_name}"
                
                tools_list.append({
                    "name": tool_name,
                    "description": short_desc,
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                })
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools_list}
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            if tool_name not in mcp_tools:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    }
                }
            
            try:
                tool_info = mcp_tools[tool_name]
                tool_function = tool_info["function"]
                
                # Execute the tool
                if tool_info["async"]:
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)
                
                # Format the result
                if isinstance(result, str):
                    content_text = result
                elif isinstance(result, dict):
                    content_text = json.dumps(result, indent=2)
                else:
                    content_text = str(result)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content_text
                            }
                        ]
                    }
                }
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Tool execution error: {str(e)}"
                    }
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            
    except Exception as e:
        logger.error(f"Error handling MCP request: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
        )

@app.get("/mcp")
async def handle_mcp_sse(request: Request):
    """
    Handle MCP GET requests for SSE (Server-Sent Events) streaming.
    This enables server-initiated messages and long-lived connections.
    """
    if not mcp_initialized:
        return JSONResponse(
            status_code=503,
            content={"error": "MCP server not initialized"}
        )
    
    async def event_stream():
        """Generate server-sent events for MCP."""
        try:
            # Send initial connection confirmation
            yield f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'connection/ready', 'params': {'status': 'connected', 'timestamp': datetime.utcnow().isoformat()}})}\n\n"
            
            # Keep connection alive with periodic heartbeats
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'ping', 'params': {'timestamp': datetime.utcnow().isoformat()}})}\n\n"
                
        except asyncio.CancelledError:
            logger.info("SSE connection closed")
            raise
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
            yield f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'error', 'params': {'message': str(e)}})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )

# Additional MCP endpoints for better compatibility
@app.options("/mcp")
async def handle_mcp_options():
    """Handle OPTIONS requests for CORS preflight."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Max-Age": "3600"
        }
    )

def main():
    """Main entry point for HuggingFace Spaces."""
    port = 7860
    host = "0.0.0.0"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("Landing page: /")
    logger.info("Health check: /health")
    logger.info("MCP endpoint (Streamable HTTP): /mcp")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()