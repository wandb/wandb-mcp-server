#!/usr/bin/env python3
"""
Thread-safe entry point for the Weights & Biases MCP Server.
"""

import os
import sys
import logging
import contextlib
from pathlib import Path
import threading
import wandb

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
    validate_api_key,
    configure_wandb_logging,
    initialize_weave_tracing,
    register_tools,
    ServerMCPArgs
)

# Import the new API client manager
from wandb_mcp_server.api_client import WandBApiManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wandb-mcp-server")

# API key management is now handled by WandBApiManager
# which provides thread-safe context storage

# Thread-local storage for W&B client instances
# This prevents recreating clients for each request
thread_local = threading.local()

def get_thread_local_wandb_client(api_key: str):
    """Get or create a thread-local W&B client for the given API key."""
    if not hasattr(thread_local, 'clients'):
        thread_local.clients = {}
    
    if api_key not in thread_local.clients:
        # Store the API key for this thread's client
        thread_local.clients[api_key] = {
            'api_key': api_key,
            'initialized': True
        }
    
    return thread_local.clients[api_key]

# Read the index.html file content
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
with open(INDEX_HTML_PATH, "r") as f:
    INDEX_HTML_CONTENT = f.read()

# W&B Logo Favicon
WANDB_FAVICON_BASE64 = """iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAUVBMVEUAAAD/zzD/zzD/zzD/zjH/yzD/zDP/zDP/zTL/zDP/zTL/yzL/yzL/zDL/zDL/zDP/zDP/zDP/zDP/yzL/yzP/zDL/zDL/zDL/zDL/zDP/zDNs+ITNAAAAGnRSTlMAECAwP0BQX2BvcICPkJ+gr7C/wM/Q3+Dv8ORN9PUAAAEOSURBVBgZfcEJkpswAADBEVphB0EwzmJg/v/QcKbKC3E3FI/xN5fa8VEAjRq5ENUGaNXIhai2QBrsOJTf3yWHziHxw6AvPpl04pOsmXehfvksOYTAoXz6qgONi8hJdNEwuMicZBcvXGVOsit6FxWboq4LNpWLntLZFNj0+s0mTM5KSLmpAjtn7ELV5MQPnXZ8VJacxFvgUrhFZnc1cCGod6BTE7t7Xd/YJbUDKjWw6Zw92AS1AsK9SWyiq4JNau6BN8lV4n+Sq8Sb8PXri93gbOBNGtUnm6Kbpq7gUDDrXFRc6B0TuMqcJbWFyUXmLKoNtC4SmzyOmUMztAUUf9TMbtKRk8g/gw58UvZ9yZu/MeoYEFwSwuAAAAAASUVORK5CYII=""".strip()

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
        validate_api_key(api_key)
        initialize_weave_tracing()
        wandb_configured = True
        logger.info("Server W&B API key configured successfully")
    except Exception as e:
        logger.warning(f"Failed to configure server W&B API key: {e}")
else:
    logger.info("No server W&B API key configured - clients will provide their own")

# Create the MCP server in stateless mode
# All clients (OpenAI, Cursor, etc.) must provide Bearer token with each request
# Session IDs are used only as correlation IDs, no state is persisted
logger.info("Creating W&B MCP server in stateless HTTP mode...")
mcp = FastMCP("wandb-mcp-server", stateless_http=True)

# Register all W&B tools
# The tools will use WandBApiManager.get_api_key() to get the current request's API key
register_tools(mcp)

# Custom authentication middleware
async def thread_safe_auth_middleware(request: Request, call_next):
    """
    Stateless authentication middleware for MCP endpoints.
    
    Pure stateless operation - every request must include authentication:
    - Session IDs are only used as correlation IDs
    - No session state is stored between requests
    - Each request must include Bearer token authentication
    
    This works with all clients (OpenAI, Cursor, etc.) that support MCP.
    """
    # Only apply auth to MCP endpoints
    if not request.url.path.startswith("/mcp"):
        return await call_next(request)
    
    # Skip auth if explicitly disabled (development only)
    if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true":
        logger.warning("MCP authentication is disabled - endpoints are publicly accessible")
        env_key = os.environ.get("WANDB_API_KEY")
        if env_key:
            token = WandBApiManager.set_context_api_key(env_key)
            try:
                response = await call_next(request)
                return response
            finally:
                WandBApiManager.reset_context_api_key(token)
        return await call_next(request)
    
    try:
        api_key = None
        
        # Check if request has MCP session ID (correlation ID only in stateless mode)
        session_id = request.headers.get("Mcp-Session-Id") or request.headers.get("mcp-session-id")
        if session_id:
            logger.debug(f"Request has correlation ID: {session_id[:8]}...")
        
        # Check for Bearer token (for new sessions or explicit auth)
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            bearer_token = authorization[7:].strip()
            
            # Basic validation
            if len(bearer_token) < 20 or len(bearer_token) > 100:
                return JSONResponse(
                    status_code=401,
                    content={"error": f"Invalid W&B API key format. Get your key at: https://wandb.ai/authorize"},
                    headers={"WWW-Authenticate": 'Bearer realm="W&B MCP", error="invalid_token"'}
                )
            
            # Use Bearer token
            api_key = bearer_token
            logger.info(f"Using Bearer token for authentication")
        
        # Handle session cleanup (stateless mode - just acknowledge and pass through)
        if request.method == "DELETE" and session_id:
            logger.debug(f"Session cleanup: DELETE for {session_id[:8]}... (stateless - no action needed)")
            return await call_next(request)
        
        if api_key:
            # Set the API key in context variable (thread-safe)
            token = WandBApiManager.set_context_api_key(api_key)
            
            # Also store in request state
            request.state.wandb_api_key = api_key
            
            try:
                # Process the request
                response = await call_next(request)
                
                # In stateless mode, we don't store any session state
                response_session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
                if response_session_id:
                    logger.debug(f"Response includes correlation ID: {response_session_id[:8]}...")
                
                return response
            finally:
                # Reset context variable
                WandBApiManager.reset_context_api_key(token)
        else:
            # No API key available - in stateless mode, this is expected to fail
            logger.warning(f"No Bearer token provided for {request.url.path}")
            logger.debug(f"   Request method: {request.method}")
            logger.debug("   Passing to MCP (will likely return 401)")
            return await call_next(request)
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication failed"},
            headers={"WWW-Authenticate": 'Bearer realm="W&B MCP"'}
        )

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
    description="Model Context Protocol server for W&B (Thread-Safe)",
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

# Add request logging middleware for debugging
@app.middleware("http")
async def logging_middleware(request, call_next):
    """Log all incoming requests for debugging."""
    import time
    start_time = time.time()
    
    # Log request details
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    # Log MCP-specific headers
    mcp_session_id = request.headers.get("mcp-session-id")
    if mcp_session_id:
        logger.info(f"   MCP Session ID in request: {mcp_session_id[:8]}...")
    
    # Try to log request body for POST requests
    if request.method == "POST" and request.url.path in ["/mcp", "/"]:
        try:
            # Clone the request body so we can read it
            body_bytes = await request.body()
            if body_bytes:
                import json
                try:
                    body_json = json.loads(body_bytes)
                    method = body_json.get("method", "unknown")
                    request_id = body_json.get("id", "unknown")
                    logger.info(f"   JSON-RPC request: method={method}, id={request_id}")
                    if method == "tools/call":
                        tool_name = body_json.get("params", {}).get("name", "unknown")
                        logger.info(f"   Tool call request for: {tool_name}")
                except json.JSONDecodeError:
                    logger.debug(f"   Request body (non-JSON): {body_bytes[:100]}")
                
                # Reconstruct the request with the body we read
                from starlette.datastructures import Headers
                from starlette.requests import Request as StarletteRequest
                
                # Create a new request with the body we read
                scope = request.scope
                scope["body"] = body_bytes
                
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                
                request = StarletteRequest(scope, receive)
            else:
                logger.debug("   No request body")
        except Exception as e:
            logger.debug(f"   Could not read request body: {e}")
    
    # Track if this is an MCP endpoint
    is_mcp = request.url.path.startswith("/mcp") or request.url.path == "/"
    
    try:
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log response details
        status_label = "SUCCESS" if response.status_code < 400 else "ERROR" if response.status_code >= 400 else "WARNING"
        logger.info(f"[{status_label}] Response: {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
        
        # Log detailed info for 404s
        if response.status_code == 404:
            logger.warning(f"404 Not Found for {request.url.path}")
            logger.debug(f"   Full URL: {request.url}")
            logger.debug(f"   Available routes: /, /health, /favicon.ico, /favicon.png, /mcp")
            if is_mcp:
                logger.debug("   This appears to be an MCP endpoint that wasn't handled")
        
        return response
    except Exception as e:
        logger.error(f"Error processing {request.method} {request.url.path}: {e}")
        raise

# Add authentication middleware
@app.middleware("http")
async def auth_middleware(request, call_next):
    """Add thread-safe OAuth 2.1 Bearer token authentication for MCP endpoints."""
    return await thread_safe_auth_middleware(request, call_next)

# Add custom routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the landing page."""
    return INDEX_HTML_CONTENT

@app.get("/favicon.ico")
async def favicon():
    """Serve the official W&B logo favicon."""
    return Response(
        content=base64.b64decode(FAVICON_BASE64),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000",
            "Content-Type": "image/png"
        }
    )

@app.get("/favicon.png")
async def favicon_png():
    """Alternative PNG favicon endpoint for better browser compatibility."""
    return Response(
        content=base64.b64decode(FAVICON_BASE64),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000",
            "Content-Type": "image/png"
        }
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        tools = await mcp.list_tools()
        tool_count = len(tools)
    except:
        tool_count = 0
    
    auth_status = "disabled" if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true" else "enabled"
    
    # Include worker information for debugging
    worker_info = {
        "pid": os.getpid(),
        "thread_id": threading.current_thread().name
    }
    
    return {
        "status": "healthy",
        "service": "wandb-mcp-server",
        "wandb_configured": wandb_configured,
        "tools_registered": tool_count,
        "authentication": auth_status,
        "worker_info": worker_info
    }

# Mount the MCP streamable HTTP app
# NOTE: MCP app is mounted at root "/" to handle all MCP protocol requests
# This means it will catch all unhandled routes, which is why we define our
# custom routes (/, /health, etc.) BEFORE mounting the MCP app
mcp_app = mcp.streamable_http_app()
logger.info("Mounting MCP streamable HTTP app at root /")
logger.info("Note: MCP will handle all unmatched routes, returning 404 for non-MCP requests")

# For debugging: Log incoming requests to understand routing
@app.middleware("http")
async def mcp_routing_debug(request, call_next):
    """Debug middleware to understand MCP routing issues."""
    path = request.url.path
    method = request.method
    
    # Check if this should be an MCP request
    is_mcp_request = (
        request.headers.get("Content-Type") == "application/json" and
        (request.headers.get("Accept", "").find("text/event-stream") >= 0 or
         request.headers.get("Accept", "").find("application/json") >= 0)
    )
    
    if path == "/" and method == "GET":
        logger.debug("Root GET request - should show landing page")
    elif path == "/health" and method == "GET":
        logger.debug("Health check request")
    elif path in ["/", "/mcp"] and is_mcp_request:
        logger.debug(f"MCP protocol request detected on {path}")
    elif path == "/" and method in ["POST", "GET"] and not is_mcp_request:
        logger.debug(f"Non-MCP {method} request to root - may get 404 from MCP app")
    
    return await call_next(request)

app.mount("/", mcp_app)

# Port for HF Spaces
PORT = int(os.environ.get("PORT", "7860"))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    logger.info("Landing page: /")
    logger.info("Health check: /health")
    logger.info("MCP endpoint: /mcp")
    
    # In stateless mode, we can scale horizontally with multiple workers
    # However, for HuggingFace Spaces we use single worker for simplicity
    logger.info("Starting server (stateless mode - supports horizontal scaling)")
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)  # Can increase workers if needed
