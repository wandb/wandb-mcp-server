#!/usr/bin/env python3
"""
Thread-safe entry point for the Weights & Biases MCP Server.

CLIENT-AUTHENTICATED MODE:
- No server-level API key is used
- All requests MUST include Bearer token authentication
- Each client provides their own W&B API key in the Authorization header
- Session IDs are used only as correlation IDs (stateless operation)
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
    configure_wandb_logging,
    register_tools,
    ServerMCPArgs
)
from wandb_mcp_server.utils import get_request_session, get_request_session_prefix, get_rich_logger

# Import the new API client manager and session manager
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.session_manager import get_session_manager

# Configure logging via shared utility (includes session prefix support)
logger = get_rich_logger(
    "wandb-mcp-server", default_level_str="INFO", env_var_name="MCP_SERVER_LOG_LEVEL"
)

# API key management is handled by WandBApiManager
# which provides thread-safe context storage per-request
# No server-level API keys are stored or used

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
    wandb_api_key=None  # Pure client-authentication mode
)

# Pure client-authenticated mode - no server-level API key
# All API keys MUST come from client Bearer tokens
logger.info("=" * 60)
logger.info("CLIENT-AUTHENTICATED MODE")
logger.info("No server API key configured")
logger.info("All requests MUST include Bearer token authentication")
logger.info("Get your API key at: https://wandb.ai/authorize")
logger.info("=" * 60)

# Create the MCP server in stateless mode
# All clients (OpenAI, Cursor, etc.) must provide Bearer token with each request
# Session IDs are used only as correlation IDs, no state is persisted
logger.info("Creating W&B MCP server in stateless HTTP mode...")
mcp = FastMCP("wandb-mcp-server", stateless_http=True)

# Register all W&B tools
# The tools will use WandBApiManager.get_api_key() to get the current request's API key
register_tools(mcp)

# Custom authentication middleware with enhanced multi-tenant isolation
async def thread_safe_auth_middleware(request: Request, call_next):
    """
    Enhanced authentication middleware with multi-tenant session isolation.
    
    Features:
    - Session-based API key isolation
    - Request tracking and auditing
    - Automatic cleanup of expired sessions
    - Validation to prevent cross-tenant leakage
    
    This provides strong isolation for multi-tenant environments where
    multiple concurrent requests use different W&B API keys.
    """
    # Only apply auth to MCP endpoints
    if not request.url.path.startswith("/mcp"):
        return await call_next(request)
    
    # Skip auth if explicitly disabled (development only)
    # WARNING: This is insecure and should only be used for local testing
    if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true":
        logger.warning("=" * 60)
        logger.warning("SECURITY WARNING: MCP authentication is DISABLED")
        logger.warning("Endpoints are publicly accessible without API keys")
        logger.warning("This should ONLY be used for local development/testing")
        logger.warning("=" * 60)
        return await call_next(request)
    
    session_manager = get_session_manager()
    request_id = f"req_{threading.current_thread().name}_{id(request)}"
    session_id = None
    api_key = None
    
    try:
        # Compute session prefix once (before state assignment)
        session_id = get_request_session(request)
        session_prefix = get_request_session_prefix(request)
        log = logging.LoggerAdapter(
            logger, {"session_id_prefix": f"[{session_prefix}] " if session_prefix else ""}
        )
        # Check if request has MCP session ID (prefer canonical casing)

        # Check for Bearer token
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            bearer_token = authorization[7:].strip()
            
            # Basic validation
            if len(bearer_token) < 20 or len(bearer_token) > 100:
                logger.warning(f"Invalid API key format from {request.client.host if request.client else 'unknown'}")
                return JSONResponse(
                    status_code=401,
                    content={"error": f"Invalid W&B API key format. Get your key at: https://wandb.ai/authorize"},
                    headers={"WWW-Authenticate": 'Bearer realm="W&B MCP", error="invalid_token"'}
                )
            
            api_key = bearer_token
        
        # Handle session cleanup
        if request.method == "DELETE" and session_id:
            log.debug(f"Session cleanup request for {session_id[:8]}...")
            session_manager.cleanup_session(session_id)
            return JSONResponse(status_code=204)  # No Content
        
        if api_key:
            # Create or validate session
            try:
                if session_id:
                    # Check if session exists
                    session = session_manager.get_session(session_id)
                    if not session:
                        # Session not found - return 404 per MCP spec
                        log.warning(f"Session {session_id[:8]}... not found - returning 404")
                        return JSONResponse(
                            status_code=404,
                            content={"error": "Session not found. Please reinitialize."},
                            headers={"WWW-Authenticate": 'Bearer realm="W&B MCP"'}
                        )
                    
                    # Validate existing session matches API key
                    if not session_manager.validate_session(session_id, api_key):
                        log.error(f"Session validation failed for {session_id[:8]}")
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Session validation failed - session unauthorized"},
                            headers={"WWW-Authenticate": 'Bearer realm="W&B MCP", error="invalid_session"'}
                        )
                else:
                    # For MCP compliance: Sessions should be created at initialization
                    # However, for stateless HTTP mode with multiple clients, we create on first auth
                    # This is a reasonable compromise for multi-tenant scenarios
                    session_id = session_manager.create_session(api_key)

                # Track request start
                session_manager.start_request(session_id, request_id)
                
            except ValueError as e:
                log.error(f"Session creation/validation error: {e}")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Session creation/validation failed"},
                    headers={"WWW-Authenticate": 'Bearer realm="W&B MCP"'}
                )
            
            # Set the API key in context variable (thread-safe)
            token = WandBApiManager.set_context_api_key(api_key)
            
            # Store session info in request state
            request.state.wandb_api_key = api_key
            request.state.session_id = session_id
            request.state.request_id = request_id
            # Refresh adapter with authoritative session prefix from state
            session_prefix = get_request_session_prefix(request)
            log = logging.LoggerAdapter(
                logger, {"session_id_prefix": f"[{session_prefix}] " if session_prefix else ""}
            )
            
            try:
                api = WandBApiManager.get_api()
                viewer = api.viewer
                log.info(f"Authenticated W&B viewer: {viewer}")
                request.state.viewer = str(viewer)
            except Exception as viewer_err:
                log.warning(f"Could not fetch W&B viewer: {viewer_err}")
            
            try:
                # Log request details for audit
                log.info(f"Processing request {request_id[:16]}... in session {session_id[:8]}...")
                
                # Process the request
                response = await call_next(request)
                
                # Add session ID to response headers for client tracking
                if session_id:
                    response.headers["Mcp-Session-Id"] = session_id
                    log.debug(f"Response includes session ID: {session_id[:8]}...")
                
                return response
                
            finally:
                # CRITICAL: Always cleanup context and mark request as ended
                try:
                    WandBApiManager.reset_context_api_key(token)
                    if session_id:
                        session_manager.end_request(session_id, request_id)
                except Exception as cleanup_error:
                    log.error(f"Error during cleanup for request {request_id[:16]}...: {cleanup_error}")
        else:
            # No API key provided - return 401 immediately to challenge the client
            log.warning(f"No Bearer token provided for {request.url.path} - returning 401")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authorization required",
                    "message": "Please provide your W&B API key as a Bearer token",
                    "instructions": "Get your API key at: https://wandb.ai/authorize"
                },
                headers={
                    "WWW-Authenticate": 'Bearer realm="W&B MCP", charset="UTF-8"'
                }
            )
        
    except Exception as e:
        log.error(f"Authentication error: {e}", exc_info=True)
        # Cleanup on error
        if session_id and request_id:
            try:
                session_manager.end_request(session_id, request_id)
            except:
                pass
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
    session_prefix = get_request_session_prefix(request)
    log = logging.LoggerAdapter(
        logger, {"session_id_prefix": f"[{session_prefix}] " if session_prefix else ""}
    )
    
    # Log request details
    log.info(f"Incoming request: {request.method} {request.url.path}")
    
    # Log MCP-specific headers
    mcp_session_id = request.headers.get("Mcp-Session-Id") or request.headers.get("mcp-session-id")
    if mcp_session_id:
        log.info(f"   MCP Session ID in request: {mcp_session_id[:8]}...")
    
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
                    log.info(f"   JSON-RPC request: method={method}, id={request_id}")
                    if method == "tools/call":
                        tool_name = body_json.get("params", {}).get("name", "unknown")
                        log.info(f"   Tool call request for: {tool_name}")
                except json.JSONDecodeError:
                    log.debug(f"   Request body (non-JSON): {body_bytes[:100]}")
                
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
                log.debug(f"   No request body")
        except Exception as e:
            log.debug(f"   Could not read request body: {e}")
    
    # Track if this is an MCP endpoint
    is_mcp = request.url.path.startswith("/mcp") or request.url.path == "/"
    
    try:
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log response details
        status_label = "SUCCESS" if response.status_code < 400 else "ERROR" if response.status_code >= 400 else "WARNING"
        log.info(f"[{status_label}] Response: {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
        
        # Log detailed info for 404s
        if response.status_code == 404:
            log.warning(f"404 Not Found for {request.url.path}")
            log.debug(f"   Full URL: {request.url}")
            log.debug(f"   Available routes: /, /health, /favicon.ico, /favicon.png, /mcp")
            if is_mcp:
                log.debug(f"   This appears to be an MCP endpoint that wasn't handled")
        
        return response
    except Exception as e:
        log.error(f"Error processing {request.method} {request.url.path}: {e}")
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
    """Health check endpoint with session manager statistics."""
    try:
        tools = await mcp.list_tools()
        tool_count = len(tools)
    except:
        tool_count = 0
    
    auth_status = "disabled" if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true" else "client-bearer-token"
    
    # Include worker information for debugging
    worker_info = {
        "pid": os.getpid(),
        "thread_id": threading.current_thread().name
    }
    
    # Get session manager statistics
    try:
        session_manager = get_session_manager()
        session_stats = session_manager.get_stats()
    except:
        session_stats = {"error": "Session manager not available"}
    
    return {
        "status": "healthy",
        "service": "wandb-mcp-server",
        "mode": "client-authenticated-multi-tenant",
        "server_api_key_configured": False,  # Always false in client-auth mode
        "tools_registered": tool_count,
        "authentication": auth_status,
        "session_management": session_stats,
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
