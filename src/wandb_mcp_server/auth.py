"""
Authentication middleware for W&B MCP Server.

Implements Bearer token validation for HTTP transport as per 
MCP specification: https://modelcontextprotocol.io/specification/draft/basic/authorization

Clients send their W&B API keys as Bearer tokens, which the server
then uses for all W&B operations on behalf of that client.
"""

import os
import logging
import re
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Bearer token security scheme
bearer_scheme = HTTPBearer(auto_error=False)


class MCPAuthConfig:
    """
    Configuration for MCP authentication.
    
    For HTTP transport: Accepts any W&B API key as a Bearer token.
    The server uses the client's token for all W&B operations.
    """
    pass  # Simple config, no OAuth metadata needed


def is_valid_wandb_api_key(token: str) -> bool:
    """
    Check if a token looks like a valid W&B API key format.
    W&B API keys are typically 40 characters but we'll be permissive.
    """
    if not token:
        return False
    
    # Strip any whitespace that might have been included
    token = token.strip()
    
    # Be permissive - accept keys between 20 and 100 characters
    # The actual W&B API will validate the exact format
    if len(token) < 20 or len(token) > 100:
        return False
    
    # Basic validation - W&B keys contain alphanumeric and some special characters
    if re.match(r'^[a-zA-Z0-9_\-\.]+$', token):
        return True
    
    return False


async def validate_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials],
    config: MCPAuthConfig
) -> str:
    """
    Validate Bearer token (W&B API key) for MCP access.
    
    Accepts any valid-looking W&B API key. The actual validation
    happens when the key is used to call W&B APIs.
    
    Returns:
        The W&B API key to use for operations
        
    Raises:
        HTTPException: 401 Unauthorized with WWW-Authenticate header
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization required - please provide your W&B API key as a Bearer token",
            headers={
                "WWW-Authenticate": 'Bearer realm="W&B MCP"'
            }
        )
    
    token = credentials.credentials.strip()  # Strip any whitespace
    
    # Basic format validation
    if not is_valid_wandb_api_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid W&B API key format. Got {len(token)} characters. "
                   f"Get your key at: https://wandb.ai/authorize",
            headers={
                "WWW-Authenticate": 'Bearer realm="W&B MCP", error="invalid_token"'
            }
        )
    
    logger.debug(f"Bearer token validated successfully (length: {len(token)})")
    return token


async def mcp_auth_middleware(request: Request, call_next):
    """
    FastAPI middleware for MCP authentication on HTTP transport.
    
    Only applies to MCP endpoints (/mcp/*).
    Extracts the client's W&B API key from the Bearer token and stores it
    for use in W&B operations.
    """
    # Only apply auth to MCP endpoints
    if not request.url.path.startswith("/mcp"):
        return await call_next(request)
    
    # Skip auth if explicitly disabled (development only)
    if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true":
        logger.warning("MCP authentication is disabled - endpoints are publicly accessible")
        return await call_next(request)
    
    config = MCPAuthConfig()
    
    try:
        # Extract bearer token from Authorization header
        authorization = request.headers.get("Authorization", "")
        credentials = None
        if authorization.startswith("Bearer "):
            # Remove "Bearer " prefix and strip any whitespace
            token = authorization[7:].strip()
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=token
            )
        
        # Validate and get the W&B API key
        wandb_api_key = await validate_bearer_token(credentials, config)
        
        # Make sure the key is clean (no extra whitespace or encoding issues)
        wandb_api_key = wandb_api_key.strip()
        
        # Store the API key in request state for W&B operations
        request.state.wandb_api_key = wandb_api_key
        
        # Set the API key in context for this request
        # Tools will use WandBApiManager.get_api_key() to retrieve it
        from wandb_mcp_server.api_client import WandBApiManager
        token = WandBApiManager.set_context_api_key(wandb_api_key)
        
        # Debug logging
        logger.debug(f"Auth middleware: Set API key in context with length={len(wandb_api_key)}")

        try:
            api = WandBApiManager.get_api()
            viewer = api.viewer
            logger.info(f"Authenticated W&B viewer: {viewer}")
        except Exception as viewer_err:
            logger.warning(f"Could not fetch W&B viewer: {viewer_err}")
        
        try:
            # Continue processing the request
            response = await call_next(request)
        finally:
            # Reset the context after request processing
            WandBApiManager.reset_context_api_key(token)
        
        return response
        
    except HTTPException as e:
        # Return proper error response
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail},
            headers=e.headers
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication failed"},
            headers={
                "WWW-Authenticate": 'Bearer realm="W&B MCP"'
            }
        )


# OAuth-related functions removed - see AUTH_README.md for details
