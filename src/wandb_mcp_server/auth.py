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
    
    def __init__(self):
        self.resource_metadata_url = os.environ.get(
            "MCP_RESOURCE_METADATA_URL", 
            "/.well-known/oauth-protected-resource"
        )
        # Point to W&B's Auth0 instance for reference
        self.authorization_server = os.environ.get(
            "MCP_AUTH_SERVER", 
            "https://wandb.auth0.com"
        )


def is_valid_wandb_api_key(token: str) -> bool:
    """
    Check if a token looks like a valid W&B API key format.
    W&B API keys are typically 40 characters of alphanumeric + some special chars.
    """
    if not token or len(token) < 20 or len(token) > 100:
        return False
    # Basic validation - W&B keys contain alphanumeric and some special characters
    # This is a permissive check since W&B key format may vary
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
                "WWW-Authenticate": f'Bearer realm="W&B MCP", '
                                   f'resource_metadata="{config.resource_metadata_url}"'
            }
        )
    
    token = credentials.credentials
    
    # Basic format validation
    if not is_valid_wandb_api_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid W&B API key format. Get your key at: https://wandb.ai/authorize",
            headers={
                "WWW-Authenticate": f'Bearer realm="W&B MCP", '
                                   f'error="invalid_token", '
                                   f'resource_metadata="{config.resource_metadata_url}"'
            }
        )
    
    logger.debug("Bearer token validated successfully")
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
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=authorization[7:]  # Remove "Bearer " prefix
            )
        
        # Validate and get the W&B API key
        wandb_api_key = await validate_bearer_token(credentials, config)
        
        # Store the API key in request state for W&B operations
        # The MCP tools should access this from the request context
        request.state.wandb_api_key = wandb_api_key
        
        # For now, we'll set it in environment (in production, use contextvars)
        # Save the original value to restore later
        original_api_key = os.environ.get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_api_key
        
        try:
            # Continue processing
            response = await call_next(request)
        finally:
            # Restore original environment
            if original_api_key:
                os.environ["WANDB_API_KEY"] = original_api_key
            elif "WANDB_API_KEY" in os.environ:
                del os.environ["WANDB_API_KEY"]
            
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
                "WWW-Authenticate": f'Bearer realm="W&B MCP", '
                                   f'resource_metadata="{config.resource_metadata_url}"'
            }
        )


def create_resource_metadata_response(config: MCPAuthConfig) -> Dict[str, Any]:
    """
    Create OAuth 2.0 Protected Resource Metadata response (RFC 9728).
    
    This tells MCP clients that we use W&B API keys as Bearer tokens.
    Points to W&B's Auth0 instance where users can get their API keys.
    """
    return {
        "resource": os.environ.get("MCP_SERVER_URL", "https://wandb-mcp-server.hf.space"),
        "authorization_servers": [config.authorization_server],
        "bearer_methods_supported": ["header"],
        "resource_documentation": "https://github.com/wandb/wandb-mcp-server",
        "authentication_note": "Use your W&B API key as a Bearer token. Get your key at https://wandb.ai/authorize",
    }
