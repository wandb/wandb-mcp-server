"""
Authentication middleware for W&B MCP Server.

Implements Bearer token validation for HTTP transport as per
MCP specification: https://modelcontextprotocol.io/specification/draft/basic/authorization

Clients send their W&B API keys as Bearer tokens, which the server
then uses for all W&B operations on behalf of that client.

Session management follows MCP Streamable HTTP transport: the server
issues an ``Mcp-Session-Id`` on the first authenticated request.
Clients must include it on subsequent requests.  Sessions are tracked
via :class:`MultiTenantSessionManager` with TTL-based cleanup.
"""

import hashlib
import logging
import os
import re
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Bearer token security scheme
bearer_scheme = HTTPBearer(auto_error=False)


MCP_AUTH_MODE = os.environ.get("MCP_AUTH_MODE", "api-key")


class MCPAuthConfig:
    """Configuration for MCP authentication.

    Supports two modes controlled by ``MCP_AUTH_MODE``:

    - ``api-key`` (default): Bearer token is a raw W&B API key.
    - ``oauth``: Bearer token is an OAuth 2.1 JWT validated against
      the issuer's JWKS. The JWT ``sub`` claim is used for identity;
      downstream W&B calls still require an API key resolved from the
      token (see token-to-API-key bridge).
    """

    pass


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
    if re.match(r"^[a-zA-Z0-9_\-\.]+$", token):
        return True

    return False


async def validate_bearer_token(credentials: Optional[HTTPAuthorizationCredentials], config: MCPAuthConfig) -> str:
    """Validate a Bearer token for MCP access.

    Behavior depends on ``MCP_AUTH_MODE``:

    - ``api-key``: token is a raw W&B API key (validated by format).
    - ``oauth``: token is a JWT validated against the issuer JWKS.
      If the token is not a JWT, falls back to API key validation
      for backward compatibility.

    Returns:
        The W&B API key (api-key mode) or the raw JWT (oauth mode)
        to use for downstream operations.

    Raises:
        HTTPException: 401 Unauthorized with WWW-Authenticate header.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization required - please provide your W&B API key as a Bearer token",
            headers={"WWW-Authenticate": 'Bearer realm="W&B MCP"'},
        )

    token = credentials.credentials.strip()

    if MCP_AUTH_MODE == "oauth":
        from wandb_mcp_server.oauth import OAuthConfig, is_jwt, validate_oauth_token

        if is_jwt(token):
            try:
                oauth_config = OAuthConfig.from_env()
                claims = validate_oauth_token(token, oauth_config)
                logger.info(f"OAuth token validated: sub={claims.get('sub', '?')}")
                return token
            except Exception as e:
                logger.warning(f"OAuth token validation failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid OAuth token: {e}",
                    headers={"WWW-Authenticate": 'Bearer realm="W&B MCP", error="invalid_token"'},
                )
        else:
            logger.debug("Token is not a JWT; falling back to API key validation")

    if not is_valid_wandb_api_key(token):
        logger.debug(f"Rejected API key: length={len(token)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid W&B API key format. Get your key at: https://wandb.ai/authorize",
            headers={"WWW-Authenticate": 'Bearer realm="W&B MCP", error="invalid_token"'},
        )

    logger.debug(f"Bearer token validated successfully (length: {len(token)})")
    return token


def _resolve_session_id(request: Request, wandb_api_key: str) -> tuple[str, bool]:
    """Determine the MCP session ID for this request.

    Returns:
        ``(session_id, is_new)`` -- *is_new* is ``True`` when the server
        generated the ID (no client header present).
    """
    client_session = request.headers.get("Mcp-Session-Id") or request.headers.get("mcp-session-id")
    if client_session:
        if len(client_session) > 128 or not re.match(r"^[a-zA-Z0-9_\-]+$", client_session):
            logger.warning("Invalid session ID format from client; issuing server-generated ID")
            return f"sess_{uuid.uuid4().hex}", True
        return client_session, False
    return f"sess_{uuid.uuid4().hex}", True


async def mcp_auth_middleware(request: Request, call_next):
    """FastAPI middleware for MCP authentication and session management.

    Only applies to ``/mcp/*`` endpoints.  Extracts the W&B API key from
    the Bearer token, resolves or issues an ``Mcp-Session-Id``, sets the
    session contextvar, and emits analytics events.
    """
    if not request.url.path.startswith("/mcp"):
        return await call_next(request)

    if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true":
        logger.warning("MCP authentication is disabled - endpoints are publicly accessible")
        return await call_next(request)

    config = MCPAuthConfig()

    # --- Authenticate (narrow scope: only auth errors become 401) ----------
    try:
        authorization = request.headers.get("Authorization", "")
        credentials = None
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        wandb_api_key = await validate_bearer_token(credentials, config)
        wandb_api_key = wandb_api_key.strip()
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail}, headers=e.headers)
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication failed"},
            headers={"WWW-Authenticate": 'Bearer realm="W&B MCP"'},
        )

    # --- Set up request context (API key, viewer, session) ----------------
    request.state.wandb_api_key = wandb_api_key

    from wandb_mcp_server.api_client import WandBApiManager

    api_key_token = WandBApiManager.set_context_api_key(wandb_api_key)

    viewer = None
    try:
        api = WandBApiManager.get_api()
        viewer = api.viewer
        viewer_id = getattr(viewer, "username", None) or getattr(viewer, "entity", None) or "<unknown>"
        logger.info(f"Authenticated W&B viewer: {viewer_id}")
    except Exception as viewer_err:
        logger.warning(f"Could not fetch W&B viewer: {viewer_err}")

    # --- Session management -----------------------------------------------
    # Finalize session_id *before* setting the contextvar so that
    # reset() always restores the original value (None), not a
    # stale/stolen session ID from a mismatch recovery path.
    from wandb_mcp_server.session_manager import SessionCapacityError, current_session_id

    session_id, is_new_session = _resolve_session_id(request, wandb_api_key)

    try:
        from wandb_mcp_server.session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.create_session(wandb_api_key, session_id=session_id)
    except SessionCapacityError:
        logger.warning("Session capacity exceeded for API key")
        WandBApiManager.reset_context_api_key(api_key_token)
        return JSONResponse(
            status_code=429,
            content={"error": "Too many concurrent sessions for this API key"},
        )
    except ValueError:
        logger.warning("Session ID rejected (API key mismatch); issuing new server session")
        session_id = f"sess_{uuid.uuid4().hex}"
        is_new_session = True
        try:
            mgr.create_session(wandb_api_key, session_id=session_id)
        except SessionCapacityError:
            logger.warning("Session capacity exceeded on retry")
            WandBApiManager.reset_context_api_key(api_key_token)
            return JSONResponse(
                status_code=429,
                content={"error": "Too many concurrent sessions for this API key"},
            )
        except Exception:
            logger.debug("Session creation failed on mismatch retry (non-fatal)")
    except Exception as sm_err:
        logger.debug(f"Session manager unavailable (non-fatal): {sm_err}")

    request.state.session_id = session_id
    session_ctx_token = current_session_id.set(session_id)

    # --- Analytics: session event -----------------------------------------
    try:
        from wandb_mcp_server.analytics import get_analytics_tracker

        get_analytics_tracker().track_user_session(
            session_id=session_id,
            viewer_info=viewer,
            api_key_hash=hashlib.sha256(wandb_api_key.encode()).hexdigest(),
        )
    except Exception as analytics_err:
        logger.debug(f"Analytics tracking failed (non-fatal): {analytics_err}")

    # --- Execute request (errors here propagate as 500, not 401) ----------
    request_start = time.monotonic()
    request_id = str(uuid.uuid4())[:8]
    try:
        response = await call_next(request)
    except Exception:
        _track_request_event(request_start, request_id, session_id, request, 500, viewer)
        raise
    finally:
        WandBApiManager.reset_context_api_key(api_key_token)
        current_session_id.reset(session_ctx_token)

    if is_new_session:
        response.headers["Mcp-Session-Id"] = session_id

    _track_request_event(request_start, request_id, session_id, request, response.status_code, viewer)

    return response


def _track_request_event(
    request_start: float,
    request_id: str,
    session_id: str,
    request: Request,
    status_code: int,
    viewer: object,
) -> None:
    """Emit a request analytics event, swallowing any errors."""
    try:
        from wandb_mcp_server.analytics import AnalyticsTracker, get_analytics_tracker

        elapsed_ms = (time.monotonic() - request_start) * 1000
        get_analytics_tracker().track_request(
            request_id=request_id,
            session_id=session_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=round(elapsed_ms, 2),
            user_id=AnalyticsTracker._extract_user_id(viewer) if viewer else None,
            email_domain=AnalyticsTracker._extract_email_domain(viewer) if viewer else None,
        )
    except Exception:
        pass
