"""OAuth 2.1 token validation for MCP Server.

Implements JWT-based access token validation per the MCP Authorization
specification (https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization).

When ``MCP_AUTH_MODE=oauth``, Bearer tokens are validated as JWTs against
the issuer's JWKS endpoint. The issuer URL is configured via
``MCP_OAUTH_ISSUER``.

This module is only imported when ``MCP_AUTH_MODE=oauth``; the default
``api-key`` mode does not load it.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)

JWKS_CACHE_TTL_SECONDS = 300


@dataclass
class OAuthConfig:
    """OAuth 2.1 configuration read from environment variables."""

    issuer: str = ""
    audience: str = ""
    required_scopes: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "OAuthConfig":
        """Build config from MCP_OAUTH_* environment variables."""
        issuer = os.environ.get("MCP_OAUTH_ISSUER", "")
        audience = os.environ.get("MCP_OAUTH_AUDIENCE", "")
        scopes_str = os.environ.get("MCP_OAUTH_REQUIRED_SCOPES", "")
        scopes = [s.strip() for s in scopes_str.split(",") if s.strip()] if scopes_str else []
        return cls(issuer=issuer, audience=audience, required_scopes=scopes)


class JWKSClient:
    """Fetches and caches JWKS keys from the issuer's well-known endpoint."""

    def __init__(self, issuer: str) -> None:
        self._issuer = issuer.rstrip("/")
        self._jwks_uri: str | None = None
        self._jwks_data: dict[str, Any] | None = None
        self._last_fetch: float = 0

    def _discover_jwks_uri(self) -> str:
        """Fetch the OIDC discovery document to find the jwks_uri."""
        discovery_url = f"{self._issuer}/.well-known/openid-configuration"
        try:
            resp = httpx.get(discovery_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            jwks_uri = data.get("jwks_uri")
            if not jwks_uri:
                raise ValueError(f"No jwks_uri in discovery document at {discovery_url}")
            return jwks_uri
        except Exception as e:
            logger.warning(f"OIDC discovery failed at {discovery_url}: {e}")
            return f"{self._issuer}/.well-known/jwks.json"

    def get_signing_key(self, token: str) -> jwt.PyJWK:
        """Get the signing key for a JWT from the cached JWKS."""
        now = time.monotonic()
        if self._jwks_data is None or (now - self._last_fetch) > JWKS_CACHE_TTL_SECONDS:
            if self._jwks_uri is None:
                self._jwks_uri = self._discover_jwks_uri()
            try:
                resp = httpx.get(self._jwks_uri, timeout=10)
                resp.raise_for_status()
                self._jwks_data = resp.json()
                self._last_fetch = now
            except Exception as e:
                if self._jwks_data is not None:
                    logger.warning(f"JWKS refresh failed, using cached keys: {e}")
                else:
                    raise

        jwks_client = jwt.PyJWKSet.from_dict(self._jwks_data)
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if kid:
            for key in jwks_client.keys:
                if key.key_id == kid:
                    return key
        if jwks_client.keys:
            return jwks_client.keys[0]
        raise jwt.InvalidTokenError("No matching key found in JWKS")


_jwks_clients: dict[str, JWKSClient] = {}


def _get_jwks_client(issuer: str) -> JWKSClient:
    """Get or create a cached JWKSClient for the given issuer."""
    if issuer not in _jwks_clients:
        _jwks_clients[issuer] = JWKSClient(issuer)
    return _jwks_clients[issuer]


def validate_oauth_token(token: str, config: OAuthConfig) -> dict[str, Any]:
    """Validate an OAuth 2.1 access token (JWT).

    Args:
        token: The raw JWT string from the Authorization header.
        config: OAuth configuration with issuer, audience, and scopes.

    Returns:
        The decoded JWT claims dict.

    Raises:
        jwt.InvalidTokenError: If the token is invalid, expired, or
            fails audience/issuer/scope checks.
    """
    if not config.issuer:
        raise jwt.InvalidTokenError("MCP_OAUTH_ISSUER not configured")

    client = _get_jwks_client(config.issuer)
    signing_key = client.get_signing_key(token)

    decode_options: dict[str, Any] = {}
    decode_kwargs: dict[str, Any] = {
        "algorithms": ["RS256", "ES256"],
        "options": decode_options,
    }

    if config.issuer:
        decode_kwargs["issuer"] = config.issuer
    if config.audience:
        decode_kwargs["audience"] = config.audience

    claims = jwt.decode(token, signing_key.key, **decode_kwargs)

    if config.required_scopes:
        token_scopes = set(claims.get("scope", "").split())
        missing = set(config.required_scopes) - token_scopes
        if missing:
            raise jwt.InvalidTokenError(f"Token missing required scopes: {missing}")

    return claims


def is_jwt(token: str) -> bool:
    """Quick check whether a token looks like a JWT (3 dot-separated base64 segments)."""
    parts = token.split(".")
    return len(parts) == 3 and all(len(p) > 0 for p in parts)
