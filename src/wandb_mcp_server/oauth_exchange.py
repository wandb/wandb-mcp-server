"""OAuth token exchange -- converts validated JWTs into W&B-native credentials.

Follows the same pattern as the Python SDK (``wandb/sdk/lib/credentials.py``):
exchange an identity JWT at Gorilla's ``/oidc/token`` endpoint using the
RFC 7523 JWT-bearer grant, receive a short-lived ``wb_at_*`` access token,
and cache it until expiry.

The exchanged token is what downstream ``wandb.Api`` and Weave trace-server
calls actually consume -- **never** the raw JWT.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

EXCHANGE_TIMEOUT_SECONDS = 15
EXPIRY_SAFETY_MARGIN_SECONDS = 60


class OAuthExchangeError(Exception):
    """Raised when the ``/oidc/token`` exchange fails."""


@dataclass(frozen=True)
class WBAccessToken:
    """A W&B-native access token obtained via token exchange."""

    token: str
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() >= (self.expires_at - EXPIRY_SAFETY_MARGIN_SECONDS)


class _TokenCache:
    """Thread-safe TTL cache for exchanged tokens.

    Keyed by a truncated SHA-256 of the source JWT so that repeated
    requests with the same bearer token reuse the exchanged credential
    without hitting ``/oidc/token`` on every request.
    """

    def __init__(self) -> None:
        self._store: dict[str, WBAccessToken] = {}
        self._lock = threading.Lock()

    def get(self, jwt_hash: str) -> WBAccessToken | None:
        with self._lock:
            entry = self._store.get(jwt_hash)
            if entry is not None and not entry.is_expired:
                return entry
            self._store.pop(jwt_hash, None)
            return None

    def put(self, jwt_hash: str, token: WBAccessToken) -> None:
        with self._lock:
            self._store[jwt_hash] = token

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_cache = _TokenCache()


def _jwt_cache_key(jwt_str: str) -> str:
    return hashlib.sha256(jwt_str.encode()).hexdigest()[:32]


def exchange_jwt_for_wb_token(jwt_str: str, base_url: str) -> WBAccessToken:
    """Exchange a validated JWT for a W&B-native access token.

    Args:
        jwt_str: The raw JWT that has already passed JWKS validation.
        base_url: The W&B API base URL (``WANDB_BASE_URL``).

    Returns:
        A ``WBAccessToken`` containing the ``wb_at_*`` string and its
        expiration time.

    Raises:
        OAuthExchangeError: If the exchange endpoint returns non-200
            or the response is malformed.
    """
    cache_key = _jwt_cache_key(jwt_str)
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("Token exchange cache hit")
        return cached

    url = f"{base_url.rstrip('/')}/oidc/token"
    data: dict[str, Any] = {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": jwt_str,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        resp = httpx.post(url, data=data, headers=headers, timeout=EXCHANGE_TIMEOUT_SECONDS)
    except Exception as exc:
        raise OAuthExchangeError(f"Token exchange request failed: {exc}") from exc

    if resp.status_code != 200:
        raise OAuthExchangeError(f"Token exchange returned {resp.status_code}: {resp.text[:200]}")

    try:
        body = resp.json()
        access_token = body["access_token"]
        expires_in = float(body.get("expires_in", 3600))
    except (KeyError, ValueError, TypeError) as exc:
        raise OAuthExchangeError(f"Malformed exchange response: {exc}") from exc

    wb_token = WBAccessToken(
        token=access_token,
        expires_at=time.time() + expires_in,
    )
    _cache.put(cache_key, wb_token)
    logger.info("JWT exchanged for W&B access token (cached until expiry)")
    return wb_token


def clear_exchange_cache() -> None:
    """Clear the token exchange cache (useful for testing)."""
    _cache.clear()
