"""Unified API client management for W&B operations.

This module provides a consistent pattern for managing W&B API instances
with per-request credentials.  It supports two credential types:

- **API keys** (default): sent as ``Basic api:<key>`` via the standard
  ``wandb.Api(api_key=...)`` path.
- **``wb_at_*`` access tokens** (OAuth exchange path): sent as
  ``Authorization: Bearer <token>`` by patching the transport after
  construction, matching how Gorilla's auth handler dispatches them.
"""

from contextvars import ContextVar
from typing import Any, Optional

import requests.auth
import wandb

from wandb_mcp_server.config import WANDB_BASE_URL
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

api_key_context: ContextVar[Optional[str]] = ContextVar("wandb_api_key", default=None)

_DUMMY_API_KEY = "x" * 40


def is_wb_access_token(credential: str) -> bool:
    """Return True if the credential is a Gorilla-issued ``wb_at_*`` access token."""
    return credential.startswith("wb_at_")


class _BearerTokenAuth(requests.auth.AuthBase):
    """``requests`` auth handler that sends ``Authorization: Bearer <token>``.

    Gorilla recognises ``wb_at_*`` tokens only in the Bearer header
    (``auth.go`` line 399-409), not via Basic auth.  This class follows
    the same pattern the wandb SDK uses for ``_IdentityTokenAuth``.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self._token}"
        return r


class WandBApiManager:
    """Manages W&B API instances with per-request credentials."""

    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get the credential for the current request context."""
        return api_key_context.get()

    @staticmethod
    def get_api(api_key: Optional[str] = None) -> wandb.Api:
        """Get a ``wandb.Api`` instance with the appropriate auth transport.

        For regular API keys the standard ``wandb.Api(api_key=...)``
        path is used (HTTP Basic).  For ``wb_at_*`` access tokens the
        API is constructed with a dummy key, then the transport is
        patched to send ``Authorization: Bearer <token>`` instead.

        Args:
            api_key: Credential to use.  Falls back to the context var.

        Returns:
            A configured ``wandb.Api`` ready for GQL calls.
        """
        if api_key is None:
            api_key = WandBApiManager.get_api_key()

        if not api_key:
            raise ValueError(
                "No W&B API key available in request context. "
                "For HTTP: Ensure authentication middleware is configured. "
                "For STDIO: Ensure API key is set at server startup."
            )

        if is_wb_access_token(api_key):
            return _build_bearer_api(api_key)

        return wandb.Api(api_key=api_key, overrides={"base_url": WANDB_BASE_URL})

    @staticmethod
    def set_context_api_key(api_key: str) -> Any:
        """Set the credential in the current async context."""
        return api_key_context.set(api_key)

    @staticmethod
    def reset_context_api_key(token: Any) -> None:
        """Restore the previous credential context."""
        api_key_context.reset(token)


def _build_bearer_api(wb_at_token: str) -> wandb.Api:
    """Construct a ``wandb.Api`` that authenticates with Bearer.

    ``wandb.Api(api_key=...)`` rejects ``wb_at_*`` tokens because the
    dot separator fails ``check_api_key()``'s regex.  We work around
    this by constructing with a dummy key that passes validation, then
    immediately replacing the session auth with a Bearer handler.
    """
    api = wandb.Api(api_key=_DUMMY_API_KEY, overrides={"base_url": WANDB_BASE_URL})
    api._base_client.transport.session.auth = _BearerTokenAuth(wb_at_token)
    api.api_key = wb_at_token
    return api


def get_wandb_api(api_key: Optional[str] = None) -> wandb.Api:
    """Convenience wrapper around ``WandBApiManager.get_api``."""
    return WandBApiManager.get_api(api_key)
