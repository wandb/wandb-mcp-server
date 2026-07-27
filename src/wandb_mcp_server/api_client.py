"""Unified, actor-isolated W&B public API client management."""

from collections import OrderedDict
from concurrent.futures import Future
from contextvars import ContextVar
import hashlib
import threading
import time
from typing import Any, Optional

import wandb

from wandb_mcp_server.config import MCP_WANDB_REQUEST_TIMEOUT_SECONDS, WANDB_API_BASE_URL
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

# Context variable for storing the current request's API key
api_key_context: ContextVar[Optional[str]] = ContextVar("wandb_api_key", default=None)


class WandBApiManager:
    """
    Manages W&B API instances with per-request API keys.

    ``wandb.Api`` validates credentials and initializes its transport during
    construction. Rebuilding it for every tool call creates avoidable backend
    traffic and can overload authentication during a cold concurrent burst.
    Clients are therefore cached briefly by an irreversible API-key digest.
    A single-flight future prevents duplicate construction for the same actor,
    while different actors initialize independently.
    """

    _api_cache: OrderedDict[str, tuple[float, wandb.Api]] = OrderedDict()
    _api_initializations: dict[str, Future[wandb.Api]] = {}
    _api_cache_lock = threading.Lock()
    _api_cache_ttl_seconds = 300.0
    _api_cache_max_entries = 128

    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Get the API key for the current request context.

        For HTTP mode: API key comes from auth middleware via contextvar.
        For STDIO mode: API key should be set via set_context_api_key() at startup.

        Returns:
            The API key from context only, no fallbacks.
        """
        # Get from context variable only - no fallbacks!
        # HTTP: Set by auth middleware
        # STDIO: Set at startup from CLI/netrc/env
        api_key = api_key_context.get()
        return api_key

    @classmethod
    def get_api(cls, api_key: Optional[str] = None) -> wandb.Api:
        """
        Get a W&B API instance with the specified or current API key.

        Args:
            api_key: Optional API key to use. If not provided, uses context or environment.

        Returns:
            A configured wandb.Api instance.

        Raises:
            ValueError: If no API key is available.
        """
        if api_key is None:
            api_key = cls.get_api_key()

        if not api_key:
            raise ValueError(
                "No W&B API key available in request context. "
                "For HTTP: Ensure authentication middleware is configured. "
                "For STDIO: Ensure API key is set at server startup."
            )

        cache_key = hashlib.sha256(f"{WANDB_API_BASE_URL}\0{api_key}".encode()).hexdigest()
        now = time.monotonic()
        with cls._api_cache_lock:
            cls._evict_expired(now)
            cached = cls._api_cache.get(cache_key)
            if cached is not None:
                cls._api_cache.move_to_end(cache_key)
                return cached[1]

            initialization = cls._api_initializations.get(cache_key)
            if initialization is None:
                initialization = Future()
                cls._api_initializations[cache_key] = initialization
                is_initializer = True
            else:
                is_initializer = False

        if not is_initializer:
            return initialization.result()

        try:
            api = wandb.Api(
                api_key=api_key,
                overrides={"base_url": WANDB_API_BASE_URL},
                timeout=MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            with cls._api_cache_lock:
                cls._api_initializations.pop(cache_key, None)
            initialization.set_exception(exc)
            raise

        with cls._api_cache_lock:
            cls._api_cache[cache_key] = (
                time.monotonic() + cls._api_cache_ttl_seconds,
                api,
            )
            cls._api_cache.move_to_end(cache_key)
            while len(cls._api_cache) > cls._api_cache_max_entries:
                cls._api_cache.popitem(last=False)
            cls._api_initializations.pop(cache_key, None)
        initialization.set_result(api)
        return api

    @classmethod
    def _evict_expired(cls, now: float) -> None:
        """Remove expired clients while the cache lock is held."""
        expired = [key for key, (expires_at, _) in cls._api_cache.items() if expires_at <= now]
        for key in expired:
            cls._api_cache.pop(key, None)

    @classmethod
    def _clear_api_cache(cls) -> None:
        """Clear cached clients; used by tests and controlled shutdowns."""
        with cls._api_cache_lock:
            cls._api_cache.clear()
            cls._api_initializations.clear()

    @staticmethod
    def set_context_api_key(api_key: str) -> Any:
        """
        Set the API key in the current context.

        Args:
            api_key: The API key to set.

        Returns:
            A token that can be used to reset the context.
        """
        return api_key_context.set(api_key)

    @staticmethod
    def reset_context_api_key(token: Any) -> None:
        """
        Reset the API key context.

        Args:
            token: The token returned from set_context_api_key.
        """
        api_key_context.reset(token)


def get_wandb_api(api_key: Optional[str] = None) -> wandb.Api:
    """
    Convenience function to get a W&B API instance.

    This is the primary function that should be used throughout the codebase
    to get a W&B API instance with proper API key handling.

    Args:
        api_key: Optional API key. If not provided, uses context or environment.

    Returns:
        A configured wandb.Api instance.
    """
    return WandBApiManager.get_api(api_key)
