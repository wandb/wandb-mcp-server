"""Unified, actor-isolated W&B public API client management."""

from collections import OrderedDict
from concurrent.futures import Future
from contextvars import ContextVar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import re
import threading
import time
from typing import Any, Iterator, Mapping, Optional

import wandb

from wandb_mcp_server.config import MCP_WANDB_REQUEST_TIMEOUT_SECONDS, WANDB_API_BASE_URL
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

_DEFAULT_RETRY_AFTER_MS = 1_000
_MAX_RETRY_AFTER_MS = 60_000
_OVERLOAD_STATUS_RE = re.compile(r"\b(?:HTTP\s*)?(429|503)\b", re.IGNORECASE)
_OVERLOAD_HINT_RE = re.compile(
    r"\b(overload(?:ed)?|busy|capacity|rate.?limit|service unavailable|"
    r"temporar(?:y|ily)|too many requests|try again)\b",
    re.IGNORECASE,
)

# Context variable for storing the current request's API key
api_key_context: ContextVar[Optional[str]] = ContextVar("wandb_api_key", default=None)


class WandBServerBusy(RuntimeError):
    """Retryable W&B backend saturation surfaced to the MCP boundary."""

    def __init__(self, *, status_code: int, retry_after_ms: int) -> None:
        self.status_code = status_code
        self.retry_after_ms = retry_after_ms
        super().__init__("The W&B service is busy; retry this tool call.")

    def as_dict(self) -> dict[str, object]:
        return {
            "error": "server_busy",
            "message": str(self),
            "retryable": True,
            "retry_after_ms": self.retry_after_ms,
        }


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for nested in (
            getattr(current, "exc", None),
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)


def _retry_after_ms(value: object) -> int:
    if value is None:
        return _DEFAULT_RETRY_AFTER_MS
    raw = str(value).strip()
    try:
        seconds = max(0.0, float(raw))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            seconds = max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return _DEFAULT_RETRY_AFTER_MS
    return min(_MAX_RETRY_AFTER_MS, max(_DEFAULT_RETRY_AFTER_MS, round(seconds * 1_000)))


def wandb_server_busy_from_exception(exc: BaseException) -> WandBServerBusy | None:
    """Classify nested W&B overload errors without retry amplification."""
    if isinstance(exc, WandBServerBusy):
        return exc

    status_code: int | None = None
    retry_after: object = None
    messages: list[str] = []
    for current in _exception_chain(exc):
        if isinstance(current, WandBServerBusy):
            return current
        messages.append(str(current))
        response = getattr(current, "response", None)
        candidate = getattr(response, "status_code", None)
        if candidate in {429, 503}:
            status_code = int(candidate)
            headers = getattr(response, "headers", None)
            if isinstance(headers, Mapping):
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
            for attr in ("reason", "text"):
                response_text = getattr(response, attr, None)
                if response_text:
                    messages.append(str(response_text))
            break

    combined_message = " ".join(messages)
    if status_code is None:
        match = _OVERLOAD_STATUS_RE.search(combined_message)
        if match:
            status_code = int(match.group(1))

    if status_code == 503 and not _OVERLOAD_HINT_RE.search(combined_message):
        return None
    if status_code not in {429, 503}:
        return None
    return WandBServerBusy(
        status_code=status_code,
        retry_after_ms=_retry_after_ms(retry_after),
    )


def raise_for_wandb_server_busy(exc: BaseException) -> None:
    """Preserve overload backpressure across tool-specific fallback paths."""
    if busy := wandb_server_busy_from_exception(exc):
        raise busy from exc


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
