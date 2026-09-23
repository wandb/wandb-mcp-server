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


class WandBWriteOutcomeUnknown(RuntimeError):
    """A dispatched W&B write ended without a confirmed response."""

    def __init__(self) -> None:
        super().__init__("W&B did not confirm whether the write completed.")


class WandBReportCreationFailed(RuntimeError):
    """A confirmed report-creation failure safe to expose at the MCP boundary."""

    def __init__(self) -> None:
        super().__init__("The W&B report could not be created.")


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


def _http_status(value: object) -> int | None:
    """Read status codes from requests/httpx and W&B service responses."""
    for attr in ("status_code", "status", "http_status", "http_status_code"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    return None


def wandb_http_status_from_exception(exc: BaseException) -> int | None:
    """Return the first HTTP status carried anywhere in a W&B exception chain."""
    for current in _exception_chain(exc):
        if status := _http_status(current):
            return status
        response = getattr(current, "response", None)
        if response is not None and (status := _http_status(response)):
            return status
    return None


def wandb_write_outcome_unknown_from_exception(exc: BaseException) -> bool:
    """Return whether an exception chain contains an unconfirmed W&B write."""
    return any(isinstance(current, WandBWriteOutcomeUnknown) for current in _exception_chain(exc))


def wandb_report_creation_failed_from_exception(exc: BaseException) -> bool:
    """Return whether an exception chain contains a safe report failure."""
    return any(isinstance(current, WandBReportCreationFailed) for current in _exception_chain(exc))


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
        candidate = _http_status(current) or (_http_status(response) if response is not None else None)
        if candidate in {429, 503}:
            status_code = int(candidate)
            headers = getattr(response, "headers", None)
            if isinstance(headers, Mapping):
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
            for attr in ("reason", "text", "message"):
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
    _viewer_enrichments: dict[str, Future[dict[str, str] | None]] = {}
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
    def get_cached_viewer_info(cls) -> dict[str, str] | None:
        """Read an already-discovered username without creating a client or request.

        W&B's ``viewer`` property performs a lookup on first access. Only inspect
        its populated backing fields on this actor's unexpired, endpoint-bound
        client. A cold cache or a tool that has not needed viewer data therefore
        leaves telemetry on its key fingerprint.
        """
        api_key = cls.get_api_key()
        if not api_key:
            return None
        cache_key = hashlib.sha256(f"{WANDB_API_BASE_URL}\0{api_key}".encode()).hexdigest()
        with cls._api_cache_lock:
            cached = cls._api_cache.get(cache_key)
            if cached is None or cached[0] <= time.monotonic():
                return None
            try:
                viewer = vars(cached[1]).get("_viewer")
                attrs = vars(viewer).get("_attrs")
            except TypeError:
                return None
            if type(attrs) is not dict:
                return None
            username = attrs.get("username")
            if not isinstance(username, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", username) is None:
                return None
            return {"username": username}

    @classmethod
    def get_or_enrich_viewer_info(cls) -> dict[str, str] | None:
        """Resolve one authenticated username lookup per cached actor client.

        The lookup is best-effort telemetry enrichment: callers must never use
        its result for authentication or authorization. Concurrent requests for
        one actor share the same future, and a failed lookup is not retried until
        that actor's bounded API-client cache entry is replaced.
        """
        api_key = cls.get_api_key()
        if not api_key:
            return None
        cache_key = hashlib.sha256(f"{WANDB_API_BASE_URL}\0{api_key}".encode()).hexdigest()
        api = cls.get_api(api_key)
        with cls._api_cache_lock:
            enrichment = cls._viewer_enrichments.get(cache_key)
            if enrichment is None:
                enrichment = Future()
                cls._viewer_enrichments[cache_key] = enrichment
                is_enricher = True
            else:
                is_enricher = False

        if not is_enricher:
            return enrichment.result()

        result: dict[str, str] | None = None
        fatal_error: BaseException | None = None
        try:
            # ``viewer`` is cached by the SDK after this first lookup. Read only
            # its materialized backing data so telemetry never invokes another
            # lazy property while extracting the username.
            viewer = api.viewer
            try:
                attrs = vars(viewer).get("_attrs")
            except TypeError:
                attrs = None
            username = attrs.get("username") if type(attrs) is dict else None
            result = (
                {"username": username}
                if isinstance(username, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", username) is not None
                else None
            )
        except Exception:
            pass
        except BaseException as exc:
            # Wake any same-actor waiters before propagating process-control
            # exceptions such as KeyboardInterrupt or SystemExit.
            fatal_error = exc
        finally:
            enrichment.set_result(result)
        if fatal_error is not None:
            raise fatal_error
        return result

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
                evicted_key, _ = cls._api_cache.popitem(last=False)
                cls._viewer_enrichments.pop(evicted_key, None)
            cls._api_initializations.pop(cache_key, None)
        initialization.set_result(api)
        return api

    @classmethod
    def _evict_expired(cls, now: float) -> None:
        """Remove expired clients while the cache lock is held."""
        expired = [key for key, (expires_at, _) in cls._api_cache.items() if expires_at <= now]
        for key in expired:
            cls._api_cache.pop(key, None)
            cls._viewer_enrichments.pop(key, None)

    @classmethod
    def _clear_api_cache(cls) -> None:
        """Clear cached clients; used by tests and controlled shutdowns."""
        with cls._api_cache_lock:
            cls._api_cache.clear()
            cls._api_initializations.clear()
            cls._viewer_enrichments.clear()

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
