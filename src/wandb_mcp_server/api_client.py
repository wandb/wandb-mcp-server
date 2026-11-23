"""Unified, actor-isolated W&B public API client management."""

from collections import OrderedDict
from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass
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
_USERNAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MISSING_USERNAMES = frozenset({"", "anonymous", "none", "null", "unknown"})
_AUTH_FAILURE_STATUSES = frozenset({401, 403})
_AUTH_FAILURE_HINT_RE = re.compile(
    r"\b(?:HTTP\s*)?(?:401|403)\b|unauthori[sz]ed|invalid.*api.?key|authentication.?failed",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ViewerIdentity:
    """Cached, non-secret viewer classification for analytics attribution."""

    actor_type: str  # "user" | "service_account" | "unauthenticated" | "unresolved"
    username: Optional[str] = None
    email_domain: Optional[str] = None


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


def wandb_server_busy_from_exception(
    exc: BaseException, *, require_overload_hint: bool = True
) -> WandBServerBusy | None:
    """Classify nested W&B overload errors, optionally requiring a hint for 503."""
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

    if status_code == 503 and require_overload_hint and not _OVERLOAD_HINT_RE.search(combined_message):
        return None

    if status_code not in {429, 503}:
        return None

    return WandBServerBusy(
        status_code=status_code,
        retry_after_ms=_retry_after_ms(retry_after),
    )


def raise_for_wandb_server_busy(exc: BaseException, *, require_overload_hint: bool = True) -> None:
    """Preserve overload backpressure across tool-specific fallback paths."""
    if busy := wandb_server_busy_from_exception(exc, require_overload_hint=require_overload_hint):
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
    # Analytics attribution: one viewer lookup per actor and endpoint per TTL.
    # Authentication only checks key format, so misses are bounded separately
    # (well-formed bogus keys cannot evict real users) and concurrent lookups
    # are capped; a lookup that finds no free slot is skipped, not queued.
    _viewer_identity_cache: OrderedDict[str, tuple[float, ViewerIdentity]] = OrderedDict()
    _viewer_identity_misses: OrderedDict[str, tuple[float, str]] = OrderedDict()
    _viewer_identity_lookups: dict[str, Future[Optional[ViewerIdentity]]] = {}
    _viewer_identity_slots = threading.BoundedSemaphore(16)
    _viewer_identity_ttl_seconds = 6 * 60 * 60.0
    _viewer_identity_negative_ttl_seconds = 300.0
    _viewer_identity_max_entries = 4096
    _viewer_identity_max_misses = 1024

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

    @staticmethod
    def _actor_cache_key(api_key: str) -> str:
        """Return an irreversible, endpoint-bound digest for one API key."""
        return hashlib.sha256(f"{WANDB_API_BASE_URL}\0{api_key}".encode()).hexdigest()

    @staticmethod
    def _extract_viewer_email_domain(attrs: Mapping[str, Any]) -> Optional[str]:
        """Extract only the domain portion of a materialized viewer email."""
        email = attrs.get("email")
        if isinstance(email, str) and "@" in email:
            domain = email.split("@", 1)[1].strip().lower()
            if domain and "@" not in domain and len(domain) <= 253:
                return domain
        return None

    @classmethod
    def _classify_materialized_viewer(cls, viewer: Any) -> Optional[ViewerIdentity]:
        """Classify a viewer from populated fields without invoking lazy properties."""
        try:
            attrs = vars(viewer).get("_attrs")
        except TypeError:
            return None
        if type(attrs) is not dict:
            return None
        email_domain = cls._extract_viewer_email_domain(attrs)
        raw_username = attrs.get("username")
        if isinstance(raw_username, str) and raw_username.strip():
            username = raw_username.strip()
            if username.lower() in _MISSING_USERNAMES or _USERNAME_RE.fullmatch(username) is None:
                return None
            return ViewerIdentity(actor_type="user", username=username, email_domain=email_domain)
        if "username" not in attrs or raw_username not in (None, ""):
            return None
        entity = attrs.get("entity")
        viewer_id = attrs.get("id")
        has_valid_entity = isinstance(entity, str) and _USERNAME_RE.fullmatch(entity.strip()) is not None
        has_valid_id = isinstance(viewer_id, str) and bool(viewer_id.strip())
        if has_valid_entity or has_valid_id:
            return ViewerIdentity(actor_type="service_account", username=None, email_domain=email_domain)
        return None

    @classmethod
    def _materialized_username(cls, viewer: Any) -> Optional[str]:
        """Read a validated username from a viewer's populated fields only."""
        identity = cls._classify_materialized_viewer(viewer)
        return identity.username if identity is not None and identity.actor_type == "user" else None

    @staticmethod
    def _is_authentication_failure(exc: BaseException) -> bool:
        """Return True when a viewer lookup failed due to rejected credentials."""
        if wandb_http_status_from_exception(exc) in _AUTH_FAILURE_STATUSES:
            return True
        for current in _exception_chain(exc):
            if type(current).__name__ in {"AuthenticationError", "PermissionError"}:
                return True
            if _AUTH_FAILURE_HINT_RE.search(str(current)):
                return True
        return False

    @staticmethod
    def _viewer_info_dict(
        identity: Optional[ViewerIdentity],
        *,
        include_email_domain: bool = False,
    ) -> Optional[dict[str, str]]:
        """Serialize a cached ViewerIdentity to the analytics viewer_info shape."""
        if identity is None or identity.actor_type == "unresolved":
            return None
        if identity.actor_type == "user" and identity.username:
            info: dict[str, str] = {"username": identity.username}
            if include_email_domain and identity.email_domain:
                info["email_domain"] = identity.email_domain
            return info
        if identity.actor_type in {"service_account", "unauthenticated"}:
            info = {"actor_type": identity.actor_type}
            if include_email_domain and identity.email_domain:
                info["email_domain"] = identity.email_domain
            return info
        return None

    @classmethod
    def get_cached_viewer_info(cls) -> dict[str, str] | None:
        """Read already-discovered viewer attribution without creating a client or request.

        Uses the attribution cache filled by ``resolve_viewer_identity`` or,
        failing that, a viewer already materialized on this actor's unexpired,
        endpoint-bound client. W&B's ``viewer`` property performs a lookup on
        first access, so only its populated backing fields are inspected.
        """
        api_key = cls.get_api_key()
        if not api_key:
            return None
        hit, identity = cls.peek_viewer_identity(api_key)
        if hit:
            return cls._viewer_info_dict(identity)
        cache_key = cls._actor_cache_key(api_key)
        with cls._api_cache_lock:
            cached = cls._api_cache.get(cache_key)
            if cached is None or cached[0] <= time.monotonic():
                return None
            try:
                viewer = vars(cached[1]).get("_viewer")
            except TypeError:
                return None
            identity = cls._classify_materialized_viewer(viewer)
        if identity is None:
            return None
        cls._store_viewer_identity(cache_key, identity)
        return cls._viewer_info_dict(identity)

    @classmethod
    def peek_viewer_identity(cls, api_key: str) -> tuple[bool, Optional[ViewerIdentity]]:
        """Return ``(hit, identity)`` from the attribution cache without I/O."""
        cache_key = cls._actor_cache_key(api_key)
        with cls._api_cache_lock:
            return cls._peek_viewer_identity_locked(cache_key)

    @classmethod
    def peek_viewer_username(cls, api_key: str) -> tuple[bool, Optional[str]]:
        """Return ``(hit, username)`` from the attribution cache without I/O."""
        hit, identity = cls.peek_viewer_identity(api_key)
        return hit, (identity.username if identity is not None else None)

    @classmethod
    def _peek_viewer_identity_locked(cls, cache_key: str) -> tuple[bool, Optional[ViewerIdentity]]:
        now = time.monotonic()
        cached = cls._viewer_identity_cache.get(cache_key)
        if cached is not None and cached[0] > now:
            cls._viewer_identity_cache.move_to_end(cache_key)
            val = cached[1]
            if isinstance(val, ViewerIdentity):
                return True, val
            if isinstance(val, str):
                return True, ViewerIdentity(actor_type="user", username=val)
        miss_entry = cls._viewer_identity_misses.get(cache_key)
        if miss_entry is not None:
            miss_expires_at, miss_actor_type = (
                miss_entry if isinstance(miss_entry, tuple) else (float(miss_entry), "unresolved")
            )
            if miss_expires_at > now:
                return True, ViewerIdentity(actor_type=miss_actor_type)
        return False, None

    @classmethod
    def _peek_viewer_username_locked(cls, cache_key: str) -> tuple[bool, Optional[str]]:
        hit, identity = cls._peek_viewer_identity_locked(cache_key)
        return hit, (identity.username if identity is not None else None)

    @classmethod
    def resolve_viewer_identity(cls, api_key: str) -> Optional[ViewerIdentity]:
        """Return the actor's classified ViewerIdentity, looking it up at most once per TTL.

        Concurrent callers for one actor share a single lookup. Confirmed human
        users and team service accounts are cached for the full positive TTL;
        authentication failures and transient misses are cached briefly in their
        own bounded table so bogus keys cannot evict real actors.
        """
        cache_key = cls._actor_cache_key(api_key)
        with cls._api_cache_lock:
            hit, identity = cls._peek_viewer_identity_locked(cache_key)
            if hit:
                return identity
            lookup = cls._viewer_identity_lookups.get(cache_key)
            is_owner = lookup is None
            if is_owner:
                lookup = Future()
                cls._viewer_identity_lookups[cache_key] = lookup
        if not is_owner:
            return lookup.result()

        identity = None
        miss_actor_type = "unresolved"
        looked_up = False
        try:
            if cls._viewer_identity_slots.acquire(blocking=False):
                looked_up = True
                try:
                    identity = cls._classify_materialized_viewer(cls._viewer_lookup_client(api_key).viewer)
                except Exception as exc:
                    if cls._is_authentication_failure(exc):
                        miss_actor_type = "unauthenticated"
                    logger.debug("Viewer identity lookup failed (%s)", type(exc).__name__)
                finally:
                    cls._viewer_identity_slots.release()
        finally:
            if looked_up:
                cls._store_viewer_identity(cache_key, identity, miss_actor_type=miss_actor_type)
                if identity is None:
                    identity = ViewerIdentity(actor_type=miss_actor_type)
            with cls._api_cache_lock:
                cls._viewer_identity_lookups.pop(cache_key, None)
            lookup.set_result(identity)
        return identity

    @classmethod
    def resolve_viewer_username(cls, api_key: str) -> Optional[str]:
        """Return the actor's W&B username, looking it up at most once per TTL."""
        identity = cls.resolve_viewer_identity(api_key)
        return identity.username if identity is not None else None

    @classmethod
    def _viewer_lookup_client(cls, api_key: str) -> wandb.Api:
        """Reuse this actor's cached client, else build one that is not cached.

        Keys reaching attribution are only format-checked; caching their
        clients would let bogus keys evict real users' clients.
        """
        cache_key = cls._actor_cache_key(api_key)
        with cls._api_cache_lock:
            cached = cls._api_cache.get(cache_key)
            if cached is not None and cached[0] > time.monotonic():
                return cached[1]
        return wandb.Api(
            api_key=api_key,
            overrides={"base_url": WANDB_API_BASE_URL},
            timeout=MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
        )

    @classmethod
    def remember_viewer(cls, api_key: str, viewer: Any) -> None:
        """Cache the identity of a viewer that functional work already fetched."""
        identity = cls._classify_materialized_viewer(viewer)
        if identity is not None:
            cls._store_viewer_identity(cls._actor_cache_key(api_key), identity)

    @classmethod
    def _store_viewer_identity(
        cls,
        cache_key: str,
        identity: Optional[ViewerIdentity],
        *,
        miss_actor_type: str = "unresolved",
    ) -> None:
        """Cache a resolved user/service-account identity, or a short-lived miss."""
        now = time.monotonic()
        with cls._api_cache_lock:
            if identity is not None and identity.actor_type in {"user", "service_account"}:
                cls._viewer_identity_misses.pop(cache_key, None)
                cls._viewer_identity_cache[cache_key] = (now + cls._viewer_identity_ttl_seconds, identity)
                cls._viewer_identity_cache.move_to_end(cache_key)
                while len(cls._viewer_identity_cache) > cls._viewer_identity_max_entries:
                    cls._viewer_identity_cache.popitem(last=False)
            else:
                cls._viewer_identity_misses[cache_key] = (
                    now + cls._viewer_identity_negative_ttl_seconds,
                    miss_actor_type,
                )
                cls._viewer_identity_misses.move_to_end(cache_key)
                while len(cls._viewer_identity_misses) > cls._viewer_identity_max_misses:
                    cls._viewer_identity_misses.popitem(last=False)

    @classmethod
    def _store_viewer_username(cls, cache_key: str, username: Optional[str]) -> None:
        """Compatibility wrapper around ``_store_viewer_identity``."""
        identity = ViewerIdentity(actor_type="user", username=username) if username else None
        cls._store_viewer_identity(cache_key, identity)

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

        cache_key = cls._actor_cache_key(api_key)
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
            cls._viewer_identity_cache.clear()
            cls._viewer_identity_misses.clear()
            cls._viewer_identity_lookups.clear()

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
