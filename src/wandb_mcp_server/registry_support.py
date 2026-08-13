"""Shared, bounded support for read-only W&B Registry tools."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import threading
from typing import Any

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.api_client import (
    wandb_http_status_from_exception,
    wandb_server_busy_from_exception,
)
from wandb_mcp_server.config import MAX_RESPONSE_TOKENS, structured_error
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.wandb_graphql import GraphQLResponseTooLarge
from wandb_mcp_server.wandb_selective_reads import (
    SelectiveReadUnavailable,
    fetch_registry_organization_info,
)

_MAX_FILTER_DEPTH = 12
_MAX_FILTER_BYTES = 64 * 1024
_MAX_ORGANIZATION_CANDIDATES = 20
_MAX_TEXT_BYTES = 512
_MAX_PAGINATOR_REQUESTS = 8
_CACHE_ATTRIBUTE = "_wandb_mcp_registry_organization_resolution"
_LOCK_ATTRIBUTE = "_wandb_mcp_registry_organization_lock"
_CACHE_GUARD = threading.Lock()


class RegistryInputError(ValueError):
    """Raised before constructing or calling a W&B client."""


class OrganizationRequired(ValueError):
    """Raised when the authenticated actor has multiple possible organizations."""

    def __init__(self, candidates: tuple[str, ...], *, candidates_truncated: bool = False) -> None:
        super().__init__("Specify organization because more than one organization is accessible.")
        self.candidates = candidates
        self.candidates_truncated = candidates_truncated


class OrganizationResolutionFailed(ValueError):
    """Raised when no organization can be resolved from a bounded response."""

    def __init__(self) -> None:
        super().__init__("Unable to resolve a W&B organization for registry access.")


class RegistryMalformedResponse(ValueError):
    """Raised when a registry response does not match the public SDK contract."""


def validate_optional_identifier(name: str, value: str | None) -> None:
    """Validate an optional, bounded non-empty registry identifier."""
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise RegistryInputError(f"{name} must be a non-empty string when provided")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise RegistryInputError(f"{name} exceeds the {_MAX_TEXT_BYTES}-byte limit")


def validate_registry_filter(value: Mapping[str, Any] | None) -> None:
    """Apply the typed-query filter depth, type, and byte limits."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise RegistryInputError("filter must be an object")
    _validate_filter_value(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise RegistryInputError("filter must contain finite JSON-compatible values") from None
    if len(encoded) > _MAX_FILTER_BYTES:
        raise RegistryInputError(f"filter exceeds the {_MAX_FILTER_BYTES}-byte limit")


def _validate_filter_value(value: Any, *, depth: int = 0) -> None:
    if depth > _MAX_FILTER_DEPTH:
        raise RegistryInputError(f"filter exceeds the maximum depth of {_MAX_FILTER_DEPTH}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RegistryInputError("filter must use string keys")
            _validate_filter_value(child, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_filter_value(child, depth=depth + 1)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise RegistryInputError("filter must contain finite JSON-compatible values")


def resolve_registry_organization(api: Any, explicit: str | None = None) -> str:
    """Resolve a registry organization without constructing another SDK client.

    Successful and ambiguous results are cached directly on the actor-isolated
    ``wandb.Api`` instance. The manager already bounds that instance's lifetime.
    """
    validate_optional_identifier("organization", explicit)
    if explicit is not None:
        return explicit.strip()

    lock = _organization_lock(api)
    with lock:
        cached = vars(api).get(_CACHE_ATTRIBUTE)
        if cached is not None:
            return _read_cached_resolution(cached)

        settings = getattr(api, "settings", None)
        configured = settings.get("organization") if isinstance(settings, Mapping) else None
        if isinstance(configured, str) and configured.strip():
            validate_optional_identifier("organization", configured)
            resolution = ("resolved", configured.strip(), False)
            setattr(api, _CACHE_ATTRIBUTE, resolution)
            return configured.strip()

        configured_entity = settings.get("entity") if isinstance(settings, Mapping) else None
        entity = configured_entity.strip() if isinstance(configured_entity, str) and configured_entity.strip() else None
        validate_optional_identifier("entity", entity)

        try:
            data = fetch_registry_organization_info(api, entity=entity)
        except SelectiveReadUnavailable as exc:
            raise RegistryMalformedResponse("organization lookup returned a malformed response") from exc
        resolution = _parse_organization_resolution(data)
        setattr(api, _CACHE_ATTRIBUTE, resolution)
        return _read_cached_resolution(resolution)


def _organization_lock(api: Any) -> threading.Lock:
    existing = vars(api).get(_LOCK_ATTRIBUTE)
    if isinstance(existing, type(threading.Lock())):
        return existing
    with _CACHE_GUARD:
        existing = vars(api).get(_LOCK_ATTRIBUTE)
        if isinstance(existing, type(threading.Lock())):
            return existing
        lock = threading.Lock()
        setattr(api, _LOCK_ATTRIBUTE, lock)
        return lock


def _parse_organization_resolution(data: Any) -> tuple[str, str | tuple[str, ...], bool]:
    if not isinstance(data, Mapping):
        raise RegistryMalformedResponse("organization lookup returned a malformed response")
    entity = data.get("entity")
    if entity is None:
        viewer = data.get("viewer")
        if viewer is None:
            return ("failed", (), False)
        if not isinstance(viewer, Mapping):
            raise RegistryMalformedResponse("organization lookup returned a malformed viewer")
        default_entity = viewer.get("entity")
        if default_entity is not None:
            if not isinstance(default_entity, str) or not default_entity.strip():
                raise RegistryMalformedResponse("organization lookup returned a malformed default entity")
            default_entity = _bounded_upstream_identifier("default entity", default_entity)
        organizations = viewer.get("organizations")
        if organizations is not None and not isinstance(organizations, list):
            raise RegistryMalformedResponse("organization lookup returned a malformed organization list")
        if isinstance(organizations, list) and default_entity:
            matching = [
                item
                for item in organizations
                if isinstance(item, Mapping)
                and isinstance(item.get("orgEntity"), Mapping)
                and item["orgEntity"].get("name") == default_entity
            ]
            if len(matching) == 1:
                organizations = matching
        entity = {"organization": None, "user": {"organizations": organizations or []}}
    if not isinstance(entity, Mapping):
        raise RegistryMalformedResponse("organization lookup returned a malformed entity")

    organization = entity.get("organization")
    if isinstance(organization, Mapping):
        name = organization.get("name")
        org_entity = organization.get("orgEntity")
        if isinstance(name, str) and name.strip() and isinstance(org_entity, Mapping):
            entity_name = org_entity.get("name")
            if isinstance(entity_name, str) and entity_name.strip():
                _bounded_upstream_identifier("organization entity name", entity_name)
                return (
                    "resolved",
                    _bounded_upstream_identifier("organization name", name),
                    False,
                )

    user = entity.get("user")
    organizations = user.get("organizations") if isinstance(user, Mapping) else None
    if organizations is None:
        return ("failed", (), False)
    if not isinstance(organizations, list):
        raise RegistryMalformedResponse("organization lookup returned a malformed organization list")

    candidates: list[str] = []
    for item in organizations:
        if item is None:
            continue
        if not isinstance(item, Mapping):
            raise RegistryMalformedResponse("organization lookup returned a malformed organization")
        name = item.get("name")
        org_entity = item.get("orgEntity")
        entity_name = org_entity.get("name") if isinstance(org_entity, Mapping) else None
        if isinstance(name, str) and name.strip() and isinstance(entity_name, str) and entity_name.strip():
            _bounded_upstream_identifier("organization entity name", entity_name)
            candidates.append(_bounded_upstream_identifier("organization name", name))
    unique = tuple(sorted(set(candidates)))
    if len(unique) == 1:
        return ("resolved", unique[0], False)
    if not unique:
        return ("failed", (), False)
    truncated = len(unique) > _MAX_ORGANIZATION_CANDIDATES
    return ("required", unique[:_MAX_ORGANIZATION_CANDIDATES], truncated)


def _read_cached_resolution(resolution: Any) -> str:
    if not isinstance(resolution, tuple) or len(resolution) != 3:
        raise OrganizationResolutionFailed()
    kind, value, truncated = resolution
    if kind == "resolved" and isinstance(value, str):
        return value
    if kind == "required" and isinstance(value, tuple):
        raise OrganizationRequired(value, candidates_truncated=bool(truncated))
    raise OrganizationResolutionFailed()


def registry_error_result(exc: Exception) -> dict[str, object]:
    """Map registry failures to stable, non-sensitive tool errors."""
    if isinstance(exc, RegistryInputError):
        return structured_error("invalid_input", str(exc))
    if isinstance(exc, OrganizationRequired):
        return structured_error(
            "organization_required",
            str(exc),
            organization_candidates=list(exc.candidates),
            candidates_truncated=exc.candidates_truncated,
        )
    if isinstance(exc, OrganizationResolutionFailed):
        return structured_error("organization_resolution_failed", str(exc))
    if isinstance(exc, GraphQLResponseTooLarge):
        return structured_error(
            "response_too_large",
            "The bounded W&B registry response was too large; request fewer items",
        )
    if busy := wandb_server_busy_from_exception(exc):
        return busy.as_dict()

    status = wandb_http_status_from_exception(exc)
    names, message = _exception_signals(exc)
    if (
        status == 401
        or "authentication" in names
        or "unauth" in names
        or "invalid api key" in message
        or "no w&b api key" in message
    ):
        return structured_error("authentication_failed", "W&B authentication failed")
    if status == 403 or "permission" in message or "forbidden" in message:
        return structured_error("permission_denied", "W&B denied registry access")
    if status == 404 or "not found" in message or "could not find" in message:
        return structured_error("resource_not_found", "The requested W&B registry resource was not found")
    if "timeout" in names or "timed out" in message or "did not respond in time" in message:
        return structured_error(
            "upstream_timeout",
            "The bounded W&B registry request timed out",
            retryable=True,
        )
    if isinstance(exc, RegistryMalformedResponse) or "validationerror" in names:
        return structured_error("malformed_response", "W&B returned malformed registry data")
    return structured_error("registry_query_failed", "W&B registry query failed")


def require_registry(registries: Any) -> Any:
    """Read at most one exact registry match from a public SDK paginator."""
    page, _ = bounded_sdk_page(registries, 1)
    if not page:
        error = LookupError("registry not found")
        error.status_code = 404  # type: ignore[attr-defined]
        raise error from None
    return page[0]


def bounded_sdk_page(values: Any, limit: int) -> tuple[list[Any], bool]:
    """Drive real W&B paginators one backend page at a time under a hard cap.

    W&B 0.28 registry paginator ``__next__`` implementations may internally
    skip arbitrarily many empty authorization-filtered pages. Calling the SDK's
    one-page loader directly keeps its public object conversion while letting
    MCP enforce request and cursor-progress bounds.
    """
    target = max(1, limit) + 1
    loader = getattr(values, "_load_page", None)
    objects = getattr(values, "objects", None)
    per_page = getattr(values, "per_page", None)
    if not callable(loader) or not isinstance(objects, list) or not isinstance(per_page, int):
        if type(values).__module__.startswith("wandb."):
            raise RegistryMalformedResponse("installed W&B registry paginator is unsupported")
        return _bounded_plain_iterable(values, limit)

    rows: list[Any] = list(objects[:target])
    seen_cursors: set[str] = set()
    initial_cursor = getattr(values, "cursor", None)
    if isinstance(initial_cursor, str) and initial_cursor:
        seen_cursors.add(initial_cursor)
    requests = 0
    expected_requests = math.ceil(target / max(1, per_page))
    request_limit = min(_MAX_PAGINATOR_REQUESTS, expected_requests + 2)

    while len(rows) < target and bool(getattr(values, "more", False)):
        if requests >= request_limit:
            raise RegistryMalformedResponse("registry pagination exceeded its request limit")
        raise_if_tool_deadline_exceeded()
        before_count = len(objects)
        before_cursor = getattr(values, "cursor", None)
        loaded = loader()
        requests += 1
        if not isinstance(loaded, bool):
            raise RegistryMalformedResponse("registry paginator returned an invalid page state")
        rows.extend(objects[before_count:target])
        more = bool(getattr(values, "more", False))
        if not more:
            break
        after_cursor = getattr(values, "cursor", None)
        if not isinstance(after_cursor, str) or not after_cursor:
            raise RegistryMalformedResponse("registry pagination omitted its continuation cursor")
        if after_cursor == before_cursor or after_cursor in seen_cursors:
            raise RegistryMalformedResponse("registry pagination repeated its continuation cursor")
        seen_cursors.add(after_cursor)

    return rows[:limit], len(rows) > limit or bool(getattr(values, "more", False))


def _bounded_plain_iterable(values: Any, limit: int) -> tuple[list[Any], bool]:
    iterator = iter(values)
    rows: list[Any] = []
    for _ in range(limit + 1):
        raise_if_tool_deadline_exceeded()
        try:
            rows.append(next(iterator))
        except StopIteration:
            break
    return rows[:limit], len(rows) > limit


def _bounded_upstream_identifier(label: str, value: str) -> str:
    stripped = value.strip()
    if len(stripped.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise RegistryMalformedResponse(f"organization lookup returned an oversized {label}")
    return stripped


def _exception_signals(exc: BaseException) -> tuple[str, str]:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    names: list[str] = []
    messages: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        names.append(type(current).__name__.lower())
        messages.append(str(current).lower())
        response = getattr(current, "response", None)
        response_message = getattr(response, "message", None)
        if response_message:
            messages.append(str(response_message).lower())
        for nested in (getattr(current, "exc", None), getattr(current, "__cause__", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return " ".join(names), " ".join(messages)


def nullable_string(value: Any) -> str | None:
    """Serialize a nullable SDK scalar without converting ``None`` to text."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except (TypeError, ValueError):
            return None
    return str(value)


def fit_registry_response(
    payload: dict[str, Any],
    *,
    aliases: tuple[str, ...],
    optional_fields: tuple[str, ...] = ("description", "artifact_types", "tags"),
) -> dict[str, Any]:
    """Trim optional metadata and tail items to the configured response budget."""
    items = payload.get("items")
    if not isinstance(items, list):
        return structured_error("malformed_response", "W&B returned malformed registry data")
    omitted_fields: set[str] = set()
    dropped_items = 0

    if _tokens(payload) > MAX_RESPONSE_TOKENS:
        for field in optional_fields:
            removed = False
            for item in items:
                if isinstance(item, dict) and field in item:
                    item.pop(field)
                    removed = True
            if removed:
                omitted_fields.add(field)
            if _tokens(payload) <= MAX_RESPONSE_TOKENS:
                break

    while items and _tokens(payload) > MAX_RESPONSE_TOKENS:
        items.pop()
        dropped_items += 1

    if omitted_fields or dropped_items:
        payload["returned_count"] = len(items)
        payload["count"] = len(items)
        payload["has_more"] = True
        payload["project_exhaustive"] = False
        payload["truncated"] = True
        payload["truncation"] = {
            "applied": True,
            "reason": "response_token_budget",
            "omitted_fields": sorted(omitted_fields),
            "dropped_items": dropped_items,
        }
        for alias in aliases:
            payload[alias] = items

    if _tokens(payload) > MAX_RESPONSE_TOKENS:
        return structured_error(
            "response_too_large",
            "The W&B registry metadata exceeded the configured response budget",
        )
    return payload


def _tokens(payload: Mapping[str, Any]) -> int:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise RegistryMalformedResponse("registry response was not JSON-compatible") from None
    return count_tokens_conservative(encoded)


__all__ = [
    "OrganizationRequired",
    "OrganizationResolutionFailed",
    "RegistryInputError",
    "RegistryMalformedResponse",
    "bounded_sdk_page",
    "fit_registry_response",
    "nullable_string",
    "registry_error_result",
    "require_registry",
    "resolve_registry_organization",
    "validate_optional_identifier",
    "validate_registry_filter",
]
