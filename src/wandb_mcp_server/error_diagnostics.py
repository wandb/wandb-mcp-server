"""Low-cardinality failure facts, never exception messages or request values."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from httpx import Response as HttpxResponse
from pydantic import ValidationError
from requests import Response as RequestsResponse
from wandb.proto.wandb_api_pb2 import ApiErrorResponse

_CATEGORIES = frozenset(
    {
        "input_validation",
        "invalid_value",
        "authentication_failed",
        "permission_denied",
        "resource_not_found",
        "tool_timeout",
        "server_busy",
        "upstream_error",
        "internal_error",
        "tool_error",
        "cancelled",
        "selective_read_unavailable",
        "response_too_large",
    }
)
_EXCEPTION_TYPES = frozenset(
    {
        "ToolError",
        "ValidationError",
        "ValueError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "RuntimeError",
        "TimeoutError",
        "CancelledError",
        "PermissionError",
        "ConnectionError",
        "HTTPError",
        "HTTPStatusError",
        "ReadTimeout",
        "ConnectTimeout",
        "ReadError",
        "ConnectError",
        "CommError",
        "AuthenticationError",
        "AuthorizationError",
        "UsageError",
        "ToolDeadlineExceeded",
        "WandBServerBusy",
        "WandbApiFailedError",
        "WandBQueryValidationError",
        "WandBCursorValidationError",
        "ToolInputValidationError",
        "AdmissionRejected",
        "SelectiveReadUnavailable",
        "GraphQLResponseTooLarge",
        "other_exception",
    }
)
# Names are public schema fields, not nested mapping keys or customer values.
_PUBLIC_FIELDS = frozenset(
    {
        "entity_name",
        "project_name",
        "resource",
        "run_id",
        "run_id_a",
        "run_id_b",
        "sweep_id",
        "report_name",
        "filters",
        "order",
        "limit",
        "include",
        "summary_keys",
        "config_keys",
        "response_mode",
        "cursor",
        "keys",
        "samples",
        "min_step",
        "max_step",
        "x_axis",
        "target_x",
        "tolerance",
        "stream",
        "include_history_overlap",
        "history_keys",
        "history_samples",
        "loss_key",
        "val_loss_key",
        "query",
        "max_items",
        "trace_ids",
        "organization",
        "filter",
    }
)
_VALIDATION_CODES = frozenset(
    {
        "missing",
        "string_type",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "int_type",
        "int_parsing",
        "int_from_float",
        "float_type",
        "float_parsing",
        "finite_number",
        "bool_type",
        "bool_parsing",
        "list_type",
        "dict_type",
        "literal_error",
        "enum",
        "none_required",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "too_short",
        "too_long",
        "extra_forbidden",
        "value_error",
        "assertion_error",
        "model_type",
        "json_invalid",
        "json_type",
        "union_tag_invalid",
        "union_tag_not_found",
        "recursion_loop",
    }
)
_STATUSES = frozenset({400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504})


class ToolInputValidationError(ValueError):
    """Keep existing input-error text while carrying separate safe diagnostics."""

    def __init__(self, message: str, *, field: str | None = None, code: str = "value_error") -> None:
        super().__init__(message)
        self.diagnostic_field = field if field in _PUBLIC_FIELDS else None
        self.diagnostic_code = code if code in _VALIDATION_CODES else "value_error"


def sanitize_error_diagnostics(value: object) -> dict[str, Any] | None:
    """Apply the same fixed allowlists at generation and every telemetry sink."""
    if not isinstance(value, dict):
        return None
    category = value.get("category")
    if not isinstance(category, str) or category not in _CATEGORIES:
        return None
    result: dict[str, Any] = {"category": category}
    for key in ("exception_type", "cause_type"):
        item = value.get(key)
        if isinstance(item, str) and item in _EXCEPTION_TYPES:
            result[key] = item
    for key, allowed in (("validation_fields", _PUBLIC_FIELDS), ("validation_codes", _VALIDATION_CODES)):
        items = value.get(key)
        if isinstance(items, (list, tuple)):
            safe = sorted({item for item in items[:32] if isinstance(item, str) and item in allowed})[:8]
            if safe:
                result[key] = safe
    status = value.get("upstream_status")
    if type(status) is int and status in _STATUSES:
        result["upstream_status"] = status
    return result


def _exception_type(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        return "ValidationError"
    name = type(exc).__name__
    return name if name in _EXCEPTION_TYPES else "other_exception"


def _safe_attribute(value: object, name: str) -> object:
    # Read stored exception metadata, not arbitrary SDK/custom descriptors.
    if isinstance(value, BaseException):
        if name in {"__cause__", "__context__"}:
            return BaseException.__dict__[name].__get__(value)
        return BaseException.__dict__["__dict__"].__get__(value).get(name)
    if type(value) in {HttpxResponse, RequestsResponse}:
        return vars(value).get(name)
    if type(value) is ApiErrorResponse and name == "http_status":
        return value.http_status
    return None


def _http_status(value: object) -> int | None:
    for name in ("status_code", "status", "http_status", "http_status_code"):
        status = _safe_attribute(value, name)
        if type(status) is int and status in _STATUSES:
            return status
    return None


def exception_diagnostics(exc: BaseException, *, public_fields: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Follow bounded cause chains without formatting any exception or input."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    pending = [exc]
    while pending and len(chain) < 8:
        current = pending.pop()
        if id(current) in seen:
            continue
        chain.append(current)
        seen.add(id(current))
        # Explicit causes take precedence; unrelated exception-handling context
        # must not override the failure deliberately preserved by a wrapper.
        for name in ("__cause__", "exc", "__context__"):
            nested = _safe_attribute(current, name)
            if isinstance(nested, BaseException) and id(nested) not in seen:
                pending.append(nested)
                break
    result: dict[str, Any] = {"category": "tool_error", "exception_type": _exception_type(exc)}
    if len(chain) > 1:
        result["cause_type"] = _exception_type(chain[-1])
    # Prefer the original typed failure to FastMCP's generic ToolError wrapper.
    for cause in reversed(chain):
        kind = _exception_type(cause)
        response = _safe_attribute(cause, "response")
        status = _http_status(cause)
        if status is None and response is not None:
            status = _http_status(response)
        if type(status) is int and status in _STATUSES:
            result["upstream_status"] = status
            result["category"] = {
                400: "invalid_value",
                401: "authentication_failed",
                403: "permission_denied",
                404: "resource_not_found",
                408: "tool_timeout",
                422: "invalid_value",
                429: "server_busy",
                503: "server_busy",
                504: "tool_timeout",
            }.get(status, "upstream_error")
            break
        if isinstance(cause, ToolInputValidationError):
            result.update(category="input_validation", validation_codes=[cause.diagnostic_code])
            if cause.diagnostic_field in public_fields:
                result["validation_fields"] = [cause.diagnostic_field]
            break
        if isinstance(cause, ValidationError):
            result["category"] = "input_validation"
            if cause.error_count() > 32:
                break
            fields, codes = set(), set()
            for item in cause.errors(include_url=False, include_context=False, include_input=False)[:32]:
                location = item.get("loc", ())
                if location and isinstance(location[0], str) and location[0] in public_fields & _PUBLIC_FIELDS:
                    fields.add(location[0])
                code = item.get("type")
                if isinstance(code, str) and code in _VALIDATION_CODES:
                    codes.add(code)
            result.update(category="input_validation", validation_fields=sorted(fields), validation_codes=sorted(codes))
            break
        category = None
        if isinstance(cause, asyncio.CancelledError):
            category = "tool_timeout" if any(isinstance(item, TimeoutError) for item in chain) else "cancelled"
        elif isinstance(cause, TimeoutError) or kind in {"ReadTimeout", "ConnectTimeout", "ToolDeadlineExceeded"}:
            category = "tool_timeout"
        elif isinstance(cause, PermissionError) or kind == "AuthorizationError":
            category = "permission_denied"
        elif kind == "AuthenticationError":
            category = "authentication_failed"
        elif kind in {"WandBQueryValidationError", "WandBCursorValidationError"}:
            category = "input_validation"
        elif kind == "SelectiveReadUnavailable":
            category = "selective_read_unavailable"
        elif kind == "GraphQLResponseTooLarge":
            category = "response_too_large"
        elif kind in {"WandBServerBusy", "AdmissionRejected"}:
            category = "server_busy"
        elif kind in {
            "HTTPError",
            "HTTPStatusError",
            "CommError",
            "ConnectionError",
            "ReadError",
            "ConnectError",
            "WandbApiFailedError",
        }:
            category = "upstream_error"
        elif isinstance(cause, ValueError):
            category = "invalid_value"
        elif kind in {"TypeError", "KeyError", "AttributeError", "RuntimeError"}:
            category = "internal_error"
        if category is not None:
            result["category"] = category
            break
    return sanitize_error_diagnostics(result) or {"category": "tool_error"}


@dataclass
class ToolErrorDiagnostics:
    public_fields: frozenset[str]
    value: dict[str, Any] | None = None
    active: bool = True


current_error_diagnostics: ContextVar[ToolErrorDiagnostics | None] = ContextVar("tool_error_diagnostics", default=None)


def record_exception_diagnostics(exc: BaseException) -> None:
    """Preserve a swallowed terminal cause for the single public call event.

    The state object follows copied executor contexts. It stores only sanitized
    scalars, never the exception, and stops accepting updates after call cleanup.
    """
    state = current_error_diagnostics.get()
    if state is not None and state.active and state.value is None:
        try:
            state.value = exception_diagnostics(exc, public_fields=state.public_fields)
        except Exception:
            # Diagnostics must never replace the tool's real result/failure.
            state.value = {"category": "tool_error"}
