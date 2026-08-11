"""Async client helpers for the hosted W&B Agent (ARIA).

The endpoint and payload shapes in this module follow the published service
schema at https://wb-agent.wandb.ai/openapi.json.
"""

import asyncio
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from weakref import WeakKeyDictionary

import httpx

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.config import MAX_RESPONSE_TOKENS, WB_AGENT_BASE_URL, resolve_aria_base_url
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text, sanitize_sensitive_value
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

TURN_STATES = frozenset({"queued", "in_progress", "completed", "errored", "cancelled"})
TERMINAL_STATES = frozenset({"completed", "errored", "cancelled"})
MAX_WAIT_SECONDS = 30
MAX_BATCH_TURNS = 20
MAX_BATCH_CONCURRENCY = 8
MAX_BATCH_GET_REQUESTS = 100
MAX_MESSAGE_BYTES = 32 * 1024
MAX_IDENTIFIER_LENGTH = 512
MAX_SCOPE_LENGTH = 512
MAX_UPSTREAM_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROGRESS_TOOL_RECORDS = 100
MAX_PROGRESS_NODES_PER_RECORD = 200
MAX_PROGRESS_NESTING_DEPTH = 8
POLL_INTERVAL_SECONDS = 1.0
POLL_DEADLINE_SAFETY_SECONDS = 0.5
MIN_RETRY_AFTER_MS = 1_000
MAX_RETRY_AFTER_MS = 30_000
_OVERLOAD_ERROR_TYPES = frozenset({"rate_limited", "service_unavailable"})
_OUTBOUND_LIMITERS: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()
_OUTBOUND_LIMITERS_LOCK = threading.Lock()


def _outbound_limiter() -> asyncio.Semaphore:
    """Return the shared eight-request limiter for the active server loop."""
    loop = asyncio.get_running_loop()
    with _OUTBOUND_LIMITERS_LOCK:
        limiter = _OUTBOUND_LIMITERS.get(loop)
        if limiter is None:
            limiter = asyncio.Semaphore(MAX_BATCH_CONCURRENCY)
            _OUTBOUND_LIMITERS[loop] = limiter
        return limiter


class _SanitizeAriaHTTPLogFilter(logging.Filter):
    """Remove ARIA origins, turn identifiers, and internal hosts from HTTP logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        marker = "/api/v1/turns"
        if isinstance(record.args, tuple) and len(record.args) >= 2:
            rendered_url = str(record.args[1])
            if marker in rendered_url:
                suffix = f"{marker}/<redacted>" if f"{marker}/" in rendered_url else marker
                args = list(record.args)
                args[1] = f"<aria-service>{suffix}"
                record.args = tuple(args)

        # httpcore debug records do not always expose a URL argument. Render
        # once and run the shared infrastructure/credential sanitizer over the
        # complete record so a custom internal ARIA origin cannot escape.
        rendered = record.getMessage()
        sanitized = sanitize_sensitive_text(rendered, max_chars=4_096)
        if sanitized != rendered:
            record.msg = sanitized
            record.args = ()
        return True


for _http_logger_name in (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
):
    _http_logger = logging.getLogger(_http_logger_name)
    if not any(isinstance(item, _SanitizeAriaHTTPLogFilter) for item in _http_logger.filters):
        _http_logger.addFilter(_SanitizeAriaHTTPLogFilter())

ARIA_SEND_MESSAGE_TOOL_DESCRIPTION = """Start or continue an asynchronous conversation with ARIA, the hosted Weights & Biases agent (also called wb-agent).

<when_to_use>
Use ARIA to hand off W&B-native work that may take a while, manipulate or inspect
the W&B UI more directly, or use ARIA's native W&B context, data access, and
multi-step capabilities instead of scraping data yourself. Good tasks include
comparing runs, diagnosing evaluations or Weave traces, inspecting W&B pages,
and carrying out multi-step W&B workflows. Prefer narrower MCP query tools for
simple deterministic lookups and search_wandb_docs_tool for documentation.
</when_to_use>

This tool returns promptly by default because ARIA turns are asynchronous. Keep the returned turn_id. If state is queued or in_progress, call aria_get_turn until is_terminal is true. To continue a completed conversation, call this tool again with parent_turn_id set to the previous turn_id; omit entity and project for continuations. Set wait_seconds to wait briefly for a fast result, but the wait is always bounded.

Root turns may omit both entity and project. If only project is supplied, this
tool resolves the caller's default W&B entity before creating the turn. Results
are compact by default; set include_turn=true only when a bounded raw ARIA turn
snapshot is actually needed. Oversized snapshots include `_truncation` metadata.

Parameters
----------
message : str
    A focused request for ARIA, limited to 32 KiB of UTF-8 text. Include relevant W&B entity/project names, run IDs, call IDs, URLs, metrics, time windows, and the desired output shape.
entity : str, optional
    W&B entity for a new conversation, up to 512 characters. Omit to let ARIA use the account default. Must be omitted for continuations.
project : str, optional
    W&B project for a new conversation, up to 512 characters. Omit to let ARIA use the default. Must be omitted for continuations.
parent_turn_id : str, optional
    Completed ARIA turn to continue, up to 512 characters. Use the exact turn_id returned by the prior call.
wait_seconds : int
    Seconds to poll before returning, from 0 to 30. Default 0 returns the initial asynchronous handle immediately.
include_turn : bool
    Include a bounded raw turn snapshot. Oversized data is truncated with metadata. Default false avoids very large responses.

Returns
-------
dict
    Compact state, progress, scope, turn/thread handles, latest assistant response, and next action. Errors use the MCP error state with a JSON payload and never include the W&B token.
"""

ARIA_GET_TURN_TOOL_DESCRIPTION = """Get the current state and result of an asynchronous ARIA turn.

<when_to_use>
Call after aria_send_message returns queued or in_progress and you need the state
of one turn. For several turns, use aria_get_turns so their bounded wait happens
concurrently in one MCP call. Always rely on is_terminal rather than the presence
of latest_response because ARIA can emit a partial response while still working.
</when_to_use>

With wait_seconds=0 this performs one poll; with 1-30 it polls only for that bounded interval and returns early if the turn completes, errors, or is cancelled. In-progress results include a phase, meaningful status text, message/tool counts, and latest tool activity, plus a suggested poll interval. Reuse the same turn_id until is_terminal is true.

When state is completed, latest_response contains ARIA's newest assistant answer. Continue the conversation with aria_send_message(parent_turn_id=turn_id). When state is errored, inspect error_info. A transport/service error uses the MCP error state and keeps turn_id in its JSON payload so this tool can be retried safely. Results are compact by default; set include_turn=true only for a bounded raw snapshot, which may contain `_truncation` metadata.

Parameters
----------
turn_id : str
    Exact ARIA turn_id returned by aria_send_message, up to 512 characters.
wait_seconds : int
    Seconds to poll before returning, from 0 to 30. Default 0 performs one status request.
include_turn : bool
    Include a bounded raw turn snapshot. Oversized data is truncated with metadata. Default false avoids very large responses.

Returns
-------
dict
    Compact current state, progress, latest response, and continuation/poll guidance, or a native MCP error containing JSON details.
"""

ARIA_GET_TURNS_TOOL_DESCRIPTION = """Poll several asynchronous ARIA turns concurrently in one MCP call.

<when_to_use>
Use after launching multiple ARIA conversations. This avoids serial client-side
bounded polls: provide every outstanding turn_id and this tool fetches each state
concurrently, waiting at most one shared bounded interval. For one turn, prefer
aria_get_turn.
</when_to_use>

The result preserves input order and summarizes terminal, pending, and failed
turns. Individual retryable lookup failures remain attached to their turn IDs;
successful turns are still returned. Always rely on each result's is_terminal.
Results are compact by default.

Parameters
----------
turn_ids : list of str
    One to 20 distinct ARIA turn IDs, each up to 512 characters.
wait_seconds : int
    Shared polling window from 0 to 30 seconds. Default 0 performs one concurrent status request.
include_turn : bool
    Include each bounded raw turn snapshot. Oversized data is truncated with metadata. Default false avoids very large responses.

Returns
-------
dict
    Ordered per-turn results plus terminal, pending, and error counts and retry guidance.
"""


class AriaAPIError(Exception):
    """A safe, agent-facing ARIA service error."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        retryable: bool,
        status_code: Optional[int] = None,
        details: Optional[Any] = None,
        retry_after_ms: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.details = details
        self.retry_after_ms = retry_after_ms

    def as_result(self, *, turn_id: Optional[str] = None, operation: str) -> Dict[str, Any]:
        ambiguous_send = operation == "send" and self.error_type in {
            "request_timeout",
            "network_error",
            "invalid_response",
            "response_too_large",
        }
        error: Dict[str, Any] = {
            "type": "outcome_unknown" if ambiguous_send else self.error_type,
            "message": (
                "ARIA did not confirm whether the message submission created a turn."
                if ambiguous_send
                else self.message
            ),
            "retryable": False if ambiguous_send else self.retryable,
        }
        if self.status_code is not None:
            error["status_code"] = self.status_code
        if self.retry_after_ms is not None:
            error["retry_after_ms"] = self.retry_after_ms
        if self.details:
            error["details"] = self.details

        if turn_id and self.retryable:
            next_action = "Retry aria_get_turn with the same turn_id; the ARIA turn remains server-side."
        elif ambiguous_send or (operation == "send" and self.error_type == "service_unavailable"):
            next_action = (
                "ARIA did not return a turn handle. The submission outcome may be "
                "unknown, so do not retry blindly if duplicate work matters."
            )
        elif self.error_type == "authentication_error":
            next_action = "Check that the MCP caller supplied a valid W&B token with ARIA access."
        else:
            next_action = "Correct the request or credential before trying again."

        result: Dict[str, Any] = {
            "ok": False,
            "error": error,
            "next_action": next_action,
        }
        if turn_id:
            result["turn_id"] = turn_id
        return result


def _normalize_bounded_string(
    value: Any,
    *,
    name: str,
    max_length: int,
    max_bytes: Optional[int] = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AriaAPIError(
            "invalid_request",
            f"{name} must be a non-empty string.",
            retryable=False,
        )
    normalized = value.strip()
    if len(normalized) > max_length or (max_bytes is not None and len(normalized.encode("utf-8")) > max_bytes):
        unit = "bytes" if max_bytes is not None else "characters"
        limit = max_bytes if max_bytes is not None else max_length
        raise AriaAPIError(
            "invalid_request",
            f"{name} must contain at most {limit} UTF-8 {unit}.",
            retryable=False,
        )
    return normalized


def _replace_exact_secrets(value: Any, secrets: tuple[str, ...], *, depth: int = 0) -> Any:
    """Redact request-local credentials not visible to the central sanitizer."""
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if len(secret) >= 8:
                result = result.replace(secret, "<redacted>")
        return result
    if depth >= 12:
        return "<value omitted>"
    if isinstance(value, dict):
        return {
            _replace_exact_secrets(key, secrets, depth=depth + 1): _replace_exact_secrets(
                item,
                secrets,
                depth=depth + 1,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_secrets(item, secrets, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_exact_secrets(item, secrets, depth=depth + 1) for item in value)
    return value


def normalize_aria_json_value(value: Any) -> Any:
    """Return a JSON-safe ARIA value with explicit non-finite sentinels."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "NaN"
        elif value > 0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return f"<non-finite float: {label}>"
    if isinstance(value, dict):
        return {str(key): normalize_aria_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_aria_json_value(item) for item in value]
    return value


def _estimate_response_tokens(value: Any) -> int:
    return count_tokens_conservative(
        json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _compact_response_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 5,
    max_string_chars: int = 2_048,
    max_items: int = 20,
    stats: Optional[Dict[str, int]] = None,
) -> Any:
    stats = stats if stats is not None else {}
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        omitted = len(value) - max_string_chars
        stats["strings_truncated"] = stats.get("strings_truncated", 0) + 1
        stats["string_characters_omitted"] = stats.get("string_characters_omitted", 0) + omitted
        return f"{value[:max_string_chars]}… <truncated {omitted} chars>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= max_depth:
        stats["values_omitted_by_depth"] = stats.get("values_omitted_by_depth", 0) + 1
        return "<value omitted>"
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            str(key)[:200]: _compact_response_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string_chars=max_string_chars,
                max_items=max_items,
                stats=stats,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            omitted = len(items) - max_items
            result["_truncated_keys"] = omitted
            stats["mapping_keys_omitted"] = stats.get("mapping_keys_omitted", 0) + omitted
        return result
    if isinstance(value, (list, tuple)):
        items = list(value)
        result = [
            _compact_response_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string_chars=max_string_chars,
                max_items=max_items,
                stats=stats,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            omitted = len(items) - max_items
            result.append({"_truncated_items": omitted})
            stats["list_items_omitted"] = stats.get("list_items_omitted", 0) + omitted
        return result
    return sanitize_sensitive_text(value, max_chars=max_string_chars)


def _minimal_bounded_result(result: Dict[str, Any], truncation: Dict[str, Any]) -> Dict[str, Any]:
    minimal: Dict[str, Any] = {
        "ok": result.get("ok") is not False,
        "_truncation": truncation,
    }
    for key in ("turn_id", "state", "is_terminal", "poll_after_seconds"):
        if key in result:
            minimal[key] = result[key]
    if "turn" in result:
        minimal["turn"] = {
            "_truncated": True,
            "reason": "response_token_budget",
        }
    error = result.get("error")
    if isinstance(error, dict):
        minimal["error"] = {
            key: error[key] for key in ("type", "message", "retryable", "status_code", "retry_after_ms") if key in error
        }
    minimal["next_action"] = (
        "The ARIA response exceeded the MCP response budget. Poll the turn again "
        "without include_turn or narrow the request."
    )
    return minimal


def _finalize_result(
    value: Dict[str, Any],
    *,
    extra_secrets: tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Sanitize every ARIA result and enforce the configured token budget."""
    sanitized = sanitize_sensitive_value(value, _error_context=value.get("ok") is False)
    sanitized = _replace_exact_secrets(sanitized, extra_secrets)
    sanitized = normalize_aria_json_value(sanitized)
    if not isinstance(sanitized, dict):
        sanitized = {
            "ok": False,
            "error": {
                "type": "invalid_response",
                "message": "ARIA produced an unsupported result shape.",
                "retryable": False,
            },
        }

    original_tokens = _estimate_response_tokens(sanitized)
    if original_tokens <= MAX_RESPONSE_TOKENS:
        return sanitized

    omitted_fields: list[str] = []
    candidate = dict(sanitized)
    if "turn" in candidate:
        turn = candidate.pop("turn")
        marker: Dict[str, Any] = {
            "_truncated": True,
            "reason": "response_token_budget",
        }
        if isinstance(turn, dict):
            marker.update(
                {key: turn[key] for key in ("id", "thread_id", "parent_turn_id", "state", "updated_at") if key in turn}
            )
        candidate["turn"] = marker
        omitted_fields.append("turn")

    truncation: Dict[str, Any] = {
        "applied": True,
        "reason": "response_token_budget",
        "max_tokens": MAX_RESPONSE_TOKENS,
        "original_estimated_tokens": original_tokens,
    }
    if omitted_fields:
        truncation["omitted_fields"] = omitted_fields
    candidate["_truncation"] = truncation
    if _estimate_response_tokens(candidate) <= MAX_RESPONSE_TOKENS:
        return candidate

    compaction: Dict[str, int] = {}
    candidate = _compact_response_value(candidate, stats=compaction)
    if isinstance(candidate, dict):
        if compaction:
            truncation["compaction"] = compaction
        candidate["_truncation"] = truncation
        if _estimate_response_tokens(candidate) <= MAX_RESPONSE_TOKENS:
            return candidate

    minimal = _minimal_bounded_result(sanitized, truncation)
    if _estimate_response_tokens(minimal) <= MAX_RESPONSE_TOKENS:
        return minimal

    # Extremely small operator budgets may not fit the normal metadata. Keep a
    # stable, truthful error rather than returning the oversized upstream body.
    return {
        "ok": False,
        "error": {
            "type": "response_too_large",
            "message": "The ARIA result exceeded the configured response budget.",
            "retryable": True,
        },
        "_truncation": {
            "applied": True,
            "reason": "response_token_budget",
        },
    }


def _validate_wait_seconds(wait_seconds: int) -> None:
    if not isinstance(wait_seconds, int) or isinstance(wait_seconds, bool):
        raise AriaAPIError(
            "invalid_request",
            "wait_seconds must be an integer from 0 to 30.",
            retryable=False,
        )
    if wait_seconds < 0 or wait_seconds > MAX_WAIT_SECONDS:
        raise AriaAPIError(
            "invalid_request",
            f"wait_seconds must be between 0 and {MAX_WAIT_SECONDS}.",
            retryable=False,
        )


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved = api_key or WandBApiManager.get_api_key()
    if not resolved:
        raise AriaAPIError(
            "authentication_error",
            "No W&B token is available in the current MCP request context.",
            retryable=False,
        )
    return resolved


async def _resolve_default_entity(api_key: str) -> str:
    """Resolve project-only requests through the caller's W&B account default."""
    try:
        # Keep this blocking SDK read in the shared bounded worker pool. When a
        # hosted call is cancelled or times out, InstrumentedFastMCP can then
        # retain its admission permit until the physical SDK work finishes.
        from wandb_mcp_server.instrumented_server import run_sync_in_current_tool

        entity = await run_sync_in_current_tool(lambda: WandBApiManager.get_api(api_key).default_entity)
    except Exception as exc:
        raise AriaAPIError(
            "scope_resolution_error",
            "Could not resolve the caller's default W&B entity for this project.",
            retryable=True,
            details={
                "project_requires_entity": True,
                "suggestion": "Provide entity explicitly and retry.",
                "cause": type(exc).__name__,
            },
        ) from exc
    if not entity:
        raise AriaAPIError(
            "scope_resolution_error",
            "The caller's W&B account does not expose a default entity.",
            retryable=False,
            details={
                "project_requires_entity": True,
                "suggestion": "Call list_entities_tool, then retry with entity and project.",
            },
        )
    return str(entity)


def _new_http_client(api_key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=resolve_aria_base_url(WB_AGENT_BASE_URL),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=False,
    )


def _bounded_details(value: Any, *, depth: int = 0) -> Any:
    """Keep upstream details structured without allowing an unbounded error body."""
    if depth >= 4:
        return "<truncated>"
    if isinstance(value, str):
        # Sanitize the complete scalar before shortening it. Truncating first
        # can leave a credential prefix or remove the `.svc` suffix that lets
        # the shared sanitizer recognize an internal host.
        return sanitize_sensitive_text(value, max_chars=1000)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [_bounded_details(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {
            sanitize_sensitive_text(key, max_chars=200): _bounded_details(item, depth=depth + 1)
            for key, item in list(value.items())[:20]
        }
    return sanitize_sensitive_text(value, max_chars=1000)


def _response_details(response: httpx.Response) -> Optional[Any]:
    try:
        body = response.json()
    except ValueError:
        body = response.text.strip()

    if isinstance(body, dict):
        body = body.get("detail") or body.get("error") or body
    if not body:
        return None
    return _bounded_details(body)


def _bounded_retry_after_ms(response: httpx.Response) -> Optional[int]:
    raw_value = response.headers.get("Retry-After")
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    try:
        seconds = float(raw_value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            seconds = (parsed - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds):
        return None
    if seconds < 0:
        seconds = 0
    milliseconds = int(seconds * 1_000)
    return max(MIN_RETRY_AFTER_MS, min(milliseconds, MAX_RETRY_AFTER_MS))


async def _read_bounded_response(response: httpx.Response) -> bytes:
    """Read an upstream response without buffering an unbounded service body."""
    content_encoding = response.headers.get("Content-Encoding", "")
    encodings = [item.strip().lower() for item in content_encoding.split(",") if item.strip()]
    if any(encoding != "identity" for encoding in encodings):
        # httpx decodes a complete compressed wire chunk before yielding from
        # aiter_bytes(), so checking the decoded length afterward cannot bound
        # decompression memory. Request identity and fail closed if an upstream
        # ignores it.
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned an unsupported compressed response.",
            retryable=False,
            status_code=response.status_code,
        )
    raw_length = response.headers.get("Content-Length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            content_length = None
        if content_length is not None and content_length > MAX_UPSTREAM_RESPONSE_BYTES:
            raise AriaAPIError(
                "response_too_large",
                "ARIA returned a response larger than the permitted service-response budget.",
                retryable=False,
                status_code=response.status_code,
            )

    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > MAX_UPSTREAM_RESPONSE_BYTES:
            raise AriaAPIError(
                "response_too_large",
                "ARIA returned a response larger than the permitted service-response budget.",
                retryable=False,
                status_code=response.status_code,
            )
    return bytes(body)


async def _perform_bounded_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]],
) -> httpx.Response:
    async with _outbound_limiter():
        async with client.stream(
            method,
            path,
            json=payload,
            headers={"Accept-Encoding": "identity"},
        ) as streamed:
            body = await _read_bounded_response(streamed)
            # `aiter_bytes()` yields decoded bytes. Reusing compression or transfer
            # headers on the in-memory response would make httpx decode them a
            # second time and would leave a stale wire Content-Length.
            headers = [
                (name, value)
                for name, value in streamed.headers.multi_items()
                if name.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
            ]
            return httpx.Response(
                streamed.status_code,
                headers=headers,
                content=body,
                request=streamed.request,
            )


def _http_error(response: httpx.Response, *, safe_to_retry: bool) -> AriaAPIError:
    status_code = response.status_code
    details = _response_details(response)
    retry_after_ms = _bounded_retry_after_ms(response)

    if status_code in {401, 403}:
        return AriaAPIError(
            "authentication_error",
            "ARIA rejected the W&B token or the user does not have ARIA access.",
            retryable=False,
            status_code=status_code,
            details=details,
        )
    if status_code == 404:
        return AriaAPIError(
            "turn_not_found",
            "ARIA could not find a turn visible to the current W&B user.",
            retryable=False,
            status_code=status_code,
            details=details,
        )
    if status_code == 409:
        return AriaAPIError(
            "turn_conflict",
            "ARIA rejected the turn operation in its current lifecycle state.",
            retryable=safe_to_retry,
            status_code=status_code,
            details=details,
        )
    if status_code == 422:
        return AriaAPIError(
            "invalid_request",
            "ARIA rejected the request payload.",
            retryable=False,
            status_code=status_code,
            details=details,
        )
    if status_code == 429:
        return AriaAPIError(
            "rate_limited",
            "ARIA is rate limiting requests.",
            retryable=safe_to_retry,
            status_code=status_code,
            details=details,
            retry_after_ms=retry_after_ms,
        )
    if status_code >= 500:
        return AriaAPIError(
            "service_unavailable",
            "ARIA is currently unavailable or failed to process the request.",
            retryable=safe_to_retry,
            status_code=status_code,
            details=details,
            retry_after_ms=retry_after_ms if status_code == 503 else None,
        )
    return AriaAPIError(
        "upstream_error",
        "ARIA returned an unexpected HTTP error.",
        retryable=False,
        status_code=status_code,
        details=details,
    )


async def _request_turn(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_to_retry = method == "GET"
    try:
        outer_deadline = current_tool_deadline.get()
        if outer_deadline is None:
            response = await _perform_bounded_request(client, method, path, payload=payload)
        else:
            request_seconds = outer_deadline - POLL_DEADLINE_SAFETY_SECONDS - time.monotonic()
            if request_seconds <= 0:
                raise TimeoutError
            async with asyncio.timeout(request_seconds):
                response = await _perform_bounded_request(client, method, path, payload=payload)
    except TimeoutError as exc:
        raise AriaAPIError(
            "request_timeout",
            "The ARIA request stopped before the MCP tool deadline.",
            retryable=safe_to_retry,
        ) from exc
    except httpx.TimeoutException as exc:
        raise AriaAPIError(
            "request_timeout",
            "Timed out waiting for the ARIA service.",
            retryable=safe_to_retry,
        ) from exc
    except httpx.RequestError as exc:
        raise AriaAPIError(
            "network_error",
            "Could not reach the ARIA service.",
            retryable=safe_to_retry,
        ) from exc

    if not 200 <= response.status_code < 300:
        raise _http_error(response, safe_to_retry=safe_to_retry)

    try:
        turn = response.json()
    except ValueError as exc:
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned a non-JSON response.",
            retryable=safe_to_retry,
            status_code=response.status_code,
        ) from exc

    if not isinstance(turn, dict):
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned a response without the required turn id or state.",
            retryable=safe_to_retry,
            status_code=response.status_code,
        )
    turn_id = turn.get("id")
    state = turn.get("state")
    if (
        not isinstance(turn_id, str)
        or not turn_id
        or len(turn_id) > MAX_IDENTIFIER_LENGTH
        or not isinstance(state, str)
        or state not in TURN_STATES
    ):
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned an invalid turn id or state.",
            retryable=safe_to_retry,
            status_code=response.status_code,
        )
    return turn


async def _fetch_turn(client: httpx.AsyncClient, turn_id: str) -> Dict[str, Any]:
    encoded_turn_id = quote(turn_id, safe="")
    turn = await _request_turn(client, "GET", f"/api/v1/turns/{encoded_turn_id}")
    if turn["id"] != turn_id:
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned a different turn than the one requested.",
            retryable=False,
        )
    return turn


async def _poll_turn(
    client: httpx.AsyncClient,
    turn: Dict[str, Any],
    wait_seconds: int,
) -> Dict[str, Any]:
    if wait_seconds == 0 or turn.get("state") in TERMINAL_STATES:
        return turn

    deadline = time.monotonic() + wait_seconds
    outer_deadline = current_tool_deadline.get()
    if outer_deadline is not None:
        deadline = min(deadline, outer_deadline - POLL_DEADLINE_SAFETY_SECONDS)
    while turn.get("state") not in TERMINAL_STATES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))
        if deadline - time.monotonic() <= 0:
            break
        try:
            turn = await _fetch_turn(client, str(turn["id"]))
        except AriaAPIError as exc:
            # The public dispatch timeout must remain available to serialize a
            # truthful pending result. If the request consumed that headroom,
            # return the last confirmed turn rather than racing the outer timer.
            outer_remaining = (
                None if outer_deadline is None else outer_deadline - POLL_DEADLINE_SAFETY_SECONDS - time.monotonic()
            )
            if exc.error_type == "request_timeout" and outer_remaining is not None and outer_remaining <= 0:
                break
            raise
    return turn


def _latest_assistant_response(turn: Dict[str, Any]) -> Optional[str]:
    message_sets = (turn.get("messages"), turn.get("updated_messages"))
    for messages in message_sets:
        if not isinstance(messages, list):
            continue
        # The decoded upstream response is already capped at 4 MiB, so this
        # reverse scan is finite while preserving the newest assistant answer
        # even when many tool records follow it.
        for record in reversed(messages):
            if not isinstance(record, dict):
                continue
            role = record.get("role")
            content = record.get("content")
            payload = record.get("message")
            if isinstance(payload, dict):
                role = role or payload.get("role")
                content = content or payload.get("content")
            if role == "assistant" and isinstance(content, str) and content:
                return content
    return None


def _is_nonzero_exit_code(value: Any) -> bool:
    """Recognize normalized executor exit codes without treating booleans as ints."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        try:
            return int(value.strip()) != 0
        except ValueError:
            return False
    return False


def _inspect_tool_response(record: Any) -> tuple[bool, bool]:
    """Detect tool-protocol errors and recovered executor command failures."""
    if not isinstance(record, dict) or record.get("type") != "response":
        return False, True
    if record.get("is_error") is True:
        return True, True

    # Hosted shell responses use is_error for the tool protocol itself. A
    # successfully delivered shell result can therefore have is_error=false
    # while its normalized executor outcome has a nonzero exit_code nested in
    # the response payload. Inspect structure only; never parse or expose raw
    # stdout/stderr text.
    stack: list[tuple[Any, int]] = [(record, 0)]
    seen: set[int] = set()
    nodes_inspected = 0
    complete = True
    while stack and nodes_inspected < MAX_PROGRESS_NODES_PER_RECORD:
        value, depth = stack.pop()
        nodes_inspected += 1
        if depth >= MAX_PROGRESS_NESTING_DEPTH:
            if isinstance(value, (dict, list)) and value:
                complete = False
            continue
        if isinstance(value, dict):
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)
            remaining_capacity = MAX_PROGRESS_NODES_PER_RECORD - nodes_inspected - len(stack)
            if len(value) > remaining_capacity:
                complete = False
            for index, (key, child) in enumerate(value.items()):
                if index >= remaining_capacity:
                    break
                if key == "exit_code" and _is_nonzero_exit_code(child):
                    return True, True
                if isinstance(child, (dict, list)):
                    stack.append((child, depth + 1))
        elif isinstance(value, list):
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)
            remaining_capacity = MAX_PROGRESS_NODES_PER_RECORD - nodes_inspected - len(stack)
            if len(value) > remaining_capacity:
                complete = False
            for child in value[:remaining_capacity]:
                if isinstance(child, (dict, list)):
                    stack.append((child, depth + 1))
    return False, complete and not stack


def _tool_response_has_error(record: Any) -> bool:
    return _inspect_tool_response(record)[0]


def _progress_summary(turn: Dict[str, Any]) -> Dict[str, Any]:
    messages = turn.get("messages")
    tool_calls = turn.get("tool_calls")
    state = str(turn.get("state", "unknown"))
    latest_response = _latest_assistant_response(turn)
    summary: Dict[str, Any] = {
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_call_record_count": (len(tool_calls) if isinstance(tool_calls, list) else 0),
    }
    if isinstance(tool_calls, list):
        inspected_tool_calls = tool_calls[-MAX_PROGRESS_TOOL_RECORDS:]
        inspections = [_inspect_tool_response(record) for record in inspected_tool_calls]
        tool_error_count = sum(has_error for has_error, _ in inspections)
        summary["tool_call_records_inspected"] = len(inspected_tool_calls)
        summary["tool_error_count_exact"] = len(inspected_tool_calls) == len(tool_calls) and all(
            complete for _, complete in inspections
        )
        if tool_error_count:
            summary["tool_error_count"] = tool_error_count
            qualifier = "" if summary["tool_error_count_exact"] else "at least "
            summary["internal_error_note"] = (
                f"ARIA encountered {qualifier}{tool_error_count} internal tool or executor error(s) "
                "in the bounded progress window and continued working."
            )
    if isinstance(tool_calls, list) and tool_calls:
        latest = tool_calls[-1]
        if isinstance(latest, dict):
            latest_activity = {
                key: latest[key] for key in ("type", "name", "call_id", "is_error", "timestamp") if key in latest
            }
            if _tool_response_has_error(latest):
                latest_activity["recovered_error"] = True
            summary["latest_tool_activity"] = latest_activity

    if state == "queued":
        summary.update(phase="queued", status_text="ARIA accepted the turn and is waiting for execution.")
    elif state == "in_progress" and latest_response:
        summary.update(
            phase="responding",
            status_text="ARIA has produced a partial response but is still working; keep polling.",
        )
    elif state == "in_progress" and summary.get("latest_tool_activity"):
        activity = summary["latest_tool_activity"]
        tool_name = activity.get("name") or "an internal tool"
        if activity.get("type") == "response" and activity.get("recovered_error"):
            status_text = f"ARIA's latest {tool_name} attempt failed internally; ARIA is still working."
        elif activity.get("type") == "response":
            status_text = f"ARIA received a result from {tool_name} and is still working."
        else:
            status_text = f"ARIA is running {tool_name}."
        summary.update(phase="working", status_text=status_text)
    elif state == "in_progress":
        summary.update(phase="working", status_text="ARIA is working on the turn.")
    elif state == "completed":
        summary.update(phase="completed", status_text="ARIA completed the turn.")
    elif state == "errored":
        summary.update(phase="errored", status_text="ARIA stopped with an execution error.")
    elif state == "cancelled":
        summary.update(phase="cancelled", status_text="The ARIA turn was cancelled.")
    else:
        summary.update(phase=state, status_text=f"ARIA reported state '{state}'.")
    return summary


def _turn_result(turn: Dict[str, Any], *, include_turn: bool = False) -> Dict[str, Any]:
    state = str(turn["state"])
    turn_id = str(turn["id"])
    is_terminal = state in TERMINAL_STATES

    if state == "completed":
        next_action = (
            "Use latest_response as ARIA's answer. To continue this conversation, "
            "call aria_send_message with parent_turn_id set to this turn_id."
        )
    elif state == "errored":
        next_action = (
            "Inspect error_info. Start a new turn or retry the task only if the "
            "reported execution failure is transient."
        )
    elif state == "cancelled":
        next_action = "This turn was cancelled; start a new turn if the work is still needed."
    else:
        next_action = "Call aria_get_turn again with this turn_id; use wait_seconds up to 30 for a bounded wait."

    result = {
        "ok": True,
        "turn_id": turn_id,
        "thread_id": turn.get("thread_id"),
        "parent_turn_id": turn.get("parent_turn_id"),
        "scope": {
            "entity": turn.get("wandb_entity"),
            "project": turn.get("wandb_project"),
        },
        "state": state,
        "is_terminal": is_terminal,
        "updated_at": turn.get("updated_at"),
        "poll_after_seconds": None if is_terminal else 2,
        "latest_response": _latest_assistant_response(turn),
        "error_info": turn.get("error_info"),
        "agent_questions": turn.get("agent_questions"),
        "permission_requests": turn.get("permission_requests"),
        "progress": _progress_summary(turn),
        "next_action": next_action,
    }
    if include_turn:
        result["turn"] = turn
    return result


async def send_aria_message(
    message: str,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    parent_turn_id: Optional[str] = None,
    wait_seconds: int = 0,
    include_turn: bool = False,
    *,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Create a root or continuation ARIA turn and optionally poll it briefly."""
    owns_client = client is None
    resolved_api_key: Optional[str] = None
    try:
        message = _normalize_bounded_string(
            message,
            name="message",
            max_length=MAX_MESSAGE_BYTES,
            max_bytes=MAX_MESSAGE_BYTES,
        )
        _validate_wait_seconds(wait_seconds)

        if entity is not None:
            entity = _normalize_bounded_string(
                entity,
                name="entity",
                max_length=MAX_SCOPE_LENGTH,
            )
        if project is not None:
            project = _normalize_bounded_string(
                project,
                name="project",
                max_length=MAX_SCOPE_LENGTH,
            )

        if parent_turn_id is not None:
            parent_turn_id = _normalize_bounded_string(
                parent_turn_id,
                name="parent_turn_id",
                max_length=MAX_IDENTIFIER_LENGTH,
            )
            if entity is not None or project is not None:
                raise AriaAPIError(
                    "invalid_request",
                    "entity and project must be omitted when parent_turn_id is set.",
                    retryable=False,
                )

        resolved_api_key = _resolve_api_key(api_key)
        if parent_turn_id is None and project is not None and entity is None:
            entity = _normalize_bounded_string(
                await _resolve_default_entity(resolved_api_key),
                name="resolved entity",
                max_length=MAX_SCOPE_LENGTH,
            )
        if client is None:
            client = _new_http_client(resolved_api_key)

        payload: Dict[str, Any] = {"user_prompt": message}
        if parent_turn_id is not None:
            payload["parent_turn_id"] = parent_turn_id
        else:
            if entity is not None:
                payload["entity"] = entity
            if project is not None:
                payload["project"] = project

        logger.info(
            "Submitting ARIA turn: continuation=%s scoped=%s wait=%ss",
            parent_turn_id is not None,
            entity is not None or project is not None,
            wait_seconds,
        )
        turn = await _request_turn(client, "POST", "/api/v1/turns", payload=payload)
        try:
            turn = await _poll_turn(client, turn, wait_seconds)
        except AriaAPIError as exc:
            # The non-idempotent POST was confirmed and returned a durable
            # handle. A later GET failure must never erase that handle or imply
            # that the submission outcome is unknown: the caller can safely
            # resume polling the same turn instead of creating duplicate work.
            return _finalize_result(
                exc.as_result(turn_id=str(turn["id"]), operation="get"),
                extra_secrets=(resolved_api_key,),
            )
        return _finalize_result(
            _turn_result(turn, include_turn=include_turn),
            extra_secrets=(resolved_api_key,),
        )
    except AriaAPIError as exc:
        logger.warning("ARIA send failed: %s", exc.error_type)
        return _finalize_result(
            exc.as_result(operation="send"),
            extra_secrets=(resolved_api_key,) if resolved_api_key else (),
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def get_aria_turn(
    turn_id: str,
    wait_seconds: int = 0,
    include_turn: bool = False,
    *,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Fetch an ARIA turn and optionally poll until terminal or a short deadline."""
    owns_client = client is None
    resolved_api_key: Optional[str] = None
    try:
        turn_id = _normalize_bounded_string(
            turn_id,
            name="turn_id",
            max_length=MAX_IDENTIFIER_LENGTH,
        )
        _validate_wait_seconds(wait_seconds)
        resolved_api_key = _resolve_api_key(api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        turn = await _fetch_turn(client, turn_id)
        turn = await _poll_turn(client, turn, wait_seconds)
        return _finalize_result(
            _turn_result(turn, include_turn=include_turn),
            extra_secrets=(resolved_api_key,),
        )
    except AriaAPIError as exc:
        logger.warning("ARIA get failed: %s", exc.error_type)
        safe_turn_id = turn_id if isinstance(turn_id, str) and len(turn_id) <= MAX_IDENTIFIER_LENGTH else None
        return _finalize_result(
            exc.as_result(turn_id=safe_turn_id, operation="get"),
            extra_secrets=(resolved_api_key,) if resolved_api_key else (),
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def _fetch_turn_as_result(
    client: httpx.AsyncClient,
    turn_id: str,
    *,
    include_turn: bool,
    extra_secrets: tuple[str, ...],
) -> Dict[str, Any]:
    try:
        turn = await _fetch_turn(client, turn_id)
        return _finalize_result(
            _turn_result(turn, include_turn=include_turn),
            extra_secrets=extra_secrets,
        )
    except AriaAPIError as exc:
        return _finalize_result(
            exc.as_result(turn_id=turn_id, operation="get"),
            extra_secrets=extra_secrets,
        )


async def _fetch_batch_wave(
    client: httpx.AsyncClient,
    turn_ids: List[str],
    *,
    include_turn: bool,
    extra_secrets: tuple[str, ...],
) -> List[Dict[str, Any]]:
    async def fetch_one(turn_id: str) -> Dict[str, Any]:
        return await _fetch_turn_as_result(
            client,
            turn_id,
            include_turn=include_turn,
            extra_secrets=extra_secrets,
        )

    return list(await asyncio.gather(*(fetch_one(turn_id) for turn_id in turn_ids)))


async def get_aria_turns(
    turn_ids: List[str],
    wait_seconds: int = 0,
    include_turn: bool = False,
    *,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Fetch and briefly poll several ARIA turns concurrently."""
    owns_client = client is None
    resolved_api_key: Optional[str] = None
    try:
        _validate_wait_seconds(wait_seconds)
        if not isinstance(turn_ids, list) or not turn_ids:
            raise AriaAPIError(
                "invalid_request",
                "turn_ids must be a non-empty list.",
                retryable=False,
            )
        if len(turn_ids) > MAX_BATCH_TURNS:
            raise AriaAPIError(
                "invalid_request",
                f"turn_ids may contain at most {MAX_BATCH_TURNS} turns.",
                retryable=False,
            )

        normalized_ids: List[str] = []
        for turn_id in turn_ids:
            normalized_ids.append(
                _normalize_bounded_string(
                    turn_id,
                    name="Every turn_id",
                    max_length=MAX_IDENTIFIER_LENGTH,
                )
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise AriaAPIError(
                "invalid_request",
                "turn_ids must not contain duplicates.",
                retryable=False,
            )

        resolved_api_key = _resolve_api_key(api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        deadline = time.monotonic() + wait_seconds
        outer_deadline = current_tool_deadline.get()
        if outer_deadline is not None:
            deadline = min(deadline, outer_deadline - POLL_DEADLINE_SAFETY_SECONDS)
        pending_ids = list(normalized_ids)
        results_by_id: Dict[str, Dict[str, Any]] = {}
        get_requests_made = 0
        request_budget_exhausted = False
        wave_number = 0

        while pending_ids:
            if wave_number > 0 and deadline - time.monotonic() <= 0:
                break
            if get_requests_made + len(pending_ids) > MAX_BATCH_GET_REQUESTS:
                request_budget_exhausted = True
                break

            fetched = await _fetch_batch_wave(
                client,
                pending_ids,
                include_turn=include_turn,
                extra_secrets=(resolved_api_key,),
            )
            get_requests_made += len(pending_ids)
            wave_number += 1
            for turn_id, result in zip(pending_ids, fetched):
                previous = results_by_id.get(turn_id)
                timed_out_at_outer_deadline = (
                    previous is not None
                    and previous.get("ok") is True
                    and previous.get("is_terminal") is False
                    and result.get("ok") is False
                    and result.get("error", {}).get("type") == "request_timeout"
                    and outer_deadline is not None
                    and time.monotonic() >= deadline
                )
                if timed_out_at_outer_deadline:
                    continue
                results_by_id[turn_id] = result

            pending_ids = [
                turn_id
                for turn_id in pending_ids
                if (results_by_id[turn_id].get("ok") is True and not results_by_id[turn_id].get("is_terminal"))
                or (
                    results_by_id[turn_id].get("ok") is False
                    and results_by_id[turn_id].get("error", {}).get("retryable") is True
                    and results_by_id[turn_id].get("error", {}).get("type") not in _OVERLOAD_ERROR_TYPES
                )
            ]
            remaining = deadline - time.monotonic()
            if wait_seconds == 0 or not pending_ids or remaining <= 0:
                break
            await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))

        ordered_results = [results_by_id[turn_id] for turn_id in normalized_ids]
        error_count = sum(result.get("ok") is False for result in ordered_results)
        terminal_count = sum(
            result.get("ok") is True and result.get("is_terminal") is True for result in ordered_results
        )
        still_pending = [
            turn_id
            for turn_id, result in zip(normalized_ids, ordered_results)
            if result.get("ok") is True and result.get("is_terminal") is False
        ]
        failed_turn_ids = [
            turn_id for turn_id, result in zip(normalized_ids, ordered_results) if result.get("ok") is False
        ]

        if error_count == len(ordered_results):
            retryable = all(result.get("error", {}).get("retryable") is True for result in ordered_results)
            return _finalize_result(
                {
                    "ok": False,
                    "error": {
                        "type": "batch_lookup_failed",
                        "message": "ARIA could not retrieve any of the requested turns.",
                        "retryable": retryable,
                        "details": {"results": ordered_results},
                    },
                    "turn_ids": normalized_ids,
                    "get_requests_made": get_requests_made,
                    "get_request_limit": MAX_BATCH_GET_REQUESTS,
                    "request_budget_exhausted": request_budget_exhausted,
                    "next_action": (
                        "Retry aria_get_turns with the same turn_ids."
                        if retryable
                        else "Inspect each turn error and correct credentials or turn IDs."
                    ),
                },
                extra_secrets=(resolved_api_key,),
            )

        return _finalize_result(
            {
                "ok": True,
                "requested_count": len(normalized_ids),
                "terminal_count": terminal_count,
                "pending_count": len(still_pending),
                "error_count": error_count,
                "pending_turn_ids": still_pending,
                "failed_turn_ids": failed_turn_ids,
                "get_requests_made": get_requests_made,
                "get_request_limit": MAX_BATCH_GET_REQUESTS,
                "request_budget_exhausted": request_budget_exhausted,
                "results": ordered_results,
                "next_action": (
                    "Call aria_get_turns again with pending_turn_ids."
                    if still_pending
                    else "All retrievable turns are terminal; continue completed conversations with aria_send_message."
                ),
            },
            extra_secrets=(resolved_api_key,),
        )
    except AriaAPIError as exc:
        logger.warning("ARIA batch get failed: %s", exc.error_type)
        return _finalize_result(
            exc.as_result(operation="get"),
            extra_secrets=(resolved_api_key,) if resolved_api_key else (),
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()
