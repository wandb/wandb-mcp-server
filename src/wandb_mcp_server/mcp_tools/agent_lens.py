"""MCP tools for the **Agent Lens** official v1, Insights, tags, and guidelines APIs.

Agent Lens classifies agent turns into intent/failure categories and clusters,
runs 16-rule OTel instrumentation checks, stores project judge guidelines, and
manages human- and judge-applied turn/conversation tags.

Official v1 endpoints (``/api/v1/tags*``) use the fixed ``{"data": ...}``
envelope and ``{"message", "param"}`` error format; remaining live Insights,
tag-assignment, guideline, and summarization primitives route through ``/api/*``
on the same Bearer-authenticated origin with ``X-Wandb-Entity`` and
``X-Wandb-Project`` headers.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Literal, Optional, Sequence
from urllib.parse import quote
from uuid import UUID

import httpx

from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.api_client import WandBApiManager, WandBServerBusy
from wandb_mcp_server.config import (
    AGENT_LENS_API_PREFIX,
    AGENT_LENS_V1_API_PREFIX,
    MAX_ACCUMULATED_BYTES,
    MAX_RESPONSE_TOKENS,
    resolve_agent_lens_base_url,
    structured_error,
)
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.tool_results import ToolErrorJSON, ToolErrorPayload, tool_error_json
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

_agent_lens_http_active: ContextVar[bool] = ContextVar("agent_lens_http_active", default=False)


class _SuppressAgentLensHTTPLogFilter(logging.Filter):
    """Keep caller-derived Agent Lens URLs out of HTTP client logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _agent_lens_http_active.get()


for _http_logger_name in (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
):
    _http_logger = logging.getLogger(_http_logger_name)
    if not any(isinstance(item, _SuppressAgentLensHTTPLogFilter) for item in _http_logger.filters):
        _http_logger.addFilter(_SuppressAgentLensHTTPLogFilter())

LATEST_WEEK_PATH = "/insights/latest-week"
CLUSTERING_STATUS_PATH = "/insights/clustering-status"
CLUSTERS_PATH = "/insights/clusters"
FAILURE_ATTRIBUTIONS_PATH = "/insights/failure-attributions/query"
TAGS_PATH = "/tags"
CONVERSATION_TAGS_PATH = "/conversation-tags"
CONVERSATION_TAGS_QUERY_PATH = "/conversation-tags/query"
TAGGED_CONVERSATIONS_PATH = "/conversation-tags/conversations/query"
TAG_DISTRIBUTION_PATH = "/conversation-tags/distribution"
GUIDELINES_PATH = "/guidelines"


def _breakdowns_path(signature_type: str) -> str:
    return f"/insights/{quote(signature_type, safe='')}/breakdowns"


def _volume_path(signature_type: str) -> str:
    return f"/insights/{quote(signature_type, safe='')}/volume"


def _category_examples_path(signature_type: str, category_id: str) -> str:
    """Build the drilldown path, percent-encoding the caller-supplied category."""
    return f"/insights/{quote(signature_type, safe='')}/categories/{quote(category_id, safe='')}/example-turns"


def _tag_path(tag_id: str) -> str:
    return f"/tags/{quote(tag_id, safe='')}"


def _tag_assignments_path(tag_id: str) -> str:
    return f"/tags/{quote(tag_id, safe='')}/assignments"


def _guideline_path(guideline_id: str) -> str:
    return f"/guidelines/{quote(guideline_id, safe='')}"


def _conversation_summarize_path(conversation_id: str) -> str:
    return f"/conversations/{quote(conversation_id, safe='')}/summarize"


_REQUEST_TIMEOUT_SECONDS = 30
MAX_AGENT_LENS_RESPONSE_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_DEFAULT_RETRY_AFTER_MS = 1_000
_MAX_RETRY_AFTER_MS = 60_000

MAX_INSIGHTS_WINDOW_DAYS = 30
MAX_CONVERSATION_IDS = 5000
MAX_TAG_IDS = 100
MAX_CATEGORY_IDS = 100
MAX_TOPIC_IDS = 20
MAX_TRACE_IDS = 500
MAX_EXAMPLE_LIMIT = 50
MAX_PER_TOPIC_LIMIT = 25
MAX_TAG_ASSIGNMENTS_LIMIT = 500
MAX_TIME_BUCKET_SECONDS = 86400
MIN_VOLUME_BUCKET_SECONDS = 60
MAX_VOLUME_BUCKETS = 1440
MAX_TAG_CRITERIA_CHARS = 1600
MAX_GUIDELINE_CHARS = 4000
_STRATIFIED_CONCURRENCY = 5

_TAG_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_INSIGHTS_FILTER_FIELDS = frozenset(
    {
        "agent_name",
        "language",
        "intent_category",
        "intent_sentiment",
        "failure_category",
        "failure_severity",
        "intent_topic_id",
        "failure_topic_id",
    }
)


def _drop_none(values: Dict[str, Any], preserve_null_keys: Sequence[str] = ()) -> Dict[str, Any]:
    """Drop None values except keys where explicit JSON null clears a server field."""
    keep_null = set(preserve_null_keys)
    return {k: v for k, v in values.items() if v is not None or k in keep_null}


class _ResponseBoundaryError(ValueError):
    """A bounded, externally safe Agent Lens response failure."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _compact_json(payload: Any) -> str:
    serializer = tool_error_json if isinstance(payload, ToolErrorPayload) else json.dumps
    return serializer(payload, default=str, ensure_ascii=False, separators=(",", ":"))


def _response_tokens(payload: Any) -> int:
    return count_tokens_conservative(_compact_json(payload))


def _truncation_metadata(field: str, returned: int, original: int) -> Dict[str, Any]:
    return {
        "applied": True,
        "field": field,
        "returned": returned,
        "original": original,
        "note": (
            f"Response truncated to fit the {MAX_RESPONSE_TOKENS}-token budget; dropped "
            f"{original - returned} '{field}' item(s). Narrow with a tighter time range, a "
            "smaller limit, or more specific filters."
        ),
    }


def _truncate_response(payload: Any) -> Any:
    """Trim the response's primary list so the serialized payload fits the budget."""
    if not isinstance(payload, dict):
        return payload
    if _response_tokens(payload) <= MAX_RESPONSE_TOKENS:
        return payload

    data = payload.get("data")
    if isinstance(data, list) and data and payload.get("next_cursor") is not None:
        cursor_error = {
            "error": "agent_lens_response_too_large",
            "message": (
                "The Agent Lens page exceeded the response budget. Retry with a lower limit "
                "so no paginated rows are skipped."
            ),
        }
        if _response_tokens(cursor_error) <= MAX_RESPONSE_TOKENS:
            return ToolErrorPayload(cursor_error)
        return ToolErrorPayload({"error": "agent_lens_response_too_large"})

    container: Optional[Dict[str, Any]] = None
    key = "data"
    nested_side: Optional[str] = None
    if isinstance(data, list) and data:
        container, key = payload, "data"
    elif isinstance(data, dict):
        if isinstance(data.get("buckets"), list) and data["buckets"]:
            container, key = data, "buckets"
        elif isinstance(data.get("categories"), list) and data["categories"]:
            container, key = data, "categories"
        else:
            for side in ("failure", "intent"):
                sub = data.get(side)
                if isinstance(sub, dict) and isinstance(sub.get("categories"), list) and sub["categories"]:
                    container, key, nested_side = sub, "categories", side
                    break

    if container is not None:
        items = container[key]
        original = len(items)
        kept = list(items)

        def rebuilt(candidate: List[Any]) -> Dict[str, Any]:
            if container is payload:
                result = {**payload, key: candidate}
            elif nested_side is not None and isinstance(data, dict):
                result = {**payload, "data": {**data, nested_side: {**container, key: candidate}}}
            elif isinstance(data, dict):
                result = {**payload, "data": {**data, key: candidate}}
            else:
                result = {**payload}
            field_label = f"{nested_side}.{key}" if nested_side else key
            result["_truncation"] = _truncation_metadata(field_label, len(candidate), original)
            return result

        while kept and _response_tokens(rebuilt(kept)) > MAX_RESPONSE_TOKENS:
            kept = kept[: -max(1, len(kept) // 10)]

        result = rebuilt(kept)
        if _response_tokens(result) <= MAX_RESPONSE_TOKENS:
            return result

    fallback: Dict[str, Any] = {
        "error": "agent_lens_response_too_large",
        "message": "The Agent Lens response exceeded the configured response budget.",
        "_truncation": {"applied": True, "reason": "response_token_budget"},
    }
    if _response_tokens(fallback) <= MAX_RESPONSE_TOKENS:
        return ToolErrorPayload(fallback)
    minimal = {"error": "agent_lens_response_too_large"}
    return ToolErrorPayload(minimal) if _response_tokens(minimal) <= MAX_RESPONSE_TOKENS else ToolErrorPayload({})


def _project_response(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove legacy message fields if returned by older Agent Lens servers."""
    if tool_name != "list_category_example_turns":
        return payload
    data = payload.get("data")
    if not isinstance(data, list):
        return payload
    projected = [
        {key: value for key, value in row.items() if key not in {"agent_message", "message"}}
        if isinstance(row, dict)
        else row
        for row in data
    ]
    return {**payload, "data": projected}


def _absolute_request_deadline() -> float:
    """Capture one deadline for connect, headers, and the complete response body."""
    request_cap = time.monotonic() + float(_REQUEST_TIMEOUT_SECONDS)
    tool_deadline = current_tool_deadline.get()
    return request_cap if tool_deadline is None else min(request_cap, tool_deadline)


def _remaining_request_timeout(deadline: float) -> float:
    """Return the remaining time under the captured absolute deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Agent Lens tool deadline elapsed")
    return max(0.001, remaining)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Agent Lens tool deadline elapsed")


def _new_agent_lens_client(timeout: float) -> httpx.AsyncClient:
    """Build one non-redirecting client for a caller-scoped Agent Lens request."""
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


async def _bounded_response_bytes(response: httpx.Response, deadline: float) -> bytes:
    """Read at most the reviewed byte cap before any JSON decoding occurs."""
    byte_limit = min(MAX_AGENT_LENS_RESPONSE_BYTES, MAX_ACCUMULATED_BYTES)
    encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding != "identity":
        raise _ResponseBoundaryError("unsupported_encoding", "The Agent Lens API returned an invalid response.")

    raw_length = response.headers.get("Content-Length")
    expected_length: int | None = None
    if raw_length is not None:
        if not raw_length.isascii() or not raw_length.isdecimal():
            raise _ResponseBoundaryError("invalid_length", "The Agent Lens API returned an invalid response.")
        normalized = raw_length.lstrip("0") or "0"
        if len(normalized) > len(str(byte_limit)) or int(normalized) > byte_limit:
            raise _ResponseBoundaryError(
                "response_too_large",
                "The Agent Lens API response exceeded the download limit.",
            )
        expected_length = int(normalized)

    raw = bytearray()
    async for chunk in response.aiter_raw(chunk_size=_READ_CHUNK_BYTES):
        _check_deadline(deadline)
        if not isinstance(chunk, (bytes, bytearray)):
            raise _ResponseBoundaryError("malformed_response", "The Agent Lens API returned an invalid response.")
        if len(raw) + len(chunk) > byte_limit:
            raise _ResponseBoundaryError(
                "response_too_large",
                "The Agent Lens API response exceeded the download limit.",
            )
        raw.extend(chunk)
    if expected_length is not None and len(raw) != expected_length:
        raise _ResponseBoundaryError("inconsistent_length", "The Agent Lens API returned an invalid response.")
    return bytes(raw)


async def _bounded_json_response(response: httpx.Response, deadline: float) -> Dict[str, Any]:
    raw = await _bounded_response_bytes(response, deadline)
    _check_deadline(deadline)
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_non_finite_json)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise _ResponseBoundaryError(
            "malformed_response", "The Agent Lens API returned an invalid response."
        ) from error
    if not isinstance(payload, dict):
        raise _ResponseBoundaryError("malformed_response", "The Agent Lens API returned an invalid response.")
    return payload


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _retry_after_ms(value: object) -> int:
    """Parse Retry-After without consulting an unbounded response body."""
    if value is None:
        return _DEFAULT_RETRY_AFTER_MS
    raw = str(value).strip()
    try:
        seconds = float(raw)
        if not math.isfinite(seconds):
            raise ValueError
        seconds = max(0.0, seconds)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            seconds = max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return _DEFAULT_RETRY_AFTER_MS
    return min(_MAX_RETRY_AFTER_MS, max(_DEFAULT_RETRY_AFTER_MS, round(seconds * 1_000)))


def parse_timestamp(value: str, field: str) -> datetime:
    """Parse an RFC 3339 bound the way Agent Lens does, normalized to UTC."""
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be an RFC 3339 timestamp")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError(f"{field} must be an RFC 3339 timestamp, for example 2026-09-01T00:00:00Z") from None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _validated_window(start_at: str, end_at: str) -> Dict[str, object]:
    """Check the range locally against the server's own bounds before sending."""
    start = parse_timestamp(start_at, "start_at")
    end = parse_timestamp(end_at, "end_at")
    if end <= start:
        raise ValueError("start_at and end_at must define a nonempty time range")
    if end - start > timedelta(days=MAX_INSIGHTS_WINDOW_DAYS):
        raise ValueError(f"time range must not exceed {MAX_INSIGHTS_WINDOW_DAYS} days")
    return {"start_at": start_at, "end_at": end_at}


def _bounded_list(values: Sequence[str], maximum: int, field: str) -> List[str]:
    """Reject an oversized or blank-containing list locally rather than trading a 422 round-trip."""
    items = [str(value).strip() for value in values]
    if not items or any(not item for item in items):
        raise ValueError(f"{field} must contain at least one non-empty value")
    if len(items) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} values (got {len(items)})")
    return items


def _validated_uuid(value: str, field: str) -> str:
    try:
        return str(UUID(str(value).strip()))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{field} must be a valid UUID") from None


def _bounded_uuid_list(values: Sequence[str], maximum: int, field: str) -> List[str]:
    """Validate opaque UUID identifiers without echoing a rejected value."""
    items = _bounded_list(values, maximum, field)
    try:
        return [str(UUID(item)) for item in items]
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{field} must contain only UUIDs") from None


def _serialize_insights_filters(filters: Optional[Sequence[Dict[str, Any]]]) -> Optional[str]:
    """Validate and serialize Agent Lens Insights turn-level filters."""
    if not filters:
        return None
    seen_fields: set[str] = set()
    normalized: List[Dict[str, Any]] = []
    for entry in filters:
        if not isinstance(entry, dict):
            raise ValueError("each entry in filters must be an object with 'field' and 'values'")
        field = str(entry.get("field") or "").strip()
        if field not in _INSIGHTS_FILTER_FIELDS:
            allowed = ", ".join(sorted(_INSIGHTS_FILTER_FIELDS))
            raise ValueError(f"filters field '{field}' must be one of: {allowed}")
        if field in seen_fields:
            raise ValueError(f"filters must not repeat field '{field}'")
        seen_fields.add(field)
        raw_values = entry.get("values")
        if not isinstance(raw_values, (list, tuple)):
            raise ValueError(f"filters[{field}].values must be a list of strings")
        values = _bounded_list(raw_values, 100, f"filters[{field}].values")
        item: Dict[str, Any] = {"field": field, "values": values}
        if "exclude" in entry and entry["exclude"] is not None:
            if not isinstance(entry["exclude"], bool):
                raise ValueError(f"filters[{field}].exclude must be a boolean")
            item["exclude"] = entry["exclude"]
        normalized.append(item)
    return json.dumps(normalized, separators=(",", ":"))


def _tool_timeout_result(ctx: Any) -> str:
    """Return one canonical timeout envelope and telemetry classification."""
    ctx.mark_error("tool_timeout")
    return _compact_json(
        structured_error(
            "tool_timeout",
            "The Agent Lens request exceeded the MCP tool deadline.",
            retryable=True,
        )
    )


async def _agent_lens_request(
    tool_name: str,
    method: str,
    path: str,
    entity_name: str,
    project_name: str,
    track_params: Dict[str, object],
    params: Optional[Dict[str, object]] = None,
    body: Optional[Dict[str, object]] = None,
    *,
    api_version: Literal["v1", "internal"] = "internal",
    preserve_null_keys: Sequence[str] = (),
) -> str:
    """Call one Agent Lens endpoint and return a JSON string trimmed to the token budget."""
    result = await fetch_agent_lens_json(
        path,
        tool_name=tool_name,
        method=method,
        entity_name=entity_name,
        project_name=project_name,
        track_params=track_params,
        params=params,
        body=body,
        api_version=api_version,
        preserve_null_keys=preserve_null_keys,
    )
    if isinstance(result, ToolErrorJSON):
        return result

    return _compact_json(_truncate_response(_project_response(tool_name, result)))


async def fetch_agent_lens_json(
    path: str,
    *,
    tool_name: str,
    method: str,
    entity_name: str,
    project_name: str,
    track_params: Dict[str, object],
    params: Optional[Dict[str, object]] = None,
    body: Optional[Dict[str, object]] = None,
    api_version: Literal["v1", "internal"] = "internal",
    preserve_null_keys: Sequence[str] = (),
) -> Dict[str, object] | ToolErrorJSON:
    """Call one Agent Lens endpoint and return the decoded payload or an error envelope."""
    try:
        base_url = resolve_agent_lens_base_url()
    except ValueError as error:
        return tool_error_json(structured_error("agent_lens_not_configured", str(error)))

    api_key = WandBApiManager.get_api_key()
    if not api_key:
        logger.error("W&B API key not found in context or environment.")
        return tool_error_json(
            structured_error("auth_required", "A W&B API key is required to query the Agent Lens API.")
        )

    prefix = AGENT_LENS_V1_API_PREFIX if api_version == "v1" else AGENT_LENS_API_PREFIX
    url = f"{base_url}{prefix}{path}"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Authorization": f"Bearer {api_key}",
        "X-Wandb-Entity": entity_name,
        "X-Wandb-Project": project_name,
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(_drop_none(body, preserve_null_keys=preserve_null_keys))

    with track_tool_execution(tool_name, None, track_params) as ctx:
        try:
            request_deadline = _absolute_request_deadline()
            request_timeout = _remaining_request_timeout(request_deadline)
            log_token = _agent_lens_http_active.set(True)
            try:
                async with asyncio.timeout(request_timeout):
                    async with _new_agent_lens_client(request_timeout) as client:
                        async with client.stream(
                            method,
                            url,
                            headers=headers,
                            params=params or None,
                            content=data,
                            follow_redirects=False,
                        ) as response:
                            _check_deadline(request_deadline)
                            if response.status_code in {401, 403}:
                                ctx.mark_error(f"http_{response.status_code}")
                                return _compact_json(
                                    structured_error(
                                        "agent_lens_forbidden",
                                        "Agent Lens rejected the W&B credential for this project. Confirm the "
                                        "credential has access and that Agent Lens is enabled for the project.",
                                        status_code=response.status_code,
                                    )
                                )
                            if response.status_code == 404:
                                not_found_detail: Optional[str] = None
                                if api_version == "v1" or method in {"POST", "PATCH", "DELETE"}:
                                    try:
                                        nf_payload = await _bounded_json_response(response, request_deadline)
                                        extracted = _optional_detail(nf_payload)
                                        if extracted:
                                            not_found_detail = extracted
                                    except _ResponseBoundaryError:
                                        not_found_detail = None
                                if not_found_detail:
                                    ctx.mark_error("http_404")
                                    return _compact_json(
                                        structured_error(
                                            "agent_lens_invalid_request",
                                            f"Agent Lens resource not found: {not_found_detail}",
                                            status_code=404,
                                        )
                                    )
                                ctx.mark_error("agent_lens_unavailable")
                                return _compact_json(
                                    structured_error(
                                        "agent_lens_unavailable",
                                        "The configured Agent Lens origin does not expose this endpoint (404); it may "
                                        "predate this API or have the feature disabled.",
                                        status_code=404,
                                    )
                                )
                            if response.status_code in {400, 409, 422}:
                                ctx.mark_error(f"http_{response.status_code}")
                                try:
                                    validation_payload = await _bounded_json_response(response, request_deadline)
                                    detail = _detail(validation_payload)
                                except _ResponseBoundaryError:
                                    detail = "the request did not satisfy the API schema"
                                return _compact_json(
                                    structured_error(
                                        "agent_lens_invalid_request",
                                        f"Agent Lens rejected the request arguments: {detail}",
                                        status_code=response.status_code,
                                    )
                                )
                            if response.status_code == 204:
                                await _bounded_response_bytes(response, request_deadline)
                                _check_deadline(request_deadline)
                                result = {"data": {"ok": True}}
                            elif response.status_code in {200, 201}:
                                result = await _bounded_json_response(response, request_deadline)
                                _check_deadline(request_deadline)
                            else:
                                ctx.mark_error(f"http_{response.status_code}")
                                if response.status_code in {429, 503}:
                                    raise WandBServerBusy(
                                        status_code=response.status_code,
                                        retry_after_ms=_retry_after_ms(
                                            response.headers.get("Retry-After") or response.headers.get("retry-after")
                                        ),
                                    )
                                return _compact_json(
                                    structured_error(
                                        "agent_lens_query_failed",
                                        f"The Agent Lens API returned HTTP {response.status_code}.",
                                        status_code=response.status_code,
                                    )
                                )
            finally:
                _agent_lens_http_active.reset(log_token)
        except WandBServerBusy:
            raise
        except (TimeoutError, httpx.TimeoutException):
            return _tool_timeout_result(ctx)
        except _ResponseBoundaryError as error:
            ctx.mark_error(error.reason)
            return _compact_json(
                structured_error("agent_lens_query_failed", sanitize_sensitive_text(error, max_chars=512))
            )
        except httpx.RequestError as error:
            ctx.mark_error(type(error).__name__)
            return _compact_json(structured_error("agent_lens_query_failed", "The Agent Lens request failed."))
        except Exception as error:
            logger.error("Agent Lens request failed (%s)", type(error).__name__)
            ctx.mark_error(type(error).__name__)
            return _compact_json(structured_error("agent_lens_query_failed", "The Agent Lens request failed."))

    return result


def _optional_detail(payload: Dict[str, Any]) -> Optional[str]:
    """Extract official v1 {'message', 'param'} or Huma {'detail', 'title'} details."""
    message = payload.get("message")
    param = payload.get("param")
    if isinstance(message, str) and message.strip():
        prefix = f"{param}: " if isinstance(param, str) and param.strip() else ""
        return sanitize_sensitive_text(f"{prefix}{message.strip()}", max_chars=512)
    detail = payload.get("detail") or payload.get("title")
    if detail is not None and str(detail).strip():
        return sanitize_sensitive_text(detail, max_chars=512)
    return None


def _detail(payload: Dict[str, Any]) -> str:
    """Pull a sanitized v1 or Huma validation detail from a bounded JSON object."""
    return _optional_detail(payload) or "the request did not satisfy the API schema"


def _invalid_argument(message: str) -> str:
    """Return the standard envelope for an argument this module rejected locally."""
    return tool_error_json(structured_error("agent_lens_invalid_request", message))


def _rank_top_clusters(breakdown_data: Any, top_n: int) -> List[Dict[str, Any]]:
    """Extract and sort clusters across categories by turn count descending."""
    if not isinstance(breakdown_data, dict):
        return []
    categories = breakdown_data.get("categories")
    if not isinstance(categories, list):
        return []
    ranked: List[Dict[str, Any]] = []
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        category_id = cat.get("category")
        clusters = cat.get("cluster_breakdowns")
        if not isinstance(clusters, list):
            continue
        for cluster in clusters:
            if isinstance(cluster, dict):
                ranked.append({"category": category_id, **cluster})
    ranked.sort(key=lambda item: int(item.get("count") or 0), reverse=True)
    return ranked[:top_n]


async def get_insights_coverage(entity_name: str, project_name: str) -> str:
    """Report the first and latest weeks that have classified Insights turns."""
    return await _agent_lens_request(
        "get_insights_coverage",
        "GET",
        LATEST_WEEK_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


async def get_clustering_status(entity_name: str, project_name: str) -> str:
    """Report the latest successful clustering run per signature type."""
    return await _agent_lens_request(
        "get_clustering_status",
        "GET",
        CLUSTERING_STATUS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


async def get_category_breakdowns(
    entity_name: str,
    project_name: str,
    start_at: str,
    end_at: str,
    signature_type: Literal["intent", "failure", "both"] = "both",
    filters: Optional[List[Dict[str, Any]]] = None,
    top_n_clusters: Optional[int] = None,
) -> str:
    """Return intent and/or failure category and cluster breakdowns for a bounded range."""
    if signature_type not in {"intent", "failure", "both"}:
        return _invalid_argument('signature_type must be "intent", "failure", or "both"')
    if top_n_clusters is not None and (isinstance(top_n_clusters, bool) or not 1 <= top_n_clusters <= 100):
        return _invalid_argument("top_n_clusters must be an integer between 1 and 100")
    try:
        params = _validated_window(start_at, end_at)
        serialized_filters = _serialize_insights_filters(filters)
        if serialized_filters is not None:
            params["filters"] = serialized_filters
    except ValueError as error:
        return _invalid_argument(str(error))

    track = {
        "entity_name": entity_name,
        "project_name": project_name,
        "signature_type": signature_type,
        "filters": bool(filters),
    }
    if signature_type in {"intent", "failure"}:
        res = await fetch_agent_lens_json(
            _breakdowns_path(signature_type),
            tool_name="get_category_breakdowns",
            method="GET",
            entity_name=entity_name,
            project_name=project_name,
            track_params=track,
            params=params,
        )
        if isinstance(res, ToolErrorJSON):
            return res
        payload: Dict[str, Any] = dict(res)
        if top_n_clusters is not None:
            payload["top_clusters"] = _rank_top_clusters(payload.get("data"), top_n_clusters)
        return _compact_json(_truncate_response(payload))

    intent_res, failure_res = await asyncio.gather(
        fetch_agent_lens_json(
            _breakdowns_path("intent"),
            tool_name="get_category_breakdowns",
            method="GET",
            entity_name=entity_name,
            project_name=project_name,
            track_params={**track, "signature_type": "intent"},
            params=params,
        ),
        fetch_agent_lens_json(
            _breakdowns_path("failure"),
            tool_name="get_category_breakdowns",
            method="GET",
            entity_name=entity_name,
            project_name=project_name,
            track_params={**track, "signature_type": "failure"},
            params=params,
        ),
    )
    if isinstance(intent_res, ToolErrorJSON):
        return intent_res
    if isinstance(failure_res, ToolErrorJSON):
        return failure_res

    combined: Dict[str, Any] = {
        "data": {
            "intent": intent_res.get("data"),
            "failure": failure_res.get("data"),
        }
    }
    if top_n_clusters is not None:
        combined["top_clusters"] = {
            "intent": _rank_top_clusters(intent_res.get("data"), top_n_clusters),
            "failure": _rank_top_clusters(failure_res.get("data"), top_n_clusters),
        }
    return _compact_json(_truncate_response(combined))


async def list_clusters(
    entity_name: str,
    project_name: str,
    signature_type: Literal["intent", "failure"],
    start_at: str,
    end_at: str,
    category_ids: Optional[List[str]] = None,
    exclude_categories: bool = False,
    top_n: Optional[int] = None,
    include_volume: bool = False,
    bucket_seconds: int = 3600,
    filters: Optional[List[Dict[str, Any]]] = None,
    include_cluster_counts: bool = False,
) -> str:
    """List ranked Agent Lens intent or failure clusters, with optional bucketed volume."""
    if signature_type not in {"intent", "failure"}:
        return _invalid_argument('signature_type must be "intent" or "failure"')
    if top_n is not None and (isinstance(top_n, bool) or not 1 <= top_n <= 200):
        return _invalid_argument("top_n must be an integer between 1 and 200")
    try:
        window = _validated_window(start_at, end_at)
        cluster_params: Dict[str, Any] = {**window, "type": signature_type}
        if category_ids:
            cluster_params["category_ids[]"] = _bounded_list(category_ids, MAX_CATEGORY_IDS, "category_ids")
        if exclude_categories:
            cluster_params["exclude_categories"] = True
        serialized_filters = _serialize_insights_filters(filters)
    except ValueError as error:
        return _invalid_argument(str(error))

    if not include_volume and serialized_filters is not None:
        return _invalid_argument("filters applies to volume series; set include_volume=True or use category_ids")

    volume_params: Optional[Dict[str, Any]] = None
    if include_volume:
        if not MIN_VOLUME_BUCKET_SECONDS <= bucket_seconds <= MAX_TIME_BUCKET_SECONDS:
            return _invalid_argument(
                f"bucket_seconds must be between {MIN_VOLUME_BUCKET_SECONDS} and {MAX_TIME_BUCKET_SECONDS}"
            )
        span_seconds = (parse_timestamp(end_at, "end_at") - parse_timestamp(start_at, "start_at")).total_seconds()
        if math.ceil(span_seconds / bucket_seconds) > MAX_VOLUME_BUCKETS:
            return _invalid_argument(
                f"time window and bucket_seconds exceed {MAX_VOLUME_BUCKETS} buckets; increase bucket_seconds"
            )
        volume_params = {**window, "bucket_seconds": bucket_seconds}
        if serialized_filters is not None:
            volume_params["filters"] = serialized_filters
        if include_cluster_counts:
            volume_params["include_cluster_counts"] = True

    track = {
        "entity_name": entity_name,
        "project_name": project_name,
        "signature_type": signature_type,
        "include_volume": include_volume,
    }
    if volume_params is None:
        clusters_res = await fetch_agent_lens_json(
            CLUSTERS_PATH,
            tool_name="list_clusters",
            method="GET",
            entity_name=entity_name,
            project_name=project_name,
            track_params=track,
            params=cluster_params,
        )
        volume_res = None
    else:
        clusters_res, volume_res = await asyncio.gather(
            fetch_agent_lens_json(
                CLUSTERS_PATH,
                tool_name="list_clusters",
                method="GET",
                entity_name=entity_name,
                project_name=project_name,
                track_params=track,
                params=cluster_params,
            ),
            fetch_agent_lens_json(
                _volume_path(signature_type),
                tool_name="list_clusters",
                method="GET",
                entity_name=entity_name,
                project_name=project_name,
                track_params=track,
                params=volume_params,
            ),
        )
    if isinstance(clusters_res, ToolErrorJSON):
        return clusters_res
    if isinstance(volume_res, ToolErrorJSON):
        return volume_res

    rows = clusters_res.get("data")
    if isinstance(rows, list):
        sorted_rows = sorted(
            rows,
            key=lambda item: float(item.get("share") or 0.0) if isinstance(item, dict) else 0.0,
            reverse=True,
        )
        if top_n is not None:
            sorted_rows = sorted_rows[:top_n]
    else:
        sorted_rows = []
    payload: Dict[str, Any] = {"data": sorted_rows}
    if isinstance(volume_res, dict):
        payload["volume"] = volume_res.get("data")
    return _compact_json(_truncate_response(payload))


async def list_category_example_turns(
    entity_name: str,
    project_name: str,
    signature_type: Literal["intent", "failure"],
    category_id: str,
    start_at: str,
    end_at: str,
    topic_ids: Optional[List[str]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    limit: int = 10,
    cursor: Optional[str] = None,
    per_topic_limit: Optional[int] = None,
) -> str:
    """Page or stratified-sample example turns for one intent or failure category."""
    if signature_type not in {"intent", "failure"}:
        return _invalid_argument('signature_type must be "intent" or "failure"')
    if not str(category_id).strip():
        return _invalid_argument("category_id must be a non-empty category identifier")
    if not 1 <= limit <= MAX_EXAMPLE_LIMIT:
        return _invalid_argument(f"limit must be between 1 and {MAX_EXAMPLE_LIMIT}")
    if per_topic_limit is not None:
        if isinstance(per_topic_limit, bool) or not 1 <= per_topic_limit <= MAX_PER_TOPIC_LIMIT:
            return _invalid_argument(f"per_topic_limit must be between 1 and {MAX_PER_TOPIC_LIMIT}")
        if not topic_ids:
            return _invalid_argument("per_topic_limit requires at least one topic_id in topic_ids")
        if cursor:
            return _invalid_argument("cursor cannot be combined with per_topic_limit stratified sampling")

    try:
        base_params = _validated_window(start_at, end_at)
        validated_topics = _bounded_list(topic_ids, MAX_TOPIC_IDS, "topic_ids") if topic_ids else None
        serialized_filters = _serialize_insights_filters(filters)
        if serialized_filters is not None:
            base_params["filters"] = serialized_filters
    except ValueError as error:
        return _invalid_argument(str(error))

    path = _category_examples_path(signature_type, category_id)
    track = {
        "entity_name": entity_name,
        "project_name": project_name,
        "signature_type": signature_type,
        "filters": bool(filters),
    }

    if per_topic_limit is not None and validated_topics:
        semaphore = asyncio.Semaphore(_STRATIFIED_CONCURRENCY)

        async def _fetch_one_topic(topic_id: str) -> tuple[str, Dict[str, object] | ToolErrorJSON]:
            async with semaphore:
                topic_params = {**base_params, "topic_ids[]": [topic_id], "limit": per_topic_limit}
                res = await fetch_agent_lens_json(
                    path,
                    tool_name="list_category_example_turns",
                    method="GET",
                    entity_name=entity_name,
                    project_name=project_name,
                    track_params=track,
                    params=topic_params,
                )
                return topic_id, res

        pairs = await asyncio.gather(*(_fetch_one_topic(tid) for tid in validated_topics))
        combined_rows: List[Dict[str, Any]] = []
        by_topic: Dict[str, List[Any]] = {}
        for tid, res in pairs:
            if isinstance(res, ToolErrorJSON):
                return res
            projected = _project_response("list_category_example_turns", dict(res))
            topic_rows = projected.get("data") if isinstance(projected.get("data"), list) else []
            tagged_rows = [{**row, "topic_id": tid} if isinstance(row, dict) else row for row in topic_rows]
            by_topic[tid] = tagged_rows
            combined_rows.extend(tagged_rows)
        return _compact_json(_truncate_response({"data": combined_rows, "by_topic": by_topic, "next_cursor": None}))

    params = dict(base_params)
    if validated_topics:
        params["topic_ids[]"] = validated_topics
    params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    return await _agent_lens_request(
        "list_category_example_turns",
        "GET",
        path,
        entity_name,
        project_name,
        track,
        params=params,
    )


async def get_failure_attributions(
    entity_name: str,
    project_name: str,
    trace_ids: List[str],
    selected_failure: Optional[Dict[str, str]] = None,
) -> str:
    """Return Agent Lens failure attribution and cluster detail for the requested trace IDs."""
    try:
        ids = _bounded_list(trace_ids, MAX_TRACE_IDS, "trace_ids")
    except ValueError as error:
        return _invalid_argument(str(error))

    body: Dict[str, Any] = {"trace_ids": ids}
    if selected_failure is not None:
        if not isinstance(selected_failure, dict):
            return _invalid_argument("selected_failure must be an object with 'trace_id' and 'failure_id'")
        sel_trace = str(selected_failure.get("trace_id") or "").strip()
        sel_failure = str(selected_failure.get("failure_id") or "").strip()
        if not sel_trace or not sel_failure:
            return _invalid_argument("selected_failure requires non-empty 'trace_id' and 'failure_id'")
        body["selected_failure"] = {"trace_id": sel_trace, "failure_id": sel_failure}

    return await _agent_lens_request(
        "get_failure_attributions",
        "POST",
        FAILURE_ATTRIBUTIONS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name, "trace_count": len(ids)},
        body=body,
    )


async def list_tags(
    entity_name: str,
    project_name: str,
    tag_id: Optional[str] = None,
) -> str:
    """List the Agent Lens tag catalog or retrieve one tag via the official v1 API."""
    if tag_id is not None:
        try:
            validated_id = _validated_uuid(tag_id, "tag_id")
        except ValueError as error:
            return _invalid_argument(str(error))
        path = _tag_path(validated_id)
    else:
        path = TAGS_PATH
    return await _agent_lens_request(
        "list_tags",
        "GET",
        path,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        api_version="v1",
    )


async def get_conversation_tags(entity_name: str, project_name: str, conversation_ids: List[str]) -> str:
    """Return every tag applied to the given conversations, with provenance."""
    try:
        ids = _bounded_list(conversation_ids, MAX_CONVERSATION_IDS, "conversation_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    return await _agent_lens_request(
        "get_conversation_tags",
        "POST",
        CONVERSATION_TAGS_QUERY_PATH,
        entity_name,
        project_name,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "conversation_count": len(ids),
        },
        body={"conversation_ids": ids},
    )


async def list_tagged_conversations(
    entity_name: str,
    project_name: str,
    tag_ids: List[str],
    include_assignments: bool = False,
    applied_after: Optional[str] = None,
    applied_before: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List conversation IDs for tag_ids, or paginated turn-level assignments for one tag."""
    try:
        ids = _bounded_uuid_list(tag_ids, MAX_TAG_IDS, "tag_ids")
    except ValueError as error:
        return _invalid_argument(str(error))

    use_assignments = include_assignments or applied_after is not None or applied_before is not None
    if use_assignments:
        if len(ids) != 1:
            return _invalid_argument("include_assignments and applied_after/applied_before require exactly one tag_id")
        if isinstance(limit, bool) or not 1 <= limit <= MAX_TAG_ASSIGNMENTS_LIMIT:
            return _invalid_argument(f"limit must be between 1 and {MAX_TAG_ASSIGNMENTS_LIMIT}")
        if isinstance(offset, bool) or offset < 0:
            return _invalid_argument("offset must be >= 0")
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        try:
            if applied_after is not None:
                parse_timestamp(applied_after, "applied_after")
                params["applied_after"] = applied_after
            if applied_before is not None:
                parse_timestamp(applied_before, "applied_before")
                params["applied_before"] = applied_before
            if applied_after is not None and applied_before is not None:
                if parse_timestamp(applied_before, "applied_before") <= parse_timestamp(applied_after, "applied_after"):
                    return _invalid_argument("applied_before must be later than applied_after")
        except ValueError as error:
            return _invalid_argument(str(error))
        return await _agent_lens_request(
            "list_tagged_conversations",
            "GET",
            _tag_assignments_path(ids[0]),
            entity_name,
            project_name,
            {"entity_name": entity_name, "project_name": project_name, "include_assignments": True},
            params=params,
        )

    return await _agent_lens_request(
        "list_tagged_conversations",
        "POST",
        TAGGED_CONVERSATIONS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name, "tag_count": len(ids)},
        body={"tag_ids": ids},
    )


async def get_tag_distribution(
    entity_name: str,
    project_name: str,
    after_ms: int,
    before_ms: int,
    time_bucket_seconds: int,
) -> str:
    """Return per-tag counts bucketed over time."""
    if after_ms < 0 or before_ms < 1:
        return _invalid_argument("after_ms must be >= 0 and before_ms must be >= 1")
    if before_ms <= after_ms:
        return _invalid_argument("before_ms must be greater than after_ms")
    if not 1 <= time_bucket_seconds <= MAX_TIME_BUCKET_SECONDS:
        return _invalid_argument(f"time_bucket_seconds must be between 1 and {MAX_TIME_BUCKET_SECONDS}")
    return await _agent_lens_request(
        "get_tag_distribution",
        "POST",
        TAG_DISTRIBUTION_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        body={
            "after_ms": after_ms,
            "before_ms": before_ms,
            "time_bucket_seconds": time_bucket_seconds,
        },
    )


async def get_guidelines(
    entity_name: str,
    project_name: str,
    summarize_conversation_id: Optional[str] = None,
) -> str:
    """List project judge guidelines, or generate an Agent Lens conversation summary."""
    if summarize_conversation_id is not None:
        conv_id = str(summarize_conversation_id).strip()
        if not conv_id or len(conv_id) > 512:
            return _invalid_argument("summarize_conversation_id must be a non-empty identifier up to 512 chars")
        res = await fetch_agent_lens_json(
            _conversation_summarize_path(conv_id),
            tool_name="get_guidelines",
            method="POST",
            entity_name=entity_name,
            project_name=project_name,
            track_params={"entity_name": entity_name, "project_name": project_name},
        )
        if isinstance(res, ToolErrorJSON):
            return res
        wrapped = res if "data" in res else {"data": res}
        return _compact_json(_truncate_response(wrapped))

    return await _agent_lens_request(
        "get_guidelines",
        "GET",
        GUIDELINES_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        params={"type": "project"},
    )


def _validate_tag_slug(name: str) -> str:
    slug = str(name).strip()
    if not 1 <= len(slug) <= 64 or not _TAG_SLUG_RE.fullmatch(slug):
        raise ValueError("name must be 1-64 lowercase letters, digits, and single hyphens (e.g. 'wrong-track')")
    return slug


async def manage_tags(
    entity_name: str,
    project_name: str,
    action: Literal["create_tag", "update_tag", "delete_tag", "assign_tag", "remove_tag"],
    tag_id: Optional[str] = None,
    name: Optional[str] = None,
    criteria: Optional[str] = None,
    clear_criteria: bool = False,
    color: Optional[str] = None,
    agent_name_filter: Optional[List[str]] = None,
    clear_agent_name_filter: bool = False,
    llm_judge_enabled: Optional[bool] = None,
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> str:
    """Create, update, or delete Agent Lens v1 tags, or apply/remove human turn tags."""
    if action not in {"create_tag", "update_tag", "delete_tag", "assign_tag", "remove_tag"}:
        return _invalid_argument(
            'action must be "create_tag", "update_tag", "delete_tag", "assign_tag", or "remove_tag"'
        )

    track = {"entity_name": entity_name, "project_name": project_name}
    try:
        if action in {"create_tag", "update_tag"}:
            if criteria is not None and len(criteria) > MAX_TAG_CRITERIA_CHARS:
                return _invalid_argument(f"criteria must be at most {MAX_TAG_CRITERIA_CHARS} characters")
            if color is not None and not 1 <= len(color.strip()) <= 7:
                return _invalid_argument("color must be 1 to 7 characters")
            if agent_name_filter is not None:
                agent_name_filter = _bounded_list(agent_name_filter, 100, "agent_name_filter")

        if action == "create_tag":
            if name is None:
                return _invalid_argument("name is required for action='create_tag'")
            slug = _validate_tag_slug(name)
            if llm_judge_enabled and (criteria is None or not criteria.strip()):
                return _invalid_argument("criteria is required when llm_judge_enabled is True")
            body: Dict[str, Any] = {
                "name": slug,
                "criteria": criteria,
                "color": color.strip() if color else None,
                "agent_name_filter": agent_name_filter,
                "llm_judge_enabled": llm_judge_enabled,
            }
            return await _agent_lens_request(
                "manage_tags",
                "POST",
                TAGS_PATH,
                entity_name,
                project_name,
                track,
                body=body,
                api_version="v1",
            )

        if action == "update_tag":
            if tag_id is None:
                return _invalid_argument("tag_id is required for action='update_tag'")
            validated_id = _validated_uuid(tag_id, "tag_id")
            preserve_nulls: List[str] = []
            update_body: Dict[str, Any] = {}
            if name is not None:
                update_body["name"] = _validate_tag_slug(name)
            if clear_criteria:
                update_body["criteria"] = None
                preserve_nulls.append("criteria")
            elif criteria is not None:
                update_body["criteria"] = criteria
            if color is not None:
                update_body["color"] = color.strip()
            if clear_agent_name_filter:
                update_body["agent_name_filter"] = None
                preserve_nulls.append("agent_name_filter")
            elif agent_name_filter is not None:
                update_body["agent_name_filter"] = agent_name_filter
            if llm_judge_enabled is not None:
                update_body["llm_judge_enabled"] = llm_judge_enabled
            if not update_body:
                return _invalid_argument("update_tag requires at least one field to update")
            return await _agent_lens_request(
                "manage_tags",
                "POST",
                _tag_path(validated_id),
                entity_name,
                project_name,
                track,
                body=update_body,
                api_version="v1",
                preserve_null_keys=preserve_nulls,
            )

        if action == "delete_tag":
            if tag_id is None:
                return _invalid_argument("tag_id is required for action='delete_tag'")
            validated_id = _validated_uuid(tag_id, "tag_id")
            return await _agent_lens_request(
                "manage_tags",
                "DELETE",
                _tag_path(validated_id),
                entity_name,
                project_name,
                track,
                api_version="v1",
            )

        conv = str(conversation_id or "").strip()
        trace = str(trace_id or "").strip()
        if not conv or len(conv) > 255 or not trace or len(trace) > 255:
            return _invalid_argument("conversation_id and trace_id (1-255 chars) are required for turn tag actions")

        if action == "assign_tag":
            if bool(tag_id) == bool(name):
                return _invalid_argument("provide either tag_id or name (not both) for action='assign_tag'")
            assign_body: Dict[str, Any] = {"conversation_id": conv, "trace_id": trace}
            if tag_id is not None:
                assign_body["tag_id"] = _validated_uuid(tag_id, "tag_id")
            else:
                assign_body["tag_name"] = _validate_tag_slug(name or "")
            return await _agent_lens_request(
                "manage_tags",
                "POST",
                CONVERSATION_TAGS_PATH,
                entity_name,
                project_name,
                track,
                body=assign_body,
            )

        if tag_id is None:
            return _invalid_argument("tag_id is required for action='remove_tag'")
        remove_body = {
            "conversation_id": conv,
            "trace_id": trace,
            "tag_id": _validated_uuid(tag_id, "tag_id"),
        }
        return await _agent_lens_request(
            "manage_tags",
            "DELETE",
            CONVERSATION_TAGS_PATH,
            entity_name,
            project_name,
            track,
            body=remove_body,
        )
    except ValueError as error:
        return _invalid_argument(str(error))


async def manage_guidelines(
    entity_name: str,
    project_name: str,
    action: Literal["create", "update", "delete"],
    guideline_id: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    """Create, update, or delete project-level Agent Lens judge guidelines."""
    if action not in {"create", "update", "delete"}:
        return _invalid_argument('action must be "create", "update", or "delete"')
    track = {"entity_name": entity_name, "project_name": project_name}
    try:
        if action in {"create", "update"}:
            trimmed = (content or "").strip()
            if not trimmed or len(trimmed) > MAX_GUIDELINE_CHARS:
                return _invalid_argument(f"content must be 1 to {MAX_GUIDELINE_CHARS} non-whitespace characters")
            if action == "create":
                return await _agent_lens_request(
                    "manage_guidelines",
                    "POST",
                    GUIDELINES_PATH,
                    entity_name,
                    project_name,
                    track,
                    body={"type": "project", "content": trimmed},
                )
            if guideline_id is None:
                return _invalid_argument("guideline_id is required for action='update'")
            return await _agent_lens_request(
                "manage_guidelines",
                "PATCH",
                _guideline_path(_validated_uuid(guideline_id, "guideline_id")),
                entity_name,
                project_name,
                track,
                body={"content": trimmed},
            )

        if guideline_id is None:
            return _invalid_argument("guideline_id is required for action='delete'")
        return await _agent_lens_request(
            "manage_guidelines",
            "DELETE",
            _guideline_path(_validated_uuid(guideline_id, "guideline_id")),
            entity_name,
            project_name,
            track,
        )
    except ValueError as error:
        return _invalid_argument(str(error))


GET_INSIGHTS_COVERAGE_TOOL_DESCRIPTION = """Report which weeks have Agent Lens Insights data for a project.

<when_to_use>
Call FIRST before Insights queries so you use populated `first_week`/`latest_week`
bounds instead of guessing a `start_at`/`end_at` window.
</when_to_use>

Returns `{"data": {"first_week", "latest_week"}}` (values may be null if unclassified).
"""


GET_CLUSTERING_STATUS_TOOL_DESCRIPTION = """Report the latest successful Agent Lens clustering run per signature type.

<when_to_use>
Use to check whether `intent` and `failure` clusters are fresh and see each
run's covered window (`window_start`, `window_end`, `completed_at`).
</when_to_use>

Returns `{"data": [{"signature_type", "completed_at", "window_start", "window_end"}]}`.
"""


GET_CATEGORY_BREAKDOWNS_TOOL_DESCRIPTION = """Summarize Agent Lens intent and failure categories and clusters (`start_at`..`end_at` <= 30d).

<when_to_use>
Primary aggregate view for top user intents, failure modes, and severities via
`/api/insights/{type}/breakdowns`. Defaults to `signature_type="both"`; pass
`top_n_clusters` (1-100) for a flat ranked cluster summary, or `filters`
(`[{"field": "agent_name"|"language"|"intent_category"|"intent_sentiment"|"failure_category"|"failure_severity"|"intent_topic_id"|"failure_topic_id", "values": [...], "exclude"?: bool}]`).
Drill down with `list_agent_lens_category_example_turns_tool`.
</when_to_use>

Returns `data` (`turn_count`, `categories[]` with `cluster_breakdowns[]`) and optional `top_clusters`.
"""


LIST_CLUSTERS_TOOL_DESCRIPTION = """List ranked Agent Lens intent or failure clusters and optional time-series volume (`start_at`..`end_at` <= 30d).

<when_to_use>
Use for a flat `share`-ranked cluster list (`id`, `title`, `label`, `category`,
`share`) with optional `top_n` (1-200) and `category_ids` filter, or set
`include_volume=True` (`bucket_seconds` 60-86400) to include bucketed category
and per-cluster (`include_cluster_counts=True`) volume trends.
</when_to_use>

Returns `{"data": [Cluster], "volume"?: {"time_bucket_seconds", "buckets"}}`.
"""


LIST_CATEGORY_EXAMPLE_TURNS_TOOL_DESCRIPTION = f"""Page or stratified-sample example turns for an Agent Lens category (`start_at`..`end_at` <= 30d).

<when_to_use>
Ground categories/clusters in real `conversation_id` and `trace_id` handles. For
balanced sampling across clusters, pass `topic_ids` (1-{MAX_TOPIC_IDS}) AND
`per_topic_limit` (1-{MAX_PER_TOPIC_LIMIT}, omit `cursor`) to fetch up to N
turns per cluster concurrently (`by_topic`). Read full turns via
`get_weave_agent_conversation_tool` or `get_weave_agent_trace_tool`.
</when_to_use>

Returns `{{"data": [...], "next_cursor", "by_topic"?: {{...}}}}` with failure reasons, severities, and evidence span IDs.
"""


GET_FAILURE_ATTRIBUTIONS_TOOL_DESCRIPTION = f"""Fetch Agent Lens failure reasons, cluster (`id`, `title`), severity, and evidence spans for 1-{MAX_TRACE_IDS} `trace_ids`.

<when_to_use>
Use after sampling turns or querying spans when you need root-cause attribution
details (`failure_reason`, `failure_cluster`, `failure_category`,
`failure_severity`, `failure_evidence_span_ids`) or want to pin
`selected_failure={{"trace_id", "failure_id"}}`.
</when_to_use>

Returns `{{"data": [TurnFailureAttribution]}}`.
"""


LIST_TAGS_TOOL_DESCRIPTION = """List the Agent Lens v1 tag catalog or retrieve one tag by UUID (`/api/v1/tags`).

<when_to_use>
Call before filtering or assigning tags to discover tag UUIDs, lowercase
hyphenated names, judge `criteria`, `agent_name_filter`, and `llm_judge_enabled`.
Pass `tag_id` to fetch a single tag.
</when_to_use>

Returns `{"data": [Tag]}` or `{"data": Tag}`.
"""


GET_CONVERSATION_TAGS_TOOL_DESCRIPTION = f"""Fetch Agent Lens tags and provenance (`human` vs `judge`, `rationale`) for 1-{MAX_CONVERSATION_IDS} `conversation_ids`.

<when_to_use>
Use when you already have `conversation_ids` and want every applied tag,
`trace_id`, `source`, `wb_user_id`, and judge `rationale`.
</when_to_use>

Returns `{{"data": [ConversationTag]}}`.
"""


LIST_TAGGED_CONVERSATIONS_TOOL_DESCRIPTION = f"""List conversations or paginated turn-level assignments for 1-{MAX_TAG_IDS} Agent Lens `tag_ids`.

<when_to_use>
Find conversation IDs matching any listed `tag_ids` (OR match), or set
`include_assignments=True` (or `applied_after`/`applied_before` RFC 3339 bounds)
with a single `tag_id` to page turn-level assignments (`conversation_id`,
`trace_id`, `turn_number`, `source`, `rationale`, `total_count`).
</when_to_use>

Returns `{{"data": [conversation_id]}}` or `{{"data": [TagAssignment], "total_count": int}}`.
"""


GET_TAG_DISTRIBUTION_TOOL_DESCRIPTION = f"""Return Agent Lens conversation tag counts bucketed over time (`after_ms`..`before_ms`, `time_bucket_seconds` 1-{MAX_TIME_BUCKET_SECONDS}).

<when_to_use>
Use for tag volume trend questions ("is this tag spiking over the last week?")
using epoch-millisecond bounds.
</when_to_use>

Returns `{{"data": {{"time_bucket_seconds", "after_ms", "before_ms", "buckets"}}}}`.
"""


GET_GUIDELINES_TOOL_DESCRIPTION = """Read project Agent Lens judge guidelines or generate a single-conversation summary.

<when_to_use>
Inspect project-level guidelines injected into Agent Lens intent/failure judges,
or pass `summarize_conversation_id` to generate a `{title, summary}` for one
conversation.
</when_to_use>

Returns `{"data": [Guideline], "max_length", "read_only"}` or `{"data": {"title", "summary"}}`.
"""


MANAGE_TAGS_TOOL_DESCRIPTION = """Create, update, or delete Agent Lens v1 tags (`/api/v1/tags`), or assign/remove human turn tags.

<when_to_use>
Use in read-write mode for tag CRUD (`create_tag`, `update_tag`, `delete_tag`;
slug `name` 1-64 lowercase `[a-z0-9-]`, `criteria` <=1600 chars required when
`llm_judge_enabled=True`, `clear_criteria`/`clear_agent_name_filter` to null) or
human turn tagging (`assign_tag`, `remove_tag` with `conversation_id` + `trace_id`).
</when_to_use>

Returns `{"data": Tag}` or `{"data": {"ok": true}}`.
"""


MANAGE_GUIDELINES_TOOL_DESCRIPTION = """Create, update, or delete project-level Agent Lens judge guidelines (1-4000 chars).

<when_to_use>
Use in read-write mode (`action` = `create` | `update` | `delete`) to tune what
Insights intent and failure judges treat as severe vs benign across future
project evaluations.
</when_to_use>

Returns `{"data": Guideline}` or `{"data": {"ok": true}}`.
"""
