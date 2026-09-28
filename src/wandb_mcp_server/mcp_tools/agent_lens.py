"""MCP tools for the **Agent Lens** Insights and conversation-tag read APIs.

Agent Lens is a separate W&B service with its own Huma-generated HTTP API. It
classifies agent turns into intent/failure categories and clusters, and carries
human- and judge-applied conversation tags. None of that is visible to the Weave
calls or Agents data planes, so these tools complement -- they do not duplicate
-- ``query_weave_traces_tool`` and the ``*_weave_agent_*`` family.

Every tool here is a read. Agent Lens authenticates a W&B API key as a bearer
token and derives project authorization from it server-side, so an MCP caller
can never read a project the key itself cannot reach. Requests carry the project
in ``X-Wandb-Entity`` / ``X-Wandb-Project`` headers, matching ``ProjectHeaders``
in ``internal/api/project.go`` of github.com/wandb/agent-lens.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote
from uuid import UUID

import requests

from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.api_client import WandBApiManager, WandBServerBusy
from wandb_mcp_server.config import (
    AGENT_LENS_API_PREFIX,
    MAX_ACCUMULATED_BYTES,
    MAX_RESPONSE_TOKENS,
    resolve_agent_lens_base_url,
    structured_error,
)
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
from wandb_mcp_server.mcp_tools.tools_utils import get_no_retry_session, track_tool_execution
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

# Read paths on the Agent Lens API, relative to AGENT_LENS_API_PREFIX. Confirmed
# against internal/api/insights.go and internal/api/conversation_tags.go.
LATEST_WEEK_PATH = "/insights/latest-week"
CLUSTERING_STATUS_PATH = "/insights/clustering-status"
CATEGORY_BREAKDOWNS_PATH = "/insights/intent-category-breakdowns"
FAILURE_ATTRIBUTIONS_PATH = "/insights/failure-attributions/query"
TAGS_PATH = "/tags"
CONVERSATION_TAGS_QUERY_PATH = "/conversation-tags/query"
TAGGED_CONVERSATIONS_PATH = "/conversation-tags/conversations/query"
TAG_DISTRIBUTION_PATH = "/conversation-tags/distribution"


def _category_examples_path(signature_type: str, category_id: str) -> str:
    """Build the drilldown path, percent-encoding the caller-supplied category."""
    return f"/insights/{quote(signature_type, safe='')}/categories/{quote(category_id, safe='')}/example-turns"


_REQUEST_TIMEOUT_SECONDS = 30
MAX_AGENT_LENS_RESPONSE_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_DEFAULT_RETRY_AFTER_MS = 1_000
_MAX_RETRY_AFTER_MS = 60_000

# MCP keeps ranged Insights reads within 30 days to bound work and responses,
# even though current Agent Lens accepts any nonempty range.
MAX_INSIGHTS_WINDOW_DAYS = 30

# Server-enforced request bounds, mirrored here so an oversized argument fails
# with a usable message instead of a generic 422.
MAX_CONVERSATION_IDS = 5000
MAX_TAG_IDS = 100
MAX_TOPIC_IDS = 20
MAX_TRACE_IDS = 500
MAX_EXAMPLE_LIMIT = 50
MAX_TIME_BUCKET_SECONDS = 86400


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so Agent Lens applies its own field defaults."""
    return {k: v for k, v in values.items() if v is not None}


class _ResponseBoundaryError(ValueError):
    """A bounded, externally safe Agent Lens response failure."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))


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
    """Trim the response's primary list so the serialized payload fits the budget.

    Agent Lens wraps every read in ``{"data": ...}``. That is a list for most
    reads and an object holding ``buckets`` for the tag distribution, so we trim
    whichever one is actually present and annotate ``_truncation`` to tell the
    caller to narrow the range rather than silently returning a partial answer.
    """
    if not isinstance(payload, dict):
        return payload
    if _response_tokens(payload) <= MAX_RESPONSE_TOKENS:
        return payload

    data = payload.get("data")
    if isinstance(data, list) and data and payload.get("next_cursor") is not None:
        # Returning a cursor after dropping rows would make the caller skip the
        # omitted portion of this page. Fail the page instead so it can be
        # retried with a smaller upstream limit.
        cursor_error = {
            "error": "agent_lens_response_too_large",
            "message": (
                "The Agent Lens page exceeded the response budget. Retry with a lower limit "
                "so no paginated rows are skipped."
            ),
        }
        if _response_tokens(cursor_error) <= MAX_RESPONSE_TOKENS:
            return cursor_error
        return {"error": "agent_lens_response_too_large"}
    if isinstance(data, list) and data:
        container, key = payload, "data"
    elif isinstance(data, dict) and isinstance(data.get("buckets"), list) and data["buckets"]:
        container, key = data, "buckets"
    else:
        container = None
        key = "data"

    if container is not None:
        items = container[key]
        original = len(items)
        kept = list(items)

        def rebuilt(candidate: List[Any]) -> Dict[str, Any]:
            if container is payload:
                result = {**payload, key: candidate}
            else:
                result = {**payload, "data": {**data, key: candidate}}
            result["_truncation"] = _truncation_metadata(key, len(candidate), original)
            return result

        # Include the notice during every measurement so the final serialized
        # response, not just the retained upstream data, obeys the budget.
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
        return fallback
    minimal = {"error": "agent_lens_response_too_large"}
    return minimal if _response_tokens(minimal) <= MAX_RESPONSE_TOKENS else {}


def _project_response(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove response fields that are outside the reviewed public contract."""
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
    """Return Requests' inactivity timeout from the captured absolute deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Agent Lens tool deadline elapsed")
    return max(0.001, remaining)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Agent Lens tool deadline elapsed")


def _arm_response_deadline(response: Any, deadline: float) -> tuple[threading.Timer | None, threading.Event]:
    """Close a streaming response when the absolute MCP deadline expires."""
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                logger.warning("Agent Lens response close failed at deadline (%s)", type(error).__name__)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        expire()
        return None, expired
    timer = threading.Timer(remaining, expire)
    timer.daemon = True
    timer.start()
    return timer, expired


def _bounded_response_bytes(response: Any, deadline: float) -> bytes:
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
    for chunk in response.iter_content(chunk_size=_READ_CHUNK_BYTES, decode_unicode=False):
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


def _bounded_json_response(response: Any, deadline: float) -> Dict[str, Any]:
    raw = _bounded_response_bytes(response, deadline)
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


def _parse_timestamp(value: str, field: str) -> datetime:
    """Parse an RFC 3339 bound the way Agent Lens does, normalized to UTC."""
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be an RFC 3339 timestamp")
    # datetime.fromisoformat accepts "Z" only from 3.11; the server uses RFC3339Nano.
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError(f"{field} must be an RFC 3339 timestamp, for example 2026-09-01T00:00:00Z") from None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _validated_window(start_at: str, end_at: str) -> Dict[str, str]:
    """Check the range locally against the server's own bounds before sending."""
    start = _parse_timestamp(start_at, "start_at")
    end = _parse_timestamp(end_at, "end_at")
    if end <= start:
        raise ValueError("start_at and end_at must define a nonempty time range")
    if end - start > timedelta(days=MAX_INSIGHTS_WINDOW_DAYS):
        raise ValueError(f"time range must not exceed {MAX_INSIGHTS_WINDOW_DAYS} days")
    return {"start_at": start_at, "end_at": end_at}


def _bounded_list(values: Sequence[str], maximum: int, field: str) -> List[str]:
    """Reject an oversized list locally rather than trading a 422 round-trip."""
    items = [str(value) for value in values]
    if not items:
        raise ValueError(f"{field} must contain at least one value")
    if len(items) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} values (got {len(items)})")
    return items


def _bounded_uuid_list(values: Sequence[str], maximum: int, field: str) -> List[str]:
    """Validate opaque UUID identifiers without echoing a rejected value."""
    items = _bounded_list(values, maximum, field)
    try:
        return [str(UUID(item)) for item in items]
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{field} must contain only UUIDs") from None


def _agent_lens_request(
    tool_name: str,
    method: str,
    path: str,
    entity_name: str,
    project_name: str,
    track_params: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call one Agent Lens read endpoint and return a JSON string.

    Follows the Agents tools: a no-retry request wrapped in
    ``track_tool_execution`` and the standard ``structured_error`` envelope, with
    the response trimmed to the token budget. Retrying a read belongs to the MCP
    caller; doing it here would amplify an already loaded service.

    Unlike the trace server, Agent Lens parses the credential as a bearer token
    (``internal/auth/middleware.go``). An explicit Authorization header there
    never falls back to a cookie identity and never grants admin privileges, so
    the key's own project authorization is the only access this can obtain.
    """
    try:
        base_url = resolve_agent_lens_base_url()
    except ValueError as error:
        return json.dumps(structured_error("agent_lens_not_configured", str(error)))

    api_key = WandBApiManager.get_api_key()
    if not api_key:
        logger.error("W&B API key not found in context or environment.")
        return json.dumps(structured_error("auth_required", "A W&B API key is required to query the Agent Lens API."))

    url = f"{base_url}{AGENT_LENS_API_PREFIX}{path}"
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
        data = json.dumps(_drop_none(body))

    with track_tool_execution(tool_name, None, track_params) as ctx:
        # Only the HTTP round-trip can raise here, so the try wraps just that;
        # the status-code branching below is plain control flow and stays outside.
        try:
            request_deadline = _absolute_request_deadline()
            request_timeout = _remaining_request_timeout(request_deadline)
            response = get_no_retry_session().request(
                method,
                url,
                headers=headers,
                params=params or None,
                data=data,
                timeout=request_timeout,
                allow_redirects=False,
                stream=True,
            )
        except (TimeoutError, requests.Timeout):
            ctx.mark_error("tool_timeout")
            return _compact_json(
                structured_error(
                    "tool_timeout", "The Agent Lens request exceeded the MCP tool deadline.", retryable=True
                )
            )
        except Exception as e:
            logger.error("Agent Lens request failed (%s)", type(e).__name__)
            ctx.mark_error(type(e).__name__)
            return _compact_json(structured_error("agent_lens_query_failed", "The Agent Lens request failed."))

        deadline_timer, deadline_expired = _arm_response_deadline(response, request_deadline)
        try:
            _check_deadline(request_deadline)
            if response.status_code in {401, 403}:
                ctx.mark_error(f"http_{response.status_code}")
                return _compact_json(
                    structured_error(
                        "agent_lens_forbidden",
                        "Agent Lens rejected the W&B credential for this project. Confirm the credential has access "
                        "and that Agent Lens is enabled for the project.",
                        status_code=response.status_code,
                    )
                )
            if response.status_code == 404:
                ctx.mark_error("agent_lens_unavailable")
                return _compact_json(
                    structured_error(
                        "agent_lens_unavailable",
                        "The configured Agent Lens origin does not expose this endpoint (404); it may "
                        "predate this API or have the feature disabled.",
                        status_code=404,
                    )
                )
            if response.status_code == 422:
                ctx.mark_error("http_422")
                try:
                    validation_payload = _bounded_json_response(response, request_deadline)
                    detail = _detail(validation_payload)
                except _ResponseBoundaryError as error:
                    if deadline_expired.is_set():
                        raise TimeoutError("Agent Lens tool deadline elapsed") from error
                    detail = "the request did not satisfy the API schema"
                return _compact_json(
                    structured_error(
                        "agent_lens_invalid_request",
                        f"Agent Lens rejected the request arguments: {detail}",
                        status_code=422,
                    )
                )
            if response.status_code != 200:
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

            try:
                result = _bounded_json_response(response, request_deadline)
            except _ResponseBoundaryError as error:
                if deadline_expired.is_set():
                    raise TimeoutError("Agent Lens tool deadline elapsed") from error
                ctx.mark_error(error.reason)
                return _compact_json(
                    structured_error("agent_lens_query_failed", sanitize_sensitive_text(error, max_chars=512))
                )
            if deadline_expired.is_set():
                raise TimeoutError("Agent Lens tool deadline elapsed")
        except (TimeoutError, requests.Timeout):
            ctx.mark_error("tool_timeout")
            return _compact_json(
                structured_error(
                    "tool_timeout",
                    "The Agent Lens request exceeded the MCP tool deadline.",
                    retryable=True,
                )
            )
        except requests.RequestException as error:
            if deadline_expired.is_set():
                ctx.mark_error("tool_timeout")
                return _compact_json(
                    structured_error(
                        "tool_timeout",
                        "The Agent Lens request exceeded the MCP tool deadline.",
                        retryable=True,
                    )
                )
            ctx.mark_error(type(error).__name__)
            return _compact_json(structured_error("agent_lens_query_failed", "The Agent Lens request failed."))
        except Exception:
            if deadline_expired.is_set():
                ctx.mark_error("tool_timeout")
                return _compact_json(
                    structured_error(
                        "tool_timeout",
                        "The Agent Lens request exceeded the MCP tool deadline.",
                        retryable=True,
                    )
                )
            raise
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()
            close = getattr(response, "close", None)
            if callable(close):
                close()

    return _compact_json(_truncate_response(_project_response(tool_name, result)))


def _detail(payload: Dict[str, Any]) -> str:
    """Pull a sanitized Huma validation detail from a bounded JSON object."""
    detail = payload.get("detail") or payload.get("title")
    if detail is None:
        return "the request did not satisfy the API schema"
    return sanitize_sensitive_text(detail, max_chars=512)


def _invalid_argument(message: str) -> str:
    """Return the standard envelope for an argument this module rejected locally."""
    return json.dumps(structured_error("agent_lens_invalid_request", message))


def get_insights_coverage(entity_name: str, project_name: str) -> str:
    """Report the first and latest weeks that have classified Insights turns."""
    return _agent_lens_request(
        "get_insights_coverage",
        "GET",
        LATEST_WEEK_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


def get_clustering_status(entity_name: str, project_name: str) -> str:
    """Report the latest successful clustering run per signature type."""
    return _agent_lens_request(
        "get_clustering_status",
        "GET",
        CLUSTERING_STATUS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


def get_category_breakdowns(entity_name: str, project_name: str, start_at: str, end_at: str) -> str:
    """Return intent/failure category and cluster counts for a bounded range."""
    try:
        window = _validated_window(start_at, end_at)
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
        "get_category_breakdowns",
        "GET",
        CATEGORY_BREAKDOWNS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        params=window,
    )


def list_category_example_turns(
    entity_name: str,
    project_name: str,
    signature_type: str,
    category_id: str,
    start_at: str,
    end_at: str,
    topic_ids: Optional[List[str]] = None,
    limit: int = 10,
    cursor: Optional[str] = None,
) -> str:
    """Page example turns for one intent or failure category."""
    if signature_type not in {"intent", "failure"}:
        return _invalid_argument('signature_type must be "intent" or "failure"')
    if not str(category_id).strip():
        return _invalid_argument("category_id must be a non-empty category identifier")
    if not 1 <= limit <= MAX_EXAMPLE_LIMIT:
        return _invalid_argument(f"limit must be between 1 and {MAX_EXAMPLE_LIMIT}")
    try:
        params: Dict[str, Any] = dict(_validated_window(start_at, end_at))
        if topic_ids:
            params["topic_ids[]"] = _bounded_list(topic_ids, MAX_TOPIC_IDS, "topic_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    return _agent_lens_request(
        "list_category_example_turns",
        "GET",
        _category_examples_path(signature_type, category_id),
        entity_name,
        project_name,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "signature_type": signature_type,
        },
        params=params,
    )


def get_failure_attributions(entity_name: str, project_name: str, trace_ids: List[str]) -> str:
    """Return Agent Lens failure attribution for the requested trace IDs."""
    try:
        ids = _bounded_list(trace_ids, MAX_TRACE_IDS, "trace_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
        "get_failure_attributions",
        "POST",
        FAILURE_ATTRIBUTIONS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name, "trace_count": len(ids)},
        body={"trace_ids": ids},
    )


def list_tags(entity_name: str, project_name: str) -> str:
    """List the Agent Lens tag catalog for the project."""
    return _agent_lens_request(
        "list_tags",
        "GET",
        TAGS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


def get_conversation_tags(entity_name: str, project_name: str, conversation_ids: List[str]) -> str:
    """Return every tag applied to the given conversations, with provenance."""
    try:
        ids = _bounded_list(conversation_ids, MAX_CONVERSATION_IDS, "conversation_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
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


def list_tagged_conversations(entity_name: str, project_name: str, tag_ids: List[str]) -> str:
    """List conversation IDs carrying any of the given tag IDs."""
    try:
        ids = _bounded_uuid_list(tag_ids, MAX_TAG_IDS, "tag_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
        "list_tagged_conversations",
        "POST",
        TAGGED_CONVERSATIONS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name, "tag_count": len(ids)},
        body={"tag_ids": ids},
    )


def get_tag_distribution(
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
    return _agent_lens_request(
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


_WINDOW_PARAMS = """start_at : str
    Inclusive RFC 3339 start timestamp, e.g. "2026-09-01T00:00:00Z".
end_at : str
    Exclusive RFC 3339 end timestamp. The range must be nonempty and at most
    30 days; longer ranges are rejected before the request is sent."""


GET_INSIGHTS_COVERAGE_TOOL_DESCRIPTION = """Report which weeks have Agent Lens Insights data for a project.

<when_to_use>
Call this FIRST when answering any Insights question. Insights are produced by a
periodic classification job, so a project can have plenty of traces but no
classified turns, and the usable range rarely reaches today. Use the returned
first/latest weeks to choose a start_at/end_at that will actually contain data,
instead of guessing a range and getting an empty result.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.

Returns
-------
JSON with {"data": {"first_week", "latest_week"}}. Either may be null when the
project has no classified turns at all.
"""


GET_CLUSTERING_STATUS_TOOL_DESCRIPTION = """Report the latest successful Agent Lens clustering run per signature type.

<when_to_use>
Use to tell whether intent and failure clusters are fresh before relying on
cluster labels, and to explain a stale or missing breakdown. Each entry reports
the window that run covered, so a cluster absent from a recent range may simply
predate the latest run rather than having disappeared.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.

Returns
-------
JSON with {"data": [{"signature_type", "completed_at", "window_start",
"window_end"}]} -- at most one run per signature type.
"""


GET_CATEGORY_BREAKDOWNS_TOOL_DESCRIPTION = f"""Summarize Agent Lens intent and failure categories over a time range.

<when_to_use>
This is the main aggregate view: "what are users asking for, what is failing,
and how bad is it?". Returns per-category turn and conversation counts with
frustration counts, average latency and average cost per turn, plus the cluster
breakdowns beneath each category and failure-severity counts.

Start from get_agent_lens_insights_coverage_tool to pick a populated range. To
see the individual turns behind any number here, follow up with
list_agent_lens_category_example_turns_tool, then pass returned trace IDs to
get_agent_lens_failure_attributions_tool when failure detail is needed.
</when_to_use>

<reading_the_response>
Each entry mixes the two category families, and passing one where the other is
expected returns zero rows rather than an error:

- The top-level `category` is an INTENT category (what the user wanted), e.g.
  "action_request". Use it as `intent_category`, or with `signature_type="intent"`.
- `counts[].category` and `failure_breakdowns[].category` are FAILURE categories
  (what went wrong), e.g. "requirement_violation", plus the sentinel
  "no_failure" for turns that succeeded. Use these as `failure_category`, or
  with `signature_type="failure"`. "no_failure" is a count, not a drillable
  category.
- `cluster_breakdowns[].id` is a topic identifier, not a category. Pass one or
  more of these as `topic_ids` to the example-turn tool.
</reading_the_response>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
{_WINDOW_PARAMS}

Returns
-------
JSON with {{"data": [CategoryBreakdown]}}, each holding "counts",
"cluster_breakdowns", "failure_breakdowns" and "failure_severity_counts".
"""


LIST_CATEGORY_EXAMPLE_TURNS_TOOL_DESCRIPTION = f"""Page example turns for one Agent Lens intent or failure category.

<when_to_use>
Use to ground a category or cluster in concrete traces -- "show me turns from
this failure category" -- after get_agent_lens_category_breakdowns_tool names it.
Returns bounded identifiers and classification/failure metadata, but removes
the upstream `agent_message` and `message` bodies. Fetch trace content itself
with get_weave_agent_trace_tool or query_weave_traces_tool.

Pass the returned "next_cursor" back as `cursor` to page. Treat the cursor as
opaque and keep every other argument identical across pages.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
signature_type : str
    "intent" or "failure" -- which category family `category_id` belongs to.
    It must match where `category_id` came from, or the result is empty:
    "intent" for a top-level `category`, "failure" for a `counts[].category`
    or `failure_breakdowns[].category`.
category_id : str
    The category identifier from get_agent_lens_category_breakdowns_tool.
{_WINDOW_PARAMS}
topic_ids : list[str], optional
    Restrict to specific topics within the category (at most {MAX_TOPIC_IDS}).
limit : int, optional
    Turns per page, 1-{MAX_EXAMPLE_LIMIT} (default 10).
cursor : str, optional
    Opaque "next_cursor" from the previous page.

Returns
-------
JSON with bounded example-turn metadata, including "conversation_id" and
"trace_id", plus "next_cursor". Message bodies are omitted. A null
"next_cursor" means the last page.
"""


GET_FAILURE_ATTRIBUTIONS_TOOL_DESCRIPTION = f"""Fetch Agent Lens failure attribution for specific traces.

<when_to_use>
Use after list_agent_lens_category_example_turns_tool (or another trusted source
of trace IDs) when you need the failure reason, severity, and evidence spans for
those exact turns. This endpoint does not search by category; discover the
relevant trace IDs first and keep the request bounded.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
trace_ids : list[str]
    1-{MAX_TRACE_IDS} trace identifiers.

Returns
-------
JSON with {{"data": [TurnFailureAttribution]}}, each holding "trace_id",
"failure_reason", "failure_severity", and "failure_evidence_span_ids".
"""


LIST_TAGS_TOOL_DESCRIPTION = """List the Agent Lens tag catalog for a project.

<when_to_use>
Call this before filtering by tag. Tags are project-defined and the filtering
endpoint accepts tag IDs, not names. Use the returned ID with
list_agent_lens_tagged_conversations_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.

Returns
-------
JSON with {"data": [Tag]}, including each tag's "id", "name", criteria,
display color, agent filters, judge status, and timestamps.
"""


GET_CONVERSATION_TAGS_TOOL_DESCRIPTION = f"""Fetch Agent Lens conversation tags, with provenance, for specific conversations.

<when_to_use>
Use when you already have conversation IDs -- from
list_agent_lens_tagged_conversations_tool, an Insights drilldown, or a Weave
trace query -- and need to know how each was labelled and by whom. Each tag
reports its "source" (human or judge), the applying "wb_user_id" or
"judge_version", and a "rationale", which is what distinguishes a reviewed
judgement from an automated one.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
conversation_ids : list[str]
    1-{MAX_CONVERSATION_IDS} conversation identifiers.

Returns
-------
JSON with {{"data": [ConversationTag]}} holding "conversation_id", "trace_id",
"tag", "source", "rationale", agent identity and timestamps. Untagged
conversations are simply absent.
"""


LIST_TAGGED_CONVERSATIONS_TOOL_DESCRIPTION = f"""List Agent Lens conversation IDs carrying any of the given tags.

<when_to_use>
Use to go from tag to conversations -- "which conversations were tagged
escalated?". Matching is OR across the supplied tag IDs. Get IDs from
list_agent_lens_tags_tool first, then pass the returned conversation
IDs to get_agent_lens_conversation_tags_tool for provenance or to the Weave
trace tools for content.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
tag_ids : list[str]
    1-{MAX_TAG_IDS} tag UUIDs; a conversation matches if it carries any.

Returns
-------
JSON with {{"data": [str]}} -- the matching conversation IDs.
"""


GET_TAG_DISTRIBUTION_TOOL_DESCRIPTION = f"""Return Agent Lens conversation tag counts bucketed over time.

<when_to_use>
Use for trend questions -- "is this failure tag increasing?" -- rather than
fetching every tagged conversation and counting client-side. Bounds are epoch
milliseconds, unlike the RFC 3339 Insights tools.

Choose `time_bucket_seconds` to fit the range: 3600 for a day, 86400 for a
month. Very small buckets over a long range produce many buckets and may be
truncated to the response budget.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
after_ms : int
    Inclusive lower bound, epoch milliseconds.
before_ms : int
    Exclusive upper bound, epoch milliseconds; must exceed `after_ms`.
time_bucket_seconds : int
    Bucket width in seconds, 1-{MAX_TIME_BUCKET_SECONDS}.

Returns
-------
JSON with {{"data": {{"time_bucket_seconds", "after_ms", "before_ms",
"buckets": [{{"time_bucket_start_ms", "tag_counts"}}]}}}}.
"""
