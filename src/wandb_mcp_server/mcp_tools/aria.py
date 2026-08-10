"""Async client helpers for the hosted W&B Agent (ARIA).

The endpoint and payload shapes in this module follow the published service
schema at https://wb-agent.wandb.ai/openapi.json.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import WB_AGENT_BASE_URL
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

TERMINAL_STATES = frozenset({"completed", "errored", "cancelled"})
MAX_WAIT_SECONDS = 30
MAX_BATCH_TURNS = 20
POLL_INTERVAL_SECONDS = 1.0

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
are compact by default; set include_turn=true only when the complete raw ARIA
turn snapshot is actually needed.

Parameters
----------
message : str
    A focused request for ARIA. Include relevant W&B entity/project names, run IDs, call IDs, URLs, metrics, time windows, and the desired output shape.
entity : str, optional
    W&B entity for a new conversation. Omit to let ARIA use the account default. Must be omitted for continuations.
project : str, optional
    W&B project for a new conversation. Omit to let ARIA use the default. Must be omitted for continuations.
parent_turn_id : str, optional
    Completed ARIA turn to continue. Use the exact turn_id returned by the prior call.
wait_seconds : int
    Seconds to poll before returning, from 0 to 30. Default 0 returns the initial asynchronous handle immediately.
include_turn : bool
    Include the full raw turn snapshot. Default false avoids very large responses.

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

When state is completed, latest_response contains ARIA's newest assistant answer. Continue the conversation with aria_send_message(parent_turn_id=turn_id). When state is errored, inspect error_info. A transport/service error uses the MCP error state and keeps turn_id in its JSON payload so this tool can be retried safely. Results are compact by default; set include_turn=true only for the full raw snapshot.

Parameters
----------
turn_id : str
    Exact ARIA turn_id returned by aria_send_message.
wait_seconds : int
    Seconds to poll before returning, from 0 to 30. Default 0 performs one status request.
include_turn : bool
    Include the full raw turn snapshot. Default false avoids very large responses.

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
    One to 20 distinct ARIA turn IDs.
wait_seconds : int
    Shared polling window from 0 to 30 seconds. Default 0 performs one concurrent status request.
include_turn : bool
    Include each full raw turn snapshot. Default false avoids very large responses.

Returns
-------
dict
    Ordered per-turn results plus terminal, pending, and error counts and retry guidance.
"""


class AriaMCPToolError(Exception):
    """Signal a structured ARIA failure through MCP's native error channel."""

    def __init__(self, result: Dict[str, Any]) -> None:
        super().__init__()
        self.result = result

    def __str__(self) -> str:
        return json.dumps(self.result, default=str, separators=(",", ":"))


def raise_for_aria_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """Raise a JSON-preserving error when an ARIA operation failed entirely."""
    if result.get("ok") is False:
        raise AriaMCPToolError(result)
    return result


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
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.details = details

    def as_result(self, *, turn_id: Optional[str] = None, operation: str) -> Dict[str, Any]:
        error: Dict[str, Any] = {
            "type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.status_code is not None:
            error["status_code"] = self.status_code
        if self.details:
            error["details"] = self.details

        if turn_id and self.retryable:
            next_action = "Retry aria_get_turn with the same turn_id; the ARIA turn remains server-side."
        elif operation == "send" and self.error_type in {
            "request_timeout",
            "service_unavailable",
            "network_error",
        }:
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
        api = WandBApiManager.get_api(api_key)
        entity = await asyncio.to_thread(lambda: api.default_entity)
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
        base_url=WB_AGENT_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
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
        return value[:1000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [_bounded_details(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key)[:200]: _bounded_details(item, depth=depth + 1) for key, item in list(value.items())[:20]}
    return str(value)[:1000]


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


def _http_error(response: httpx.Response, *, safe_to_retry: bool) -> AriaAPIError:
    status_code = response.status_code
    details = _response_details(response)

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
        )
    if status_code >= 500:
        return AriaAPIError(
            "service_unavailable",
            "ARIA is currently unavailable or failed to process the request.",
            retryable=safe_to_retry,
            status_code=status_code,
            details=details,
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
        response = await client.request(method, path, json=payload)
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

    if not isinstance(turn, dict) or not turn.get("id") or not turn.get("state"):
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned a response without the required turn id or state.",
            retryable=safe_to_retry,
            status_code=response.status_code,
        )
    return turn


async def _fetch_turn(client: httpx.AsyncClient, turn_id: str) -> Dict[str, Any]:
    encoded_turn_id = quote(turn_id, safe="")
    return await _request_turn(client, "GET", f"/api/v1/turns/{encoded_turn_id}")


async def _poll_turn(
    client: httpx.AsyncClient,
    turn: Dict[str, Any],
    wait_seconds: int,
) -> Dict[str, Any]:
    if wait_seconds == 0 or turn.get("state") in TERMINAL_STATES:
        return turn

    deadline = time.monotonic() + wait_seconds
    while turn.get("state") not in TERMINAL_STATES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))
        turn = await _fetch_turn(client, str(turn["id"]))
    return turn


def _latest_assistant_response(turn: Dict[str, Any]) -> Optional[str]:
    message_sets = (turn.get("messages"), turn.get("updated_messages"))
    for messages in message_sets:
        if not isinstance(messages, list):
            continue
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
        tool_error_count = sum(
            isinstance(record, dict) and record.get("type") == "response" and record.get("is_error") is True
            for record in tool_calls
        )
        if tool_error_count:
            summary["tool_error_count"] = tool_error_count
            summary["internal_error_note"] = (
                f"ARIA encountered {tool_error_count} internal tool error(s) and continued working."
            )
    if isinstance(tool_calls, list) and tool_calls:
        latest = tool_calls[-1]
        if isinstance(latest, dict):
            latest_activity = {
                key: latest[key] for key in ("type", "name", "call_id", "is_error", "timestamp") if key in latest
            }
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
        if activity.get("type") == "response" and activity.get("is_error"):
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
    try:
        if not isinstance(message, str) or not message.strip():
            raise AriaAPIError(
                "invalid_request",
                "message must be a non-empty string.",
                retryable=False,
            )
        _validate_wait_seconds(wait_seconds)

        if entity is not None:
            if not isinstance(entity, str) or not entity.strip():
                raise AriaAPIError(
                    "invalid_request",
                    "entity must be a non-empty string when provided.",
                    retryable=False,
                )
            entity = entity.strip()
        if project is not None:
            if not isinstance(project, str) or not project.strip():
                raise AriaAPIError(
                    "invalid_request",
                    "project must be a non-empty string when provided.",
                    retryable=False,
                )
            project = project.strip()

        if parent_turn_id is not None:
            if not isinstance(parent_turn_id, str) or not parent_turn_id.strip():
                raise AriaAPIError(
                    "invalid_request",
                    "parent_turn_id cannot be empty.",
                    retryable=False,
                )
            if entity is not None or project is not None:
                raise AriaAPIError(
                    "invalid_request",
                    "entity and project must be omitted when parent_turn_id is set.",
                    retryable=False,
                )
            parent_turn_id = parent_turn_id.strip()

        resolved_api_key = _resolve_api_key(api_key)
        if parent_turn_id is None and project is not None and entity is None:
            entity = await _resolve_default_entity(resolved_api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        payload: Dict[str, Any] = {"user_prompt": message.strip()}
        if parent_turn_id is not None:
            payload["parent_turn_id"] = parent_turn_id
        else:
            if entity is not None:
                payload["entity"] = entity
            if project is not None:
                payload["project"] = project

        logger.info(
            "Submitting ARIA turn: continuation=%s entity=%s project=%s wait=%ss",
            parent_turn_id is not None,
            entity,
            project,
            wait_seconds,
        )
        turn = await _request_turn(client, "POST", "/api/v1/turns", payload=payload)
        turn = await _poll_turn(client, turn, wait_seconds)
        return _turn_result(turn, include_turn=include_turn)
    except AriaAPIError as exc:
        logger.warning("ARIA send failed: %s", exc.error_type)
        return exc.as_result(operation="send")
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
    try:
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise AriaAPIError(
                "invalid_request",
                "turn_id must be a non-empty string.",
                retryable=False,
            )
        _validate_wait_seconds(wait_seconds)
        resolved_api_key = _resolve_api_key(api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        turn = await _fetch_turn(client, turn_id)
        turn = await _poll_turn(client, turn, wait_seconds)
        return _turn_result(turn, include_turn=include_turn)
    except AriaAPIError as exc:
        logger.warning("ARIA get failed for turn %s: %s", turn_id, exc.error_type)
        return exc.as_result(turn_id=turn_id, operation="get")
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def _fetch_turn_as_result(
    client: httpx.AsyncClient,
    turn_id: str,
    *,
    include_turn: bool,
) -> Dict[str, Any]:
    try:
        turn = await _fetch_turn(client, turn_id)
        return _turn_result(turn, include_turn=include_turn)
    except AriaAPIError as exc:
        return exc.as_result(turn_id=turn_id, operation="get")


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
            if not isinstance(turn_id, str) or not turn_id.strip():
                raise AriaAPIError(
                    "invalid_request",
                    "Every turn_id must be a non-empty string.",
                    retryable=False,
                )
            normalized_ids.append(turn_id.strip())
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
        pending_ids = list(normalized_ids)
        results_by_id: Dict[str, Dict[str, Any]] = {}

        while pending_ids:
            fetched = await asyncio.gather(
                *(_fetch_turn_as_result(client, turn_id, include_turn=include_turn) for turn_id in pending_ids)
            )
            for turn_id, result in zip(pending_ids, fetched):
                results_by_id[turn_id] = result

            pending_ids = [
                turn_id
                for turn_id in pending_ids
                if (results_by_id[turn_id].get("ok") is True and not results_by_id[turn_id].get("is_terminal"))
                or (
                    results_by_id[turn_id].get("ok") is False
                    and results_by_id[turn_id].get("error", {}).get("retryable") is True
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
            return {
                "ok": False,
                "error": {
                    "type": "batch_lookup_failed",
                    "message": "ARIA could not retrieve any of the requested turns.",
                    "retryable": retryable,
                    "details": {"results": ordered_results},
                },
                "turn_ids": normalized_ids,
                "next_action": (
                    "Retry aria_get_turns with the same turn_ids."
                    if retryable
                    else "Inspect each turn error and correct credentials or turn IDs."
                ),
            }

        return {
            "ok": True,
            "requested_count": len(normalized_ids),
            "terminal_count": terminal_count,
            "pending_count": len(still_pending),
            "error_count": error_count,
            "pending_turn_ids": still_pending,
            "failed_turn_ids": failed_turn_ids,
            "results": ordered_results,
            "next_action": (
                "Call aria_get_turns again with pending_turn_ids."
                if still_pending
                else "All retrievable turns are terminal; continue completed conversations with aria_send_message."
            ),
        }
    except AriaAPIError as exc:
        logger.warning("ARIA batch get failed: %s", exc.error_type)
        return exc.as_result(operation="get")
    finally:
        if owns_client and client is not None:
            await client.aclose()
