"""Async client helpers for the hosted W&B Agent (ARIA).

The endpoint and payload shapes in this module follow the published service
schema at https://wb-agent.wandb.ai/openapi.json.
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import WB_AGENT_BASE_URL
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

TERMINAL_STATES = frozenset({"completed", "errored", "cancelled"})
MAX_WAIT_SECONDS = 30
POLL_INTERVAL_SECONDS = 1.0

ARIA_SEND_MESSAGE_TOOL_DESCRIPTION = """Start or continue an asynchronous conversation with ARIA, the hosted Weights & Biases agent (also called wb-agent).

Use ARIA when you want to hand off W&B-native work that may take a while, ask an agent with direct W&B context and data access to investigate something, or manipulate/analyze the W&B UI more directly instead of scraping W&B data yourself. Good tasks include comparing runs, diagnosing evals or Weave traces, inspecting a W&B page, and carrying out multi-step W&B workflows. Prefer the narrower query tools for simple, deterministic data lookups and the support bot for documentation questions.

This tool returns promptly by default because ARIA turns are asynchronous. Keep the returned turn_id. If state is queued or in_progress, call aria_get_turn until is_terminal is true. To continue a completed conversation, call this tool again with parent_turn_id set to the previous turn_id; omit entity and project for continuations. Set wait_seconds to wait briefly for a fast result, but the wait is always bounded.

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

Returns
-------
dict
    State, terminal status, turn/thread handles, latest assistant response, next action, and the full current turn snapshot. Errors are structured with retryable guidance and never include the W&B token.
"""

ARIA_GET_TURN_TOOL_DESCRIPTION = """Get the current state and result of an asynchronous ARIA turn.

Call this after aria_send_message returns queued or in_progress. With wait_seconds=0 it performs one poll; with 1-30 it polls only for that bounded interval and returns early if the turn completes, errors, or is cancelled. In-progress results include ARIA messages and tool activity available so far, plus a suggested poll interval. Reuse the same turn_id until is_terminal is true.

When state is completed, latest_response contains ARIA's newest assistant answer and the full turn snapshot remains available. Continue the conversation with aria_send_message(parent_turn_id=turn_id). When state is errored, inspect error_info. A transport/service error is returned separately with ok=false and keeps turn_id so this tool can be retried safely.

Parameters
----------
turn_id : str
    Exact ARIA turn_id returned by aria_send_message.
wait_seconds : int
    Seconds to poll before returning, from 0 to 30. Default 0 performs one status request.

Returns
-------
dict
    Current state, progress, latest response, continuation/poll guidance, and the full turn snapshot, or a structured error.
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
        details: Optional[str] = None,
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


def _response_details(response: httpx.Response) -> Optional[str]:
    try:
        body = response.json()
    except ValueError:
        body = response.text.strip()

    if isinstance(body, dict):
        body = body.get("detail") or body.get("error") or body
    if not body:
        return None
    if not isinstance(body, str):
        body = json.dumps(body, default=str)
    return body[:1000]


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
    summary: Dict[str, Any] = {
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_call_record_count": (len(tool_calls) if isinstance(tool_calls, list) else 0),
    }
    if isinstance(tool_calls, list) and tool_calls:
        latest = tool_calls[-1]
        if isinstance(latest, dict):
            summary["latest_tool_activity"] = {
                key: latest[key] for key in ("type", "name", "call_id", "is_error", "timestamp") if key in latest
            }
    return summary


def _turn_result(turn: Dict[str, Any]) -> Dict[str, Any]:
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

    return {
        "ok": True,
        "turn_id": turn_id,
        "thread_id": turn.get("thread_id"),
        "parent_turn_id": turn.get("parent_turn_id"),
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
        "turn": turn,
    }


async def send_aria_message(
    message: str,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    parent_turn_id: Optional[str] = None,
    wait_seconds: int = 0,
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

        resolved_api_key = _resolve_api_key(api_key)
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
            "Submitting ARIA turn: continuation=%s entity=%s project=%s wait=%ss",
            parent_turn_id is not None,
            entity,
            project,
            wait_seconds,
        )
        turn = await _request_turn(client, "POST", "/api/v1/turns", payload=payload)
        turn = await _poll_turn(client, turn, wait_seconds)
        return _turn_result(turn)
    except AriaAPIError as exc:
        logger.warning("ARIA send failed: %s", exc.error_type)
        return exc.as_result(operation="send")
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def get_aria_turn(
    turn_id: str,
    wait_seconds: int = 0,
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
        return _turn_result(turn)
    except AriaAPIError as exc:
        logger.warning("ARIA get failed for turn %s: %s", turn_id, exc.error_type)
        return exc.as_result(turn_id=turn_id, operation="get")
    finally:
        if owns_client and client is not None:
            await client.aclose()
