"""Async client helpers for the hosted W&B Agent (ARIA).

The endpoint and payload shapes in this module follow the published service
schema at https://wb-agent.wandb.ai/openapi.json.
"""

import asyncio
import ipaddress
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import Any, Dict, List, Literal, Optional, Sequence
from urllib.parse import quote
from weakref import WeakKeyDictionary

import httpx
from pydantic import BaseModel, Field

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.config import MAX_RESPONSE_TOKENS, WB_AGENT_BASE_URL, resolve_aria_base_url
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text, sanitize_sensitive_value
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_urls import public_forge_aria_url

logger = get_rich_logger(__name__)

TURN_STATES = frozenset({"queued", "in_progress", "completed", "errored", "cancelled"})
TERMINAL_STATES = frozenset({"completed", "errored", "cancelled"})
MAX_WAIT_SECONDS = 30
MAX_OUTBOUND_CONCURRENCY = 8
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 50
MAX_LIST_OFFSET = 10_000
MAX_PROMPT_PREVIEW_CHARS = 500
# Cooperative attribution in the turn body so ARIA can separate programmatic
# MCP-driven turns from web UI and mobile usage.
ARIA_CLIENT_PROMPT_PART = {"type": "wandb_client", "platform": "agent"}
MAX_MESSAGE_BYTES = 32 * 1024
MAX_IDENTIFIER_LENGTH = 512
MAX_SCOPE_LENGTH = 512
MAX_THREAD_TITLE_LENGTH = 255
MAX_UPSTREAM_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROGRESS_TOOL_RECORDS = 100
MAX_PROGRESS_NODES_PER_RECORD = 200
MAX_PROGRESS_NESTING_DEPTH = 8
MAX_TOOL_NAME_LOOKBACK_RECORDS = 1_000
MAX_FALLBACK_TITLE_CHARS = 80
MAX_COMPACT_TURN_MESSAGES = 12
MAX_COMPACT_TURN_TOOL_RECORDS = 16
POLL_INTERVAL_SECONDS = 1.0
POLL_BACKOFF_MULTIPLIERS = (1, 2, 3)
POLL_DEADLINE_SAFETY_SECONDS = 0.5
MIN_RETRY_AFTER_MS = 1_000
MAX_RETRY_AFTER_MS = 30_000
MAX_ALLOWED_NETWORK_DOMAINS = 32
MAX_DOMAIN_LENGTH = 253
MAX_PERMISSION_REASON_CHARS = 500
MAX_AUTO_PERMISSION_GRANTS = 2
ELICIT_TIMEOUT_SECONDS = 60.0
MAX_PARALLEL_TURN_IDS = 15
MAX_USER_PROMPT_PREVIEW_CHARS = 1_200
MAX_TOOL_STEPS = 12
MAX_PARALLEL_MEMBER_RESPONSE_CHARS = 6_000

NetworkAccessMode = Literal["prompt", "auto_safe", "auto_public", "deny"]
VALID_NETWORK_ACCESS_MODES: frozenset[str] = frozenset({"prompt", "auto_safe", "auto_public", "deny"})

# Read-only public ML research, documentation, and package registries safe for
# unattended `network_access_mode="auto_safe"` grants (extendable by operators
# via comma-separated `MCP_ARIA_AUTO_SAFE_DOMAINS`).
DEFAULT_SAFE_NETWORK_DOMAINS: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "export.arxiv.org",
        "openreview.net",
        "aclanthology.org",
        "pypi.org",
        "files.pythonhosted.org",
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "pytorch.org",
        "docs.python.org",
    }
)
_RESERVED_EXACT_DOMAINS: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "instance-data",
    }
)
_RESERVED_DOMAIN_SUFFIXES: tuple[str, ...] = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home",
    ".arpa",
    ".svc",
    ".cluster.local",
    ".corp",
)
_DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class _AriaNetworkAccessElicitation(BaseModel):
    """MCP form elicitation schema when ARIA requests sandbox network egress."""

    grant_access: bool = Field(
        default=True,
        description="Grant ARIA outbound sandbox access to the requested domain(s) and continue the turn.",
    )


_OUTBOUND_LIMITERS: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()
_OUTBOUND_LIMITERS_LOCK = threading.Lock()

try:
    _INSTALLED_VERSION = distribution_version("wandb-mcp-server")
except PackageNotFoundError:  # pragma: no cover - source-only developer fallback
    from wandb_mcp_server import __version__ as _INSTALLED_VERSION

ARIA_CLIENT_IDENTITY = f"wandb-mcp-server/{_INSTALLED_VERSION}"


def _outbound_limiter() -> asyncio.Semaphore:
    """Return the shared eight-request limiter for the active server loop."""
    loop = asyncio.get_running_loop()
    with _OUTBOUND_LIMITERS_LOCK:
        limiter = _OUTBOUND_LIMITERS.get(loop)
        if limiter is None:
            limiter = asyncio.Semaphore(MAX_OUTBOUND_CONCURRENCY)
            _OUTBOUND_LIMITERS[loop] = limiter
        return limiter


class _SanitizeAriaHTTPLogFilter(logging.Filter):
    """Remove ARIA origins, turn identifiers, and internal hosts from HTTP logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and len(record.args) >= 2:
            rendered_url = str(record.args[1])
            for marker in ("/api/v1/turns", "/api/v1/threads"):
                if marker in rendered_url:
                    suffix = f"{marker}/<redacted>" if f"{marker}/" in rendered_url else marker
                    args = list(record.args)
                    args[1] = f"<aria-service>{suffix}"
                    record.args = tuple(args)
                    break

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
ARIA is W&B's own agent: it has the caller's W&B context, runs code against the
W&B and Weave APIs, and can read and change the W&B UI. ARIA can:
- Workspaces: create/edit saved views, filters, grouping, sorting, sections,
  panels, run colours, pinned/baseline runs; describe what a workspace shows.
- Runs & Sweeps: rank, compare, diff configs, find regressions, read console
  logs, tag/rename/move/delete runs, export tables, run/analyse sweeps.
- Reports, Artifacts & Registry: create/edit reports, PDF export, lineage,
  versions, promotion, storage cleanup.
- Weave & Agent Lens: query traces, debug evaluations, prompts/datasets,
  monitors, conversations, and insights.
- Platform & Investigations: automations, Launch, Serverless training/inference,
  notebooks, Sandboxes, GitHub repos, and open-ended project audits.
Earlier ARIA conversations are found with aria_list_threads.
</when_to_use>

ARIA turns are asynchronous: keep `turn_id` and poll `aria_get_turn` until `is_terminal` is true (audits routinely take 2 to 8 minutes; do not start a duplicate thread while working). Set `wait_seconds` (up to 30) to wait server-side without client sleep calls. Continue a completed turn with `parent_turn_id` (omit `entity`, `project`, and `title`). Root turns may omit `entity`/`project` (auto-resolves default entity when only `project` is set) and auto-title when `title` is omitted. Set `include_turn=true` only when a bounded raw snapshot (`_truncation` when oversized) is needed.

Parameters
----------
message : str
    Focused request (up to 32 KiB UTF-8) with relevant entity/project, IDs, metrics, windows, and output shape.
entity : str, optional
    W&B entity for a new conversation (<= 512 chars). Omit on continuations.
project : str, optional
    W&B project for a new conversation (<= 512 chars). Omit on continuations.
title : str, optional
    Short name for a new conversation (<= 255 chars; omit with `parent_turn_id`).
parent_turn_id : str, optional
    Completed `turn_id` to continue (<= 512 chars).
wait_seconds : int
    Seconds to poll server-side before returning (0-30, default 0).
include_turn : bool
    Include bounded raw turn snapshot (default false).
allowed_network_domains : list[str], optional
    Public DNS hostnames (e.g. `["arxiv.org"]`) added to ARIA's sandbox egress allowlist; merged with parent grants on continuations.
network_access_mode : {"prompt", "auto_safe", "auto_public", "deny"}, optional
    Policy for ARIA `network_access` `permission_requests`: `"prompt"` (default; elicits or pauses with `phase="awaiting_permission"`), `"auto_safe"` (auto-grants vetted registries like `arxiv.org`, `pypi.org`, `huggingface.co`), `"auto_public"` (only when the host harness runs in user-authorized full-auto mode), or `"deny"` (continues offline).

Returns
-------
dict
    Compact state, progress, scope, turn/thread handles, latest response, and next action. Errors use the MCP error state and never include the W&B token.
"""

ARIA_GET_TURN_TOOL_DESCRIPTION = """Get the current state and result of an asynchronous ARIA turn.

<when_to_use>
Call after aria_send_message returns queued or in_progress, or to read the answer
of a turn found with aria_get_thread. Always rely on is_terminal rather than the
presence of latest_response because ARIA can emit a partial response while still
working.
</when_to_use>

With wait_seconds=0 this performs one poll; with 1-30 it polls server-side for that bounded interval and returns early if the turn completes, errors, or is cancelled. Do not insert separate sleep or wait tool calls between polls because wait_seconds=30 already waits server-side. Open-ended ARIA investigations routinely run 2 to 8 minutes, and individual sandbox scripts or final synthesis passes can run 30 to 90 seconds without changing message_count or tool_call_record_count; treat unchanged counters as active work rather than a hang, reuse the same turn_id until is_terminal is true, and do not start a duplicate ARIA thread while state is queued or in_progress. In-progress results include a phase, meaningful status text, elapsed_seconds when available, message/tool counts, and latest tool activity, plus a suggested poll interval.

When state is completed, check progress.phase: if "awaiting_permission" or "awaiting_user_input", ARIA paused on permission_requests or agent_questions and can be resumed from saved workspace state via aria_send_message(parent_turn_id=turn_id, ...). Otherwise latest_response contains ARIA's answer. When state is errored, inspect error_info. A transport/service error uses the MCP error state and keeps turn_id in its JSON payload so this tool can be retried safely. Results are compact by default; set include_turn=true only for a bounded raw snapshot, which may contain `_truncation` metadata.

Parameters
----------
turn_id : str
    Exact ARIA turn_id returned by aria_send_message or aria_get_thread, up to 512 characters.
wait_seconds : int
    Seconds to poll before returning, from 0 to 30. Default 0 performs one status request.
include_turn : bool
    Include a bounded raw turn snapshot. Oversized data is truncated with metadata. Default false avoids very large responses.
network_access_mode : {"prompt", "auto_safe", "auto_public", "deny"}, optional
    Optional policy when a polled turn pauses on network_access permission_requests in read-write mode (inherits MCP_ARIA_NETWORK_ACCESS_MODE when omitted).
parallel_turn_ids : list[str], optional
    Optional additional ARIA turn_ids (up to 15) spawned for parallel asynchronous sub-agent tasks; polls `[turn_id, *parallel_turn_ids]` concurrently within `wait_seconds` and adds `parallel_turns` plus `parallel_summary`.

Returns
-------
dict
    Compact current state, progress, latest response, optional parallel sub-agent statuses, and continuation/poll guidance, or a native MCP error containing JSON details.
"""

ARIA_LIST_THREADS_TOOL_DESCRIPTION = """Find the caller's previous ARIA conversations (their chat history with ARIA, the W&B agent), most recently active first.

<when_to_use>
This is the only source of ARIA conversation history. Call it whenever the caller
refers to an earlier, previous, past, or most recent ARIA conversation, chat, or
thread, or asks what ARIA said, found, or concluded before, for example "find my
earlier ARIA conversation about project X" or "what did ARIA say about my eval runs
last week". Do not answer that no conversation history is available without calling
this first. Filter by W&B entity and project when known. Then call aria_get_thread
with the chosen thread_id. Do not use this to poll a turn you just started; use
aria_get_turn.
</when_to_use>

Each thread includes thread_id, title, entity/project scope, pinned/visibility
flags, creation and last-turn timestamps, and legacy prompt_preview (null on
titled threads). Results are paginated: when
has_more is true, call again with offset=next_offset.

Parameters
----------
entity : str, optional
    W&B entity filter, up to 512 characters.
project : str, optional
    W&B project filter, up to 512 characters.
limit : int
    Threads to return, from 1 to 50. Default 20.
offset : int
    Threads to skip for pagination, from 0 to 10000. Default 0.

Returns
-------
dict
    Compact thread summaries plus pagination guidance, or a native MCP error containing JSON details.
"""

ARIA_GET_THREAD_TOOL_DESCRIPTION = """Get one ARIA conversation thread and a compact list of its turns, newest first.

<when_to_use>
Use after aria_list_threads to inspect an earlier conversation: see what was asked,
which turns finished, and which turn to continue. Call aria_get_turn with a turn_id
to read that turn's answer. To continue the conversation, call aria_send_message
with parent_turn_id set to the newest completed turn_id.
</when_to_use>

Each turn includes turn_id, parent_turn_id, state, is_terminal, and timestamps.
Prompts and answers are not included; fetch them with aria_get_turn. Turns are paginated: when has_more is true, call again with
offset=next_offset.

Parameters
----------
thread_id : str
    Exact thread_id from aria_list_threads or a prior ARIA result, up to 512 characters.
limit : int
    Turns to return, from 1 to 50. Default 20.
offset : int
    Turns to skip for pagination, from 0 to 10000. Default 0.

Returns
-------
dict
    Thread summary, compact turns, and continuation/pagination guidance, or a native MCP error containing JSON details.
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


def _compact_turn_snapshot(turn: Any, stats: Dict[str, int]) -> Dict[str, Any]:
    if not isinstance(turn, dict):
        return {"_truncated": True, "reason": "response_token_budget"}
    compacted: Dict[str, Any] = {
        "_truncated": True,
        "reason": "response_token_budget",
    }
    for key in (
        "id",
        "thread_id",
        "parent_turn_id",
        "state",
        "created_at",
        "updated_at",
        "wandb_entity",
        "wandb_project",
        "error_info",
        "agent_questions",
        "permission_requests",
    ):
        if key in turn:
            compacted[key] = _compact_response_value(
                turn[key],
                max_depth=3,
                max_string_chars=512,
                max_items=10,
                stats=stats,
            )

    messages = turn.get("messages")
    if isinstance(messages, list):
        compacted["message_count"] = len(messages)
        tail_messages = messages[-MAX_COMPACT_TURN_MESSAGES:]
        if len(messages) > len(tail_messages):
            omitted = len(messages) - len(tail_messages)
            compacted["messages_omitted"] = omitted
            stats["turn_messages_omitted"] = stats.get("turn_messages_omitted", 0) + omitted
        compact_messages = []
        for record in tail_messages:
            if isinstance(record, dict) and ("role" in record or "content" in record):
                slim = {k: v for k, v in record.items() if k != "message"}
            else:
                slim = record
            compact_messages.append(
                _compact_response_value(
                    slim,
                    max_depth=3,
                    max_string_chars=1_024,
                    max_items=10,
                    stats=stats,
                )
            )
        compacted["messages"] = compact_messages

    tool_calls = turn.get("tool_calls")
    if isinstance(tool_calls, list):
        compacted["tool_call_record_count"] = len(tool_calls)
        tail_tools = tool_calls[-MAX_COMPACT_TURN_TOOL_RECORDS:]
        if len(tool_calls) > len(tail_tools):
            omitted = len(tool_calls) - len(tail_tools)
            compacted["tool_calls_omitted"] = omitted
            stats["turn_tool_calls_omitted"] = stats.get("turn_tool_calls_omitted", 0) + omitted
        compact_tools = []
        for record in tail_tools:
            enriched = record
            if isinstance(record, dict) and "name" not in record:
                resolved_name = _resolve_tool_name(tool_calls, record)
                if resolved_name is not None:
                    enriched = {**record, "name": resolved_name}
            compact_tools.append(
                _compact_response_value(
                    enriched,
                    max_depth=3,
                    max_string_chars=512,
                    max_items=10,
                    stats=stats,
                )
            )
        compacted["tool_calls"] = compact_tools
    return compacted


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

    truncation_base: Dict[str, Any] = {
        "applied": True,
        "reason": "response_token_budget",
        "max_tokens": MAX_RESPONSE_TOKENS,
        "original_estimated_tokens": original_tokens,
    }

    if "turn" in sanitized and isinstance(sanitized.get("turn"), dict):
        turn_compaction: Dict[str, int] = {}
        compact_turn_candidate = dict(sanitized)
        compact_turn_candidate["turn"] = _compact_turn_snapshot(sanitized["turn"], turn_compaction)
        compact_truncation = dict(truncation_base)
        compact_truncation["compaction"] = turn_compaction or {"turn_compacted": 1}
        compact_turn_candidate["_truncation"] = compact_truncation
        if _estimate_response_tokens(compact_turn_candidate) <= MAX_RESPONSE_TOKENS:
            return compact_turn_candidate

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
                {
                    key: turn[key]
                    for key in ("id", "thread_id", "parent_turn_id", "state", "created_at", "updated_at")
                    if key in turn
                }
            )
        candidate["turn"] = marker
        omitted_fields.append("turn")

    truncation = dict(truncation_base)
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
    params: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    async with _outbound_limiter():
        async with client.stream(
            method,
            path,
            json=payload,
            params=params,
            headers={
                "Accept-Encoding": "identity",
                # This is cooperative product attribution only. ARIA must not
                # use it for authentication, authorization, routing, quotas,
                # or billing. Supplying it per request also covers injected
                # request-scoped clients used by hosted deployments.
                "X-Wandb-Client": ARIA_CLIENT_IDENTITY,
            },
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
    retry_after_ms = _bounded_retry_after_ms(response)

    if status_code in {401, 403}:
        return AriaAPIError(
            "authentication_error",
            "ARIA rejected the W&B token or the user does not have ARIA access.",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 404:
        return AriaAPIError(
            "turn_not_found",
            "ARIA could not find a turn visible to the current W&B user.",
            retryable=False,
            status_code=status_code,
        )
    # Authentication and visibility failures intentionally use only stable
    # local messages. Other bounded upstream errors may retain sanitized
    # details when they are useful for correcting a request.
    details = _response_details(response)
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


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> tuple[int, Any]:
    safe_to_retry = method == "GET"
    try:
        outer_deadline = current_tool_deadline.get()
        if outer_deadline is None:
            response = await _perform_bounded_request(client, method, path, payload=payload, params=params)
        else:
            request_seconds = outer_deadline - POLL_DEADLINE_SAFETY_SECONDS - time.monotonic()
            if request_seconds <= 0:
                raise TimeoutError
            async with asyncio.timeout(request_seconds):
                response = await _perform_bounded_request(client, method, path, payload=payload, params=params)
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
        return response.status_code, response.json()
    except ValueError as exc:
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned a non-JSON response.",
            retryable=safe_to_retry,
            status_code=response.status_code,
        ) from exc


async def _request_turn(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_to_retry = method == "GET"
    status_code, turn = await _request_json(client, method, path, payload=payload)
    if not isinstance(turn, dict):
        raise AriaAPIError(
            "invalid_response",
            "ARIA returned a response without the required turn id or state.",
            retryable=safe_to_retry,
            status_code=status_code,
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
            status_code=status_code,
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


async def _poll_turn_with_status(
    client: httpx.AsyncClient,
    turn: Dict[str, Any],
    wait_seconds: int,
    *,
    confirmed_get: bool = False,
) -> tuple[Dict[str, Any], Optional[AriaAPIError]]:
    if wait_seconds == 0 or turn.get("state") in TERMINAL_STATES:
        return turn, None

    deadline = time.monotonic() + wait_seconds
    outer_deadline = current_tool_deadline.get()
    if outer_deadline is not None:
        deadline = min(deadline, outer_deadline - POLL_DEADLINE_SAFETY_SECONDS)
    attempt = 0
    while turn.get("state") not in TERMINAL_STATES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        multiplier = POLL_BACKOFF_MULTIPLIERS[min(attempt, len(POLL_BACKOFF_MULTIPLIERS) - 1)]
        await asyncio.sleep(min(POLL_INTERVAL_SECONDS * multiplier, remaining))
        if deadline - time.monotonic() <= 0:
            break
        try:
            turn = await _fetch_turn(client, str(turn["id"]))
            confirmed_get = True
            attempt += 1
        except AriaAPIError as exc:
            # The public dispatch timeout must remain available to serialize a
            # truthful pending result. If the request consumed that headroom,
            # return the last confirmed turn rather than racing the outer timer.
            outer_remaining = (
                None if outer_deadline is None else outer_deadline - POLL_DEADLINE_SAFETY_SECONDS - time.monotonic()
            )
            if exc.error_type == "request_timeout" and outer_remaining is not None and outer_remaining <= 0:
                break
            if confirmed_get and exc.retryable:
                return turn, exc
            raise
    return turn, None


async def _poll_turn(
    client: httpx.AsyncClient,
    turn: Dict[str, Any],
    wait_seconds: int,
) -> Dict[str, Any]:
    polled, _ = await _poll_turn_with_status(client, turn, wait_seconds)
    return polled


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


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _turn_elapsed_seconds(turn: Dict[str, Any], *, is_terminal: bool) -> Optional[float]:
    started = _parse_iso_timestamp(turn.get("created_at"))
    if started is None:
        return None
    end = _parse_iso_timestamp(turn.get("updated_at")) if is_terminal else None
    if end is None:
        end = datetime.now(timezone.utc)
    elapsed = (end - started).total_seconds()
    if not math.isfinite(elapsed):
        return None
    return round(max(0.0, elapsed), 1)


def _resolve_tool_name(tool_calls: list[Any], record: Dict[str, Any]) -> Optional[str]:
    explicit_name = record.get("name")
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name
    call_id = record.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return None
    for candidate in reversed(tool_calls[-MAX_TOOL_NAME_LOOKBACK_RECORDS:]):
        if (
            isinstance(candidate, dict)
            and candidate.get("call_id") == call_id
            and candidate.get("type") == "invocation"
        ):
            candidate_name = candidate.get("name")
            if isinstance(candidate_name, str) and candidate_name:
                return candidate_name
    return None


def _normalize_public_domain(raw_domain: Any) -> Optional[str]:
    """Validate and canonicalize one public DNS hostname, rejecting IP literals and internal zones."""
    if not isinstance(raw_domain, str):
        return None
    candidate = raw_domain.strip().rstrip(".")
    if not candidate or len(candidate) > MAX_DOMAIN_LENGTH:
        return None
    if any(ch in candidate for ch in ("/", ":", "@", "?", "#", "*", "\\", " ", "\t", "\r", "\n")):
        return None
    try:
        ascii_host = candidate.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if not ascii_host or len(ascii_host) > MAX_DOMAIN_LENGTH:
        return None
    if ascii_host in _RESERVED_EXACT_DOMAINS or ascii_host.endswith(_RESERVED_DOMAIN_SUFFIXES):
        return None
    try:
        ipaddress.ip_address(ascii_host.strip("[]"))
        return None
    except ValueError:
        pass
    labels = ascii_host.split(".")
    if len(labels) < 2:
        return None
    if labels[-1].isdigit():
        return None
    for label in labels:
        if not _DOMAIN_LABEL_RE.fullmatch(label):
            return None
    return ascii_host


def normalize_allowed_network_domains(raw_domains: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Validate caller-supplied `allowed_network_domains`, raising `AriaAPIError` on invalid input."""
    if raw_domains is None:
        return None
    if isinstance(raw_domains, (str, bytes)) or not isinstance(raw_domains, Sequence):
        raise AriaAPIError(
            "invalid_request",
            "allowed_network_domains must be a list of public DNS hostnames.",
            retryable=False,
        )
    if len(raw_domains) > MAX_ALLOWED_NETWORK_DOMAINS:
        raise AriaAPIError(
            "invalid_request",
            f"allowed_network_domains may include at most {MAX_ALLOWED_NETWORK_DOMAINS} hostnames.",
            retryable=False,
        )
    deduped: List[str] = []
    seen: set[str] = set()
    for item in raw_domains:
        normalized = _normalize_public_domain(item)
        if normalized is None:
            raise AriaAPIError(
                "invalid_request",
                "Each entry in allowed_network_domains must be a valid public DNS hostname "
                "(no schemes, ports, paths, wildcards, IP literals, or internal domains).",
                retryable=False,
            )
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _resolve_network_access_mode(
    explicit_mode: Optional[str],
) -> tuple[NetworkAccessMode, bool]:
    """Return `(mode, explicitly_configured)` from call argument or `MCP_ARIA_NETWORK_ACCESS_MODE`."""
    if explicit_mode is not None:
        cleaned = str(explicit_mode).strip().lower() if isinstance(explicit_mode, str) else ""
        if cleaned not in VALID_NETWORK_ACCESS_MODES:
            raise AriaAPIError(
                "invalid_request",
                "network_access_mode must be one of: prompt, auto_safe, auto_public, deny.",
                retryable=False,
            )
        return cleaned, True  # type: ignore[return-value]
    env_raw = (os.environ.get("MCP_ARIA_NETWORK_ACCESS_MODE") or "").strip().lower()
    if not env_raw:
        return "prompt", False
    if env_raw not in VALID_NETWORK_ACCESS_MODES:
        logger.warning("Ignoring invalid MCP_ARIA_NETWORK_ACCESS_MODE=%r; defaulting to prompt.", env_raw)
        return "prompt", False
    return env_raw, True  # type: ignore[return-value]


def _configured_safe_network_domains() -> frozenset[str]:
    extra_raw = os.environ.get("MCP_ARIA_AUTO_SAFE_DOMAINS") or ""
    if not extra_raw.strip():
        return DEFAULT_SAFE_NETWORK_DOMAINS
    merged = set(DEFAULT_SAFE_NETWORK_DOMAINS)
    for token in extra_raw.split(","):
        normalized = _normalize_public_domain(token)
        if normalized is not None:
            merged.add(normalized)
    return frozenset(merged)


def _turn_allowed_network_domains(turn: Dict[str, Any]) -> List[str]:
    perms = turn.get("permissions")
    if not isinstance(perms, dict):
        return []
    raw_list = perms.get("allowed_network_domains")
    if not isinstance(raw_list, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw_list:
        norm = _normalize_public_domain(item)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _inspect_permission_requests(turn: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize ARIA `permission_requests` and partition valid vs rejected domains."""
    raw_reqs = turn.get("permission_requests")
    if not isinstance(raw_reqs, list) or not raw_reqs:
        return {
            "has_requests": False,
            "all_network_access": False,
            "valid_domains": [],
            "rejected_domains": [],
            "reasons": [],
        }
    valid_domains: List[str] = []
    seen_valid: set[str] = set()
    rejected_domains: List[str] = []
    reasons: List[str] = []
    all_network_access = True

    for item in raw_reqs:
        if not isinstance(item, dict):
            all_network_access = False
            continue
        if item.get("type") != "network_access":
            all_network_access = False
            continue
        reason = item.get("reason")
        if isinstance(reason, str) and reason.strip():
            bounded_reason = sanitize_sensitive_text(reason.strip(), max_chars=MAX_PERMISSION_REASON_CHARS)
            if bounded_reason and bounded_reason not in reasons:
                reasons.append(bounded_reason)
        domains = item.get("domains")
        if isinstance(domains, list):
            for dom in domains[:MAX_ALLOWED_NETWORK_DOMAINS]:
                norm = _normalize_public_domain(dom)
                if norm is not None:
                    if norm not in seen_valid:
                        seen_valid.add(norm)
                        valid_domains.append(norm)
                elif isinstance(dom, str) and dom.strip():
                    rejected_domains.append(sanitize_sensitive_text(dom.strip(), max_chars=128))

    return {
        "has_requests": True,
        "all_network_access": all_network_access and bool(valid_domains or rejected_domains),
        "valid_domains": valid_domains,
        "rejected_domains": rejected_domains,
        "reasons": reasons,
    }


def _has_agent_questions(turn: Dict[str, Any]) -> bool:
    questions = turn.get("agent_questions")
    return isinstance(questions, list) and len(questions) > 0


def _client_supports_elicitation(ctx: Any) -> bool:
    if ctx is None:
        return False
    if (os.environ.get("MCP_SESSION_MODE") or "").strip().lower() == "stateless":
        return False
    try:
        session = ctx.request_context.session
        client_params = getattr(session, "client_params", None)
        capabilities = getattr(client_params, "capabilities", None)
        return getattr(capabilities, "elicitation", None) is not None
    except Exception:
        return False


async def _elicit_network_permission(
    ctx: Any,
    *,
    domains: List[str],
    reasons: List[str],
) -> Optional[bool]:
    """Prompt the MCP client via `ctx.elicit` when supported; return True/False or None on fallback."""
    if not _client_supports_elicitation(ctx):
        return None
    reason_text = "; ".join(reasons) if reasons else "Complete the current ARIA task."
    prompt = (
        f"ARIA is requesting outbound sandbox network access to {', '.join(domains)}.\n"
        f"Reason: {reason_text}\n"
        "Grant access and continue from ARIA's saved workspace state?"
    )
    deadline = current_tool_deadline.get()
    timeout = ELICIT_TIMEOUT_SECONDS
    if deadline is not None:
        remaining = deadline - time.monotonic() - 1.0
        if remaining <= 0.5:
            return None
        timeout = min(timeout, remaining)
    try:
        outcome = await asyncio.wait_for(
            ctx.elicit(message=prompt, schema=_AriaNetworkAccessElicitation),
            timeout=timeout,
        )
    except Exception as exc:
        logger.debug(
            "MCP elicitation unavailable or failed (%s); falling back to awaiting_permission.", type(exc).__name__
        )
        return None
    action = getattr(outcome, "action", None)
    if action == "accept":
        data = getattr(outcome, "data", None)
        return bool(getattr(data, "grant_access", True))
    if action in ("decline", "cancel"):
        return False
    return None


def _progress_summary(turn: Dict[str, Any], *, poll_error: Optional[AriaAPIError] = None) -> Dict[str, Any]:
    messages = turn.get("messages")
    tool_calls = turn.get("tool_calls")
    state = str(turn.get("state", "unknown"))
    is_terminal = state in TERMINAL_STATES
    latest_response = _latest_assistant_response(turn)
    summary: Dict[str, Any] = {
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_call_record_count": (len(tool_calls) if isinstance(tool_calls, list) else 0),
    }
    elapsed_seconds = _turn_elapsed_seconds(turn, is_terminal=is_terminal)
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = elapsed_seconds
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
            latest_activity: Dict[str, Any] = {}
            if "type" in latest:
                latest_activity["type"] = latest["type"]
            resolved_name = _resolve_tool_name(tool_calls, latest)
            if resolved_name is not None:
                latest_activity["name"] = resolved_name
            for key in ("call_id", "is_error", "timestamp"):
                if key in latest:
                    latest_activity[key] = latest[key]
            if _tool_response_has_error(latest):
                latest_activity["recovered_error"] = True
            summary["latest_tool_activity"] = latest_activity

    if poll_error is not None and not is_terminal:
        interrupted: Dict[str, Any] = {
            "type": poll_error.error_type,
            "message": poll_error.message,
            "retryable": True,
        }
        if poll_error.status_code is not None:
            interrupted["status_code"] = poll_error.status_code
        if poll_error.retry_after_ms is not None:
            interrupted["retry_after_ms"] = poll_error.retry_after_ms
        summary["poll_interrupted"] = interrupted

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
        perm_info = _inspect_permission_requests(turn)
        if perm_info["has_requests"]:
            domain_label = ", ".join(perm_info["valid_domains"])
            detail = f" to {domain_label}" if domain_label else ""
            summary.update(
                phase="awaiting_permission",
                requires_permission=True,
                status_text=(
                    f"ARIA paused to request outbound network access{detail} "
                    "before finishing; saved workspace state is preserved."
                ),
            )
            if perm_info["valid_domains"]:
                summary["requested_network_domains"] = perm_info["valid_domains"]
            if perm_info["rejected_domains"]:
                summary["rejected_network_domains"] = perm_info["rejected_domains"]
        elif _has_agent_questions(turn):
            summary.update(
                phase="awaiting_user_input",
                requires_user_input=True,
                status_text="ARIA paused with follow-up question(s) in agent_questions.",
            )
        else:
            summary.update(phase="completed", status_text="ARIA completed the turn.")
    elif state == "errored":
        summary.update(phase="errored", status_text="ARIA stopped with an execution error.")
    elif state == "cancelled":
        summary.update(phase="cancelled", status_text="The ARIA turn was cancelled.")
    else:
        summary.update(phase=state, status_text=f"ARIA reported state '{state}'.")
    return summary


def _extract_user_prompt_text(turn: Dict[str, Any]) -> Optional[str]:
    """Extract a bounded, sanitized plain-text user prompt from a turn snapshot."""
    raw = turn.get("user_prompt")
    if isinstance(raw, str) and raw.strip():
        return sanitize_sensitive_text(raw.strip(), max_chars=MAX_USER_PROMPT_PREVIEW_CHARS)
    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw[:20]:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return sanitize_sensitive_text("\n\n".join(parts), max_chars=MAX_USER_PROMPT_PREVIEW_CHARS)
    return None


def _extract_tool_steps(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pair recent tool invocations and responses into compact, zero-output UI activity steps."""
    tool_calls = turn.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return []
    steps_by_id: Dict[str, Dict[str, Any]] = {}
    ordered_steps: List[Dict[str, Any]] = []
    for idx, record in enumerate(tool_calls[-MAX_PROGRESS_TOOL_RECORDS:]):
        if not isinstance(record, dict):
            continue
        raw_call_id = record.get("call_id")
        call_id = raw_call_id.strip() if isinstance(raw_call_id, str) and raw_call_id.strip() else f"step-{idx}"
        rec_type = record.get("type")
        resolved_name = _resolve_tool_name(tool_calls, record) or "tool"
        safe_name = sanitize_sensitive_text(resolved_name, max_chars=80)
        ts = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None

        existing = steps_by_id.get(call_id)
        if existing is None:
            existing = {
                "call_id": sanitize_sensitive_text(call_id, max_chars=80),
                "name": safe_name,
                "status": "running",
            }
            if ts:
                existing["started_at"] = ts
            steps_by_id[call_id] = existing
            ordered_steps.append(existing)
        elif existing.get("name") == "tool" and safe_name != "tool":
            existing["name"] = safe_name

        if rec_type == "response":
            existing["status"] = "recovered_error" if _tool_response_has_error(record) else "ok"
            if ts:
                existing["ended_at"] = ts
        elif ts and "started_at" not in existing:
            existing["started_at"] = ts

    return ordered_steps[-MAX_TOOL_STEPS:]


def _turn_result(
    turn: Dict[str, Any],
    *,
    include_turn: bool = False,
    poll_error: Optional[AriaAPIError] = None,
) -> Dict[str, Any]:
    state = str(turn["state"])
    turn_id = str(turn["id"])
    is_terminal = state in TERMINAL_STATES
    perm_info = _inspect_permission_requests(turn) if state == "completed" else None
    has_questions = _has_agent_questions(turn) if state == "completed" else False

    if state == "completed" and perm_info and perm_info["has_requests"]:
        valid_domains = perm_info["valid_domains"]
        if valid_domains:
            domains_json = json.dumps(valid_domains)
            next_action = (
                "ARIA paused on permission_requests without losing /workspace progress. "
                f"Ask the user whether to grant outbound access to {domains_json} (or if the host "
                "harness is running in user-authorized full-auto mode and these domains match the "
                "user's task, proceed directly), then call aria_send_message with "
                f"parent_turn_id='{turn_id}', allowed_network_domains={domains_json}, and a brief "
                "continuation message (or pass network_access_mode='auto_safe' for vetted research "
                "registries). To decline, call aria_send_message with parent_turn_id set to this "
                "turn_id instructing ARIA to finish without network access."
            )
        else:
            next_action = (
                "ARIA requested a permission that cannot be granted automatically (non-public or "
                "unsupported target). Inspect permission_requests and continue via aria_send_message "
                f"with parent_turn_id='{turn_id}' instructing ARIA how to proceed offline."
            )
    elif state == "completed" and has_questions:
        next_action = (
            "ARIA paused with follow-up question(s) in agent_questions. Confirm the answer with "
            f"the user (or answer from authorized task context) and call aria_send_message with "
            f"parent_turn_id='{turn_id}'."
        )
    elif state == "completed":
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
    elif poll_error is not None:
        next_action = (
            "A follow-up poll hit a transient service hiccup after reading turn progress; "
            "ARIA continues working server-side. Call aria_get_turn again with this turn_id "
            "and wait_seconds up to 30 (do not add separate sleep calls or start a duplicate thread)."
        )
    else:
        next_action = (
            "Call aria_get_turn again with this turn_id and wait_seconds up to 30 "
            "(which waits server-side; do not add separate sleep calls or start a duplicate thread)."
        )

    if is_terminal:
        poll_after_seconds: Optional[int] = None
    elif poll_error is not None and poll_error.retry_after_ms is not None:
        poll_after_seconds = max(2, min(MAX_WAIT_SECONDS, math.ceil(poll_error.retry_after_ms / 1_000)))
    elif poll_error is not None:
        poll_after_seconds = 5
    else:
        poll_after_seconds = 2

    result: Dict[str, Any] = {
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
        "created_at": turn.get("created_at"),
        "updated_at": turn.get("updated_at"),
        "poll_after_seconds": poll_after_seconds,
        "latest_response": _latest_assistant_response(turn),
        "error_info": turn.get("error_info"),
        "agent_questions": turn.get("agent_questions"),
        "permission_requests": turn.get("permission_requests"),
        "progress": _progress_summary(turn, poll_error=poll_error),
        "next_action": next_action,
    }
    active_domains = _turn_allowed_network_domains(turn)
    if active_domains:
        result["active_allowed_network_domains"] = active_domains
    user_prompt_text = _extract_user_prompt_text(turn)
    if user_prompt_text:
        result["user_prompt"] = user_prompt_text
    raw_title = turn.get("title")
    if isinstance(raw_title, str) and raw_title.strip():
        result["title"] = sanitize_sensitive_text(raw_title.strip(), max_chars=MAX_THREAD_TITLE_LENGTH)
    tool_steps = _extract_tool_steps(turn)
    if tool_steps:
        result["tool_steps"] = tool_steps
    bounded_thread_id = _bounded_identifier(turn.get("thread_id"))
    if bounded_thread_id:
        result["forge_url"] = public_forge_aria_url(bounded_thread_id)
    if perm_info and perm_info["has_requests"]:
        result["requires_permission"] = True
    elif has_questions:
        result["requires_user_input"] = True
    if include_turn:
        result["turn"] = turn
    return result


def _derive_fallback_title(message: str) -> str:
    collapsed = " ".join(message.split())
    if len(collapsed) <= MAX_FALLBACK_TITLE_CHARS:
        return collapsed
    return f"{collapsed[: MAX_FALLBACK_TITLE_CHARS - 3].rstrip()}..."


async def _apply_root_thread_title(
    client: httpx.AsyncClient,
    thread_id: Any,
    *,
    explicit_title: Optional[str],
    message: str,
) -> Dict[str, str]:
    if not isinstance(thread_id, str) or not thread_id or len(thread_id) > MAX_IDENTIFIER_LENGTH:
        return {"title_error": "missing_thread_id"}
    encoded_thread_id = quote(thread_id, safe="")
    if explicit_title is not None:
        try:
            await _request_json(
                client,
                "PATCH",
                f"/api/v1/threads/{encoded_thread_id}",
                payload={"title": explicit_title},
            )
        except AriaAPIError as exc:
            return {"title_error": exc.error_type}
        return {"title": explicit_title}

    try:
        _, generated = await _request_json(
            client,
            "POST",
            f"/api/v1/threads/{encoded_thread_id}/generate-title",
            payload=None,
        )
        if isinstance(generated, dict):
            generated_thread_id = generated.get("thread_id")
            generated_title = generated.get("title")
            if (
                (generated_thread_id is None or generated_thread_id == thread_id)
                and isinstance(generated_title, str)
                and generated_title.strip()
                and len(generated_title.strip()) <= MAX_THREAD_TITLE_LENGTH
            ):
                return {"title": generated_title.strip()}
    except AriaAPIError:
        pass

    fallback_title = _derive_fallback_title(message)
    try:
        await _request_json(
            client,
            "PATCH",
            f"/api/v1/threads/{encoded_thread_id}",
            payload={"title": fallback_title},
        )
    except AriaAPIError as exc:
        return {"title_error": exc.error_type}
    return {"title": fallback_title}


def _merge_domain_lists(existing: Sequence[str], added: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in (*existing, *added):
        if item not in seen:
            seen.add(item)
            out.append(item)
    if len(out) > MAX_ALLOWED_NETWORK_DOMAINS:
        raise AriaAPIError(
            "invalid_request",
            f"Combined allowed_network_domains exceeds {MAX_ALLOWED_NETWORK_DOMAINS} hostnames.",
            retryable=False,
        )
    return out


async def _find_existing_child_turn_id(
    client: httpx.AsyncClient,
    turn: Dict[str, Any],
) -> Optional[str]:
    """Return a child/newer turn_id if `turn` has already been continued in its thread."""
    thread_id = turn.get("thread_id")
    current_id = str(turn.get("id") or "")
    if not isinstance(thread_id, str) or not thread_id or not current_id:
        return None
    encoded_thread_id = quote(thread_id, safe="")
    try:
        _, turns = await _request_json(
            client,
            "GET",
            f"/api/v1/threads/{encoded_thread_id}/turns",
            params={
                "field_set": "metadata",
                "sort_by": "created_at",
                "sort_desc": "true",
                "limit": 5,
                "offset": 0,
            },
        )
    except AriaAPIError:
        return None
    if not isinstance(turns, list) or not turns:
        return None
    for candidate in turns:
        if not isinstance(candidate, dict):
            continue
        cand_id = _bounded_identifier(candidate.get("id"))
        if cand_id and candidate.get("parent_turn_id") == current_id:
            return cand_id
    newest = turns[0] if isinstance(turns[0], dict) else None
    newest_id = _bounded_identifier(newest.get("id")) if newest else None
    if newest_id and newest_id != current_id:
        return newest_id
    return None


async def _maybe_resolve_permission_requests(
    client: httpx.AsyncClient,
    turn: Dict[str, Any],
    *,
    mode: NetworkAccessMode,
    ctx: Any,
    wait_deadline: float,
) -> tuple[Dict[str, Any], Optional[AriaAPIError], Dict[str, Any]]:
    """Resolve terminal `network_access` `permission_requests` safely when policy/elicitation allows."""
    extra_meta: Dict[str, Any] = {}
    poll_error: Optional[AriaAPIError] = None
    auto_grants: List[Dict[str, Any]] = []
    origin_turn_id = str(turn.get("id") or "")

    for _ in range(MAX_AUTO_PERMISSION_GRANTS):
        if str(turn.get("state")) != "completed":
            break
        perm_info = _inspect_permission_requests(turn)
        if not perm_info["has_requests"] or not perm_info["all_network_access"]:
            break

        valid_domains = perm_info["valid_domains"]
        rejected_domains = perm_info["rejected_domains"]
        grant: Optional[bool] = None
        source = ""

        if mode == "deny":
            grant = False
            source = "policy_deny"
        elif rejected_domains or not valid_domains:
            break
        elif mode == "auto_safe":
            safe_domains = _configured_safe_network_domains()
            if all(d in safe_domains for d in valid_domains):
                grant = True
                source = "auto_safe"
            else:
                elicited = await _elicit_network_permission(
                    ctx,
                    domains=valid_domains,
                    reasons=perm_info["reasons"],
                )
                if elicited is None:
                    break
                grant = elicited
                source = "mcp_elicitation"
        elif mode == "auto_public":
            grant = True
            source = "auto_public"
        else:
            elicited = await _elicit_network_permission(
                ctx,
                domains=valid_domains,
                reasons=perm_info["reasons"],
            )
            if elicited is None:
                break
            grant = elicited
            source = "mcp_elicitation"

        if grant is True:
            merged_domains = _merge_domain_lists(_turn_allowed_network_domains(turn), valid_domains)
            cont_message = (
                f"Outbound network access granted for {', '.join(valid_domains)}. "
                "Continue from the saved /workspace state without repeating completed work."
            )
            cont_payload: Dict[str, Any] = {
                "parent_turn_id": str(turn["id"]),
                "user_prompt": [{"type": "text", "text": cont_message}, dict(ARIA_CLIENT_PROMPT_PART)],
                "permissions": {"allowed_network_domains": merged_domains},
            }
            logger.info(
                "Auto-continuing ARIA turn with %d network domain(s) via %s",
                len(valid_domains),
                source,
            )
            prev_turn_id = str(turn["id"])
            turn = await _request_turn(client, "POST", "/api/v1/turns", payload=cont_payload)
            auto_grants.append(
                {
                    "from_turn_id": prev_turn_id,
                    "to_turn_id": str(turn["id"]),
                    "granted_domains": valid_domains,
                    "active_allowed_network_domains": merged_domains,
                    "source": source,
                }
            )
            remaining_wait = max(0, min(MAX_WAIT_SECONDS, int(wait_deadline - time.monotonic())))
            turn, poll_error = await _poll_turn_with_status(
                client,
                turn,
                remaining_wait,
                confirmed_get=False,
            )
            if poll_error is not None:
                break
            continue

        # grant is False
        if source == "policy_deny":
            deny_targets = valid_domains or rejected_domains
            cont_message = (
                "Outbound network access is denied by caller policy. "
                "Complete the analysis offline using existing /workspace files and W&B APIs without repeating completed work."
            )
            cont_payload = {
                "parent_turn_id": str(turn["id"]),
                "user_prompt": [{"type": "text", "text": cont_message}, dict(ARIA_CLIENT_PROMPT_PART)],
            }
            prev_turn_id = str(turn["id"])
            turn = await _request_turn(client, "POST", "/api/v1/turns", payload=cont_payload)
            extra_meta["permission_resolution"] = {
                "action": "denied_and_continued_offline",
                "from_turn_id": prev_turn_id,
                "to_turn_id": str(turn["id"]),
                "declined_domains": deny_targets,
                "source": source,
            }
            remaining_wait = max(0, min(MAX_WAIT_SECONDS, int(wait_deadline - time.monotonic())))
            turn, poll_error = await _poll_turn_with_status(
                client,
                turn,
                remaining_wait,
                confirmed_get=False,
            )
            break

        extra_meta["permission_resolution"] = {
            "action": "declined_by_user",
            "declined_domains": valid_domains,
            "source": source,
        }
        break

    if auto_grants:
        extra_meta["resumed_from_turn_id"] = origin_turn_id
        extra_meta["permission_resolution"] = {
            "action": "granted_and_continued",
            "grants": auto_grants,
        }
    elif "resumed_from_turn_id" not in extra_meta and str(turn.get("id") or "") != origin_turn_id:
        extra_meta["resumed_from_turn_id"] = origin_turn_id

    return turn, poll_error, extra_meta


async def send_aria_message(
    message: str,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    parent_turn_id: Optional[str] = None,
    wait_seconds: int = 0,
    include_turn: bool = False,
    title: Optional[str] = None,
    allowed_network_domains: Optional[Sequence[str]] = None,
    network_access_mode: Optional[NetworkAccessMode] = None,
    *,
    ctx: Any = None,
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
        normalized_domains = normalize_allowed_network_domains(allowed_network_domains)
        resolved_access_mode, _ = _resolve_network_access_mode(network_access_mode)

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
            if title is not None:
                raise AriaAPIError(
                    "invalid_request",
                    "title must be omitted when parent_turn_id is set.",
                    retryable=False,
                )
        elif title is not None:
            title = _normalize_bounded_string(
                title,
                name="title",
                max_length=MAX_THREAD_TITLE_LENGTH,
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

        effective_domains = normalized_domains
        if parent_turn_id is not None and normalized_domains is not None:
            try:
                parent_snapshot = await _fetch_turn(client, parent_turn_id)
                effective_domains = _merge_domain_lists(
                    _turn_allowed_network_domains(parent_snapshot),
                    normalized_domains,
                )
            except AriaAPIError as exc:
                if not exc.retryable:
                    raise

        payload: Dict[str, Any] = {
            "user_prompt": [{"type": "text", "text": message}, dict(ARIA_CLIENT_PROMPT_PART)],
        }
        if parent_turn_id is not None:
            payload["parent_turn_id"] = parent_turn_id
        else:
            if entity is not None:
                payload["entity"] = entity
            if project is not None:
                payload["project"] = project
        if effective_domains is not None:
            payload["permissions"] = {"allowed_network_domains": effective_domains}

        wait_deadline = time.monotonic() + wait_seconds
        logger.info(
            "Submitting ARIA turn: continuation=%s scoped=%s wait=%ss",
            parent_turn_id is not None,
            entity is not None or project is not None,
            wait_seconds,
        )
        turn = await _request_turn(client, "POST", "/api/v1/turns", payload=payload)
        title_fields: Dict[str, str] = {}
        if parent_turn_id is None:
            title_fields = await _apply_root_thread_title(
                client,
                turn.get("thread_id"),
                explicit_title=title,
                message=message,
            )
        resolution_meta: Dict[str, Any] = {}
        try:
            turn, poll_error = await _poll_turn_with_status(
                client,
                turn,
                wait_seconds,
                confirmed_get=False,
            )
            if poll_error is None and str(turn.get("state")) == "completed":
                turn, poll_error, resolution_meta = await _maybe_resolve_permission_requests(
                    client,
                    turn,
                    mode=resolved_access_mode,
                    ctx=ctx,
                    wait_deadline=wait_deadline,
                )
        except AriaAPIError as exc:
            # The non-idempotent POST was confirmed and returned a durable
            # handle. A later GET/continuation failure must never erase that
            # handle or imply that the submission outcome is unknown.
            result = exc.as_result(turn_id=str(turn["id"]), operation="get")
            result.update(title_fields)
            return _finalize_result(
                result,
                extra_secrets=(resolved_api_key,),
            )
        result = _turn_result(turn, include_turn=include_turn, poll_error=poll_error)
        result.update(title_fields)
        result.update(resolution_meta)
        return _finalize_result(
            result,
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


def _normalize_parallel_turn_ids(
    primary_turn_id: str,
    parallel_turn_ids: Optional[Sequence[str]],
) -> List[str]:
    if parallel_turn_ids is None:
        return []
    if isinstance(parallel_turn_ids, (str, bytes)) or not isinstance(parallel_turn_ids, Sequence):
        raise AriaAPIError(
            "invalid_request",
            f"parallel_turn_ids must be a list of up to {MAX_PARALLEL_TURN_IDS} turn_id strings.",
            retryable=False,
        )
    if len(parallel_turn_ids) > MAX_PARALLEL_TURN_IDS:
        raise AriaAPIError(
            "invalid_request",
            f"parallel_turn_ids may contain at most {MAX_PARALLEL_TURN_IDS} additional turn_ids.",
            retryable=False,
        )
    deduped: List[str] = []
    seen: set[str] = {primary_turn_id}
    for item in parallel_turn_ids:
        normalized = _normalize_bounded_string(
            item,
            name="parallel_turn_ids item",
            max_length=MAX_IDENTIFIER_LENGTH,
        )
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


async def _poll_and_build_turn_entry(
    client: httpx.AsyncClient,
    target_turn_id: str,
    *,
    wait_seconds: int,
    wait_deadline: float,
    include_turn: bool,
    resolved_access_mode: NetworkAccessMode,
    mode_explicit: bool,
    ctx: Any,
    allow_writes: bool,
) -> Dict[str, Any]:
    active_id = target_turn_id
    turn = await _fetch_turn(client, target_turn_id)
    was_terminal_before_poll = str(turn.get("state")) in TERMINAL_STATES
    remaining_wait = max(0, min(wait_seconds, math.ceil(wait_deadline - time.monotonic())))
    turn, poll_error = await _poll_turn_with_status(
        client,
        turn,
        remaining_wait,
        confirmed_get=True,
    )
    active_id = str(turn.get("id") or target_turn_id)
    resolution_meta: Dict[str, Any] = {}
    can_auto_resolve = allow_writes and (mode_explicit or _client_supports_elicitation(ctx))
    if can_auto_resolve and poll_error is None and str(turn.get("state")) == "completed":
        perm_info = _inspect_permission_requests(turn)
        if perm_info["has_requests"] and perm_info["all_network_access"]:
            existing_child = await _find_existing_child_turn_id(client, turn) if was_terminal_before_poll else None
            if existing_child:
                resolution_meta["continued_in_turn_id"] = existing_child
            else:
                try:
                    turn, poll_error, resolution_meta = await _maybe_resolve_permission_requests(
                        client,
                        turn,
                        mode=resolved_access_mode,
                        ctx=ctx,
                        wait_deadline=wait_deadline,
                    )
                    active_id = str(turn.get("id") or active_id)
                except AriaAPIError as exc:
                    return exc.as_result(turn_id=active_id, operation="get")

    entry = _turn_result(turn, include_turn=include_turn, poll_error=poll_error)
    entry.update(resolution_meta)
    return entry


def _summarize_parallel_turns(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    completed_count = 0
    active_count = 0
    awaiting_permission_count = 0
    awaiting_user_input_count = 0
    errored_count = 0
    cancelled_count = 0
    for item in items:
        if item.get("ok") is False or item.get("state") == "errored":
            errored_count += 1
        elif item.get("state") == "cancelled":
            cancelled_count += 1
        elif not item.get("is_terminal", False):
            active_count += 1
        elif item.get("requires_permission"):
            awaiting_permission_count += 1
        elif item.get("requires_user_input"):
            awaiting_user_input_count += 1
        else:
            completed_count += 1

    return {
        "total_count": len(items),
        "completed_count": completed_count,
        "active_count": active_count,
        "awaiting_permission_count": awaiting_permission_count,
        "awaiting_user_input_count": awaiting_user_input_count,
        "errored_count": errored_count,
        "cancelled_count": cancelled_count,
        "all_terminal": active_count == 0,
    }


async def get_aria_turn(
    turn_id: str,
    wait_seconds: int = 0,
    include_turn: bool = False,
    network_access_mode: Optional[NetworkAccessMode] = None,
    parallel_turn_ids: Optional[Sequence[str]] = None,
    *,
    ctx: Any = None,
    allow_writes: bool = False,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Fetch one ARIA turn (or concurrently poll parallel sub-agent turns) within a bounded deadline."""
    owns_client = client is None
    resolved_api_key: Optional[str] = None
    active_turn_id: Optional[str] = None
    try:
        turn_id = _normalize_bounded_string(
            turn_id,
            name="turn_id",
            max_length=MAX_IDENTIFIER_LENGTH,
        )
        active_turn_id = turn_id
        _validate_wait_seconds(wait_seconds)
        extra_turn_ids = _normalize_parallel_turn_ids(turn_id, parallel_turn_ids)
        has_parallel_request = parallel_turn_ids is not None and len(parallel_turn_ids) > 0
        resolved_access_mode, mode_explicit = _resolve_network_access_mode(network_access_mode)
        resolved_api_key = _resolve_api_key(api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        wait_deadline = time.monotonic() + wait_seconds
        if not has_parallel_request:
            single_entry = await _poll_and_build_turn_entry(
                client,
                turn_id,
                wait_seconds=wait_seconds,
                wait_deadline=wait_deadline,
                include_turn=include_turn,
                resolved_access_mode=resolved_access_mode,
                mode_explicit=mode_explicit,
                ctx=ctx,
                allow_writes=allow_writes,
            )
            return _finalize_result(
                single_entry,
                extra_secrets=(resolved_api_key,),
            )

        all_turn_ids = [turn_id, *extra_turn_ids]

        async def _poll_member(tid: str) -> Dict[str, Any]:
            try:
                return await _poll_and_build_turn_entry(
                    client,
                    tid,
                    wait_seconds=wait_seconds,
                    wait_deadline=wait_deadline,
                    include_turn=include_turn and len(all_turn_ids) <= 2,
                    resolved_access_mode=resolved_access_mode,
                    mode_explicit=mode_explicit,
                    ctx=None if len(all_turn_ids) > 1 else ctx,
                    allow_writes=allow_writes,
                )
            except AriaAPIError as exc:
                err_entry = exc.as_result(turn_id=tid, operation="get")
                err_entry.setdefault("is_terminal", not exc.retryable)
                err_entry.setdefault("state", "errored" if not exc.retryable else "in_progress")
                return err_entry

        polled_entries = list(await asyncio.gather(*(_poll_member(tid) for tid in all_turn_ids)))
        primary_entry = polled_entries[0]
        if primary_entry.get("ok") is False and all(item.get("ok") is False for item in polled_entries):
            return _finalize_result(
                primary_entry,
                extra_secrets=(resolved_api_key,),
            )

        compact_members: List[Dict[str, Any]] = []
        for entry in polled_entries:
            member = dict(entry)
            if len(all_turn_ids) > 2 and isinstance(member.get("latest_response"), str):
                resp_text = member["latest_response"]
                if len(resp_text) > MAX_PARALLEL_MEMBER_RESPONSE_CHARS:
                    member["latest_response"] = (
                        f"{resp_text[:MAX_PARALLEL_MEMBER_RESPONSE_CHARS].rstrip()}\n\n...[truncated in parallel batch; poll individual turn_id for full text]"
                    )
            compact_members.append(member)

        summary = _summarize_parallel_turns(compact_members)
        result = dict(primary_entry)
        result["ok"] = True
        result["parallel_turns"] = compact_members
        result["parallel_summary"] = summary

        if not summary["all_terminal"]:
            sibling_waits = [
                int(item["poll_after_seconds"])
                for item in compact_members
                if isinstance(item.get("poll_after_seconds"), int) and item["poll_after_seconds"] > 0
            ]
            result["poll_after_seconds"] = min(sibling_waits) if sibling_waits else 2
            active_ids = [
                str(item.get("turn_id"))
                for item in compact_members
                if not item.get("is_terminal", False) and isinstance(item.get("turn_id"), str)
            ]
            result["next_action"] = (
                f"{summary['active_count']} of {summary['total_count']} parallel ARIA turns are still running "
                f"({json.dumps(active_ids)}). Call aria_get_turn again with turn_id={json.dumps(active_ids[0])}, "
                f"parallel_turn_ids={json.dumps(active_ids[1:])}, and wait_seconds up to 30."
            )
        elif summary["awaiting_permission_count"] > 0:
            paused_ids = [
                str(item.get("turn_id"))
                for item in compact_members
                if item.get("requires_permission") and isinstance(item.get("turn_id"), str)
            ]
            result["next_action"] = (
                f"{summary['awaiting_permission_count']} parallel ARIA turn(s) paused on permission_requests "
                f"({json.dumps(paused_ids)}). Inspect parallel_turns and continue each via aria_send_message "
                "with parent_turn_id and allowed_network_domains (or decline offline)."
            )

        return _finalize_result(
            result,
            extra_secrets=(resolved_api_key,),
        )
    except AriaAPIError as exc:
        logger.warning("ARIA get failed: %s", exc.error_type)
        safe_turn_id = (
            active_turn_id if isinstance(active_turn_id, str) and len(active_turn_id) <= MAX_IDENTIFIER_LENGTH else None
        )
        return _finalize_result(
            exc.as_result(turn_id=safe_turn_id, operation="get"),
            extra_secrets=(resolved_api_key,) if resolved_api_key else (),
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()


def _validate_page(limit: int, offset: int) -> None:
    for name, value, low, high in (
        ("limit", limit, 1, MAX_LIST_LIMIT),
        ("offset", offset, 0, MAX_LIST_OFFSET),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise AriaAPIError(
                "invalid_request",
                f"{name} must be an integer from {low} to {high}.",
                retryable=False,
            )


def _bounded_identifier(value: Any) -> Optional[str]:
    if isinstance(value, str) and value and len(value) <= MAX_IDENTIFIER_LENGTH:
        return value
    return None


def _invalid_list_response(kind: str) -> AriaAPIError:
    return AriaAPIError(
        "invalid_response",
        f"ARIA returned an invalid {kind} list.",
        retryable=True,
    )


def _thread_summary(thread: Any) -> Dict[str, Any]:
    if not isinstance(thread, dict) or _bounded_identifier(thread.get("id")) is None:
        raise _invalid_list_response("thread")
    return {
        "thread_id": thread["id"],
        "title": thread.get("title") or None,
        "prompt_preview": _bounded_preview(thread.get("prompt_preview")),
        "scope": {"entity": thread.get("wandb_entity"), "project": thread.get("wandb_project")},
        "is_pinned": thread.get("is_pinned"),
        "visibility": thread.get("visibility"),
        "created_at": thread.get("created_at"),
        "last_turn_created_at": thread.get("last_turn_created_at"),
        "forge_url": public_forge_aria_url(thread["id"]),
    }


def _bounded_preview(text: Any) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    if len(text) > MAX_PROMPT_PREVIEW_CHARS:
        return text[:MAX_PROMPT_PREVIEW_CHARS] + "..."
    return text


def _thread_turn_summary(turn: Any) -> Dict[str, Any]:
    if (
        not isinstance(turn, dict)
        or _bounded_identifier(turn.get("id")) is None
        or turn.get("state") not in TURN_STATES
    ):
        raise _invalid_list_response("turn")
    return {
        "turn_id": turn["id"],
        "parent_turn_id": _bounded_identifier(turn.get("parent_turn_id")),
        "state": turn["state"],
        "is_terminal": turn["state"] in TERMINAL_STATES,
        "created_at": turn.get("created_at"),
        "updated_at": turn.get("updated_at"),
    }


def _page_fields(count: int, limit: int, offset: int) -> Dict[str, Any]:
    has_more = count == limit
    return {
        "count": count,
        "offset": offset,
        "has_more": has_more,
        "next_offset": offset + count if has_more else None,
    }


async def list_aria_threads(
    entity: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
    *,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """List the caller's ARIA threads, most recently active first."""
    owns_client = client is None
    resolved_api_key: Optional[str] = None
    try:
        _validate_page(limit, offset)
        params: Dict[str, Any] = {"limit": limit, "offset": offset, "sort_desc": "true"}
        for name, value in (("entity", entity), ("project", project)):
            if value is not None:
                params[name] = _normalize_bounded_string(value, name=name, max_length=MAX_SCOPE_LENGTH)
        resolved_api_key = _resolve_api_key(api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        _, threads = await _request_json(client, "GET", "/api/v1/threads", params=params)
        if not isinstance(threads, list) or len(threads) > limit:
            raise _invalid_list_response("thread")
        summaries = [_thread_summary(thread) for thread in threads]
        return _finalize_result(
            {
                "ok": True,
                "threads": summaries,
                **_page_fields(len(summaries), limit, offset),
                "next_action": (
                    "Call aria_get_thread with a thread_id to see its turns."
                    if summaries
                    else "No ARIA threads matched; start one with aria_send_message."
                ),
            },
            extra_secrets=(resolved_api_key,),
        )
    except AriaAPIError as exc:
        logger.warning("ARIA thread list failed: %s", exc.error_type)
        return _finalize_result(
            exc.as_result(operation="get"),
            extra_secrets=(resolved_api_key,) if resolved_api_key else (),
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def get_aria_thread(
    thread_id: str,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
    *,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Fetch one ARIA thread and a compact, newest-first page of its turns."""
    owns_client = client is None
    resolved_api_key: Optional[str] = None
    try:
        thread_id = _normalize_bounded_string(thread_id, name="thread_id", max_length=MAX_IDENTIFIER_LENGTH)
        _validate_page(limit, offset)
        resolved_api_key = _resolve_api_key(api_key)
        if client is None:
            client = _new_http_client(resolved_api_key)

        encoded_thread_id = quote(thread_id, safe="")
        _, thread = await _request_json(client, "GET", f"/api/v1/threads/{encoded_thread_id}")
        if not isinstance(thread, dict) or thread.get("id") != thread_id:
            raise AriaAPIError(
                "invalid_response",
                "ARIA returned a different thread than the one requested.",
                retryable=False,
            )
        _, turns = await _request_json(
            client,
            "GET",
            f"/api/v1/threads/{encoded_thread_id}/turns",
            params={
                "field_set": "metadata",
                "sort_by": "created_at",
                "sort_desc": "true",
                "limit": limit,
                "offset": offset,
            },
        )
        if not isinstance(turns, list) or len(turns) > limit:
            raise _invalid_list_response("turn")
        summaries = [_thread_turn_summary(turn) for turn in turns]
        newest_completed = next((turn["turn_id"] for turn in summaries if turn["state"] == "completed"), None)
        if newest_completed is not None:
            next_action = (
                "Call aria_get_turn with a turn_id to read its answer. To continue this "
                f"conversation, call aria_send_message with parent_turn_id={newest_completed}."
            )
        else:
            next_action = "Call aria_get_turn with a turn_id to read its state or answer."
        return _finalize_result(
            {
                "ok": True,
                "thread": _thread_summary(thread),
                "turns": summaries,
                "newest_completed_turn_id": newest_completed,
                **_page_fields(len(summaries), limit, offset),
                "next_action": next_action,
            },
            extra_secrets=(resolved_api_key,),
        )
    except AriaAPIError as exc:
        if exc.error_type == "turn_not_found":
            exc = AriaAPIError(
                "thread_not_found",
                "ARIA could not find a thread visible to the current W&B user.",
                retryable=False,
                status_code=exc.status_code,
            )
        logger.warning("ARIA thread get failed: %s", exc.error_type)
        return _finalize_result(
            exc.as_result(operation="get"),
            extra_secrets=(resolved_api_key,) if resolved_api_key else (),
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()
