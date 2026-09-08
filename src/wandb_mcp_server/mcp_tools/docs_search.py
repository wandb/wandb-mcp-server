"""Bounded proxy search against the official W&B docs MCP."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import time

import httpx

from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.config import MAX_ACCUMULATED_BYTES, MAX_RESPONSE_TOKENS, MCP_TOOL_TIMEOUT_SECONDS
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
from wandb_mcp_server.instrumented_server import run_sync_in_current_tool
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.trace_utils import count_tokens_conservative

DOCS_MCP_URL = "https://docs.wandb.ai/mcp"
DOCS_SEARCH_TIMEOUT = 30
MAX_DOCS_QUERY_BYTES = 64 * 1024
MAX_DOCS_RESPONSE_BYTES = 4 * 1024 * 1024
_TRUNCATION_NOTICE = "\n\n[Documentation truncated to the response limit.]"

SEARCH_WANDB_DOCS_TOOL_DESCRIPTION = """Search official W&B documentation for API usage, code examples, and guides.

<when_to_use>
Call when you need to know HOW to use a W&B/Weave feature or API. Searches docs.wandb.ai.
</when_to_use>

Parameters
----------
query : str
    Natural language search query about W&B documentation.

Returns
-------
str
    Relevant documentation snippets matching the query, bounded by the response token limit.
"""


def is_docs_proxy_enabled() -> bool:
    """Check if the docs proxy is enabled via environment variable."""
    return os.environ.get("WANDB_MCP_PROXY_DOCS", "true").lower() != "false"


class _DocsError(Exception):
    def __init__(self, code: str, message: str, *, retry_after_ms: int | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after_ms = retry_after_ms


def _fit_text(text: str, deadline: float) -> str:
    """Choose a Unicode-safe prefix; the notice is included in the exact budget."""
    if time.monotonic() >= deadline:
        raise TimeoutError
    text = sanitize_sensitive_text(text)
    token_count = count_tokens_conservative(text)
    if time.monotonic() >= deadline:
        raise TimeoutError
    if token_count <= MAX_RESPONSE_TOKENS:
        return text
    notice = sanitize_sensitive_text(_TRUNCATION_NOTICE)
    if count_tokens_conservative(notice) > MAX_RESPONSE_TOKENS:
        return "."
    low, high = 0, len(text)
    while low < high:
        if time.monotonic() >= deadline:
            raise TimeoutError
        midpoint = (low + high + 1) // 2
        # A prefix may cut through a redaction marker. Sanitize that exact
        # candidate before counting so the outer boundary cannot expand it.
        candidate = sanitize_sensitive_text(text[:midpoint] + notice)
        if count_tokens_conservative(candidate) <= MAX_RESPONSE_TOKENS:
            low = midpoint
        else:
            high = midpoint - 1
    if time.monotonic() >= deadline:
        raise TimeoutError
    return sanitize_sensitive_text(text[:low] + notice)


def _decode_response(raw: bytes, content_type: str, deadline: float) -> str:
    try:
        body = raw.decode("utf-8")
        if "text/event-stream" in content_type or any(line.startswith("data:") for line in body.splitlines()):
            result = None
            for event in body.replace("\r\n", "\n").split("\n\n"):
                data = "\n".join(line[5:].lstrip(" ") for line in event.splitlines() if line.startswith("data:"))
                if data:
                    result = json.loads(data)
                    if isinstance(result, dict) and ("result" in result or "error" in result):
                        break
        else:
            result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError
        if "error" in result:
            raise _DocsError("upstream_error", "Documentation search failed upstream.")
        payload = result.get("result")
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("isError"):
            raise _DocsError("upstream_error", "Documentation search failed upstream.")
        content = payload.get("content")
        if not isinstance(content, list):
            raise ValueError
        texts = []
        for item in content:
            if not isinstance(item, dict) or ("text" in item and not isinstance(item["text"], str)):
                raise ValueError
            if item.get("text"):
                texts.append(item["text"])
        if not texts:
            raise _DocsError("upstream_error", "No documentation results were returned.")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise _DocsError("malformed_response", "Documentation search returned a malformed response.") from exc
    return _fit_text("\n\n---\n\n".join(texts), deadline)


def _retry_after_ms(value: str | None) -> int:
    try:
        return max(1000, min(60000, int(float(value or "1") * 1000)))
    except (ValueError, OverflowError):
        return 1000


async def _request_docs(query: str, deadline: float) -> str:
    byte_limit = min(MAX_DOCS_RESPONSE_BYTES, MAX_ACCUMULATED_BYTES)
    async with httpx.AsyncClient(follow_redirects=False, timeout=max(0.001, deadline - time.monotonic())) as client:
        async with client.stream(
            "POST",
            DOCS_MCP_URL,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "search_weights_biases_documentation", "arguments": {"query": query}},
                "id": 1,
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Accept-Encoding": "identity",
            },
        ) as response:
            if response.status_code in (429, 503):
                raise _DocsError(
                    "server_busy",
                    "Documentation search is temporarily busy.",
                    retry_after_ms=_retry_after_ms(response.headers.get("Retry-After")),
                )
            if not 200 <= response.status_code < 300:
                raise _DocsError("upstream_error", "Documentation search returned an HTTP error.")
            if response.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
                raise _DocsError("malformed_response", "Documentation search returned unsupported encoding.")
            length = response.headers.get("Content-Length")
            expected_length = None
            if length is not None:
                if not length.isascii() or not length.isdecimal():
                    raise _DocsError("malformed_response", "Documentation search returned an invalid length.")
                length = length.lstrip("0") or "0"
                if len(length) > len(str(byte_limit)):
                    raise _DocsError("response_too_large", "Documentation search exceeded its download limit.")
                expected_length = int(length)
                if expected_length > byte_limit:
                    raise _DocsError("response_too_large", "Documentation search exceeded its download limit.")
            raw = bytearray()
            async for chunk in response.aiter_raw():
                if len(raw) + len(chunk) > byte_limit:
                    raise _DocsError("response_too_large", "Documentation search exceeded its download limit.")
                raw.extend(chunk)
            if expected_length is not None and len(raw) != expected_length:
                raise _DocsError("malformed_response", "Documentation search returned an inconsistent length.")
            # Blocking JSON/token processing shares the tool's bounded executor
            # and retains its physical admission permit after cancellation.
            return await run_sync_in_current_tool(
                functools.partial(_decode_response, bytes(raw), response.headers.get("Content-Type", ""), deadline)
            )


async def search_wandb_docs(query: str) -> str:
    """Search docs with one bounded request; never echo queries or upstream errors."""
    with track_tool_execution("search_wandb_docs", "n/a", {}) as ctx:
        try:
            if not isinstance(query, str) or not query.strip():
                raise _DocsError("invalid_input", "query must be non-empty text of at most 64 KiB.")
            try:
                query_bytes = len(query.encode("utf-8"))
            except UnicodeError as exc:
                raise _DocsError("invalid_input", "query must contain valid UTF-8 text.") from exc
            if query_bytes > MAX_DOCS_QUERY_BYTES:
                raise _DocsError("invalid_input", "query must be non-empty text of at most 64 KiB.")
            deadline = time.monotonic() + min(DOCS_SEARCH_TIMEOUT, MCP_TOOL_TIMEOUT_SECONDS)
            outer_deadline = current_tool_deadline.get()
            if outer_deadline is not None:
                deadline = min(deadline, outer_deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                return await _request_docs(query, deadline)
        except _DocsError as exc:
            ctx.mark_error(exc.code)
            payload = {"error": exc.code, "message": str(exc), "retryable": exc.retry_after_ms is not None}
            if exc.retry_after_ms is not None:
                payload["retry_after_ms"] = exc.retry_after_ms
            return json.dumps(payload)
        except (TimeoutError, httpx.TimeoutException):
            ctx.mark_error("tool_timeout")
            return json.dumps(
                {"error": "tool_timeout", "message": "Documentation search timed out.", "retryable": True}
            )
        except Exception:
            ctx.mark_error("upstream_error")
            return json.dumps(
                {"error": "upstream_error", "message": "Documentation search failed.", "retryable": False}
            )
