"""FastMCP instrumentation at the public protocol boundary."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.harness import (
    HarnessContext,
    context_from_sdk_request,
    current_harness_context,
)
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


def _error_from_mapping(value: dict[str, Any]) -> str | None:
    error = value.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        kind = error.get("code") or error.get("type") or "ToolError"
        message = error.get("message") or error.get("detail") or str(error)
        return f"{kind}: {str(message)[:500]}"
    message = value.get("message") or error
    return f"ToolError: {str(message)[:500]}"


def structured_result_error(result: Any) -> str | None:
    """Return a bounded error for a tool result with a top-level error marker."""
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        return "ToolError: MCP result marked as an error"
    if isinstance(result, dict):
        error = _error_from_mapping(result)
        if error:
            return error
        if "result" in result:
            return structured_result_error(result["result"])
        return None
    if isinstance(result, str):
        try:
            decoded = json.loads(result)
        except (TypeError, ValueError):
            return None
        return _error_from_mapping(decoded) if isinstance(decoded, dict) else None
    content = getattr(result, "content", None)
    if content is not None and content is not result:
        error = structured_result_error(content)
        if error:
            return error
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(result, "structured_content", None)
    if structured_content is not None and structured_content is not result:
        error = structured_result_error(structured_content)
        if error:
            return error
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        for block in result:
            if getattr(block, "isError", False) or getattr(block, "is_error", False):
                return "ToolError: MCP result marked as an error"
            text = getattr(block, "text", None)
            if isinstance(text, str):
                error = structured_result_error(text)
                if error:
                    return error
            error = structured_result_error(block)
            if error:
                return error
    return None


class InstrumentedFastMCP(FastMCP):
    """FastMCP server that emits exactly one analytics event per public tool call."""

    def _tool_harness_context(self) -> HarnessContext:
        existing = current_harness_context.get()
        if existing is not None and existing.agent_harness != "unknown":
            return existing.with_call_type("tools/call")
        try:
            sdk_context = context_from_sdk_request(
                self.get_context().request_context,
                call_type="tools/call",
            )
            if sdk_context.agent_harness != "unknown":
                return sdk_context
        except Exception:
            pass
        if existing is not None:
            return existing.with_call_type("tools/call")
        return HarnessContext().with_call_type("tools/call")

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        harness_token = current_harness_context.set(self._tool_harness_context())
        started = time.monotonic()
        success = True
        error: str | None = None
        try:
            try:
                result = await super().call_tool(name, arguments)
            except BaseException as exc:
                success = False
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
                raise
            structured_error = structured_result_error(result)
            if structured_error:
                success = False
                error = structured_error
            return result
        finally:
            duration_ms = round((time.monotonic() - started) * 1000, 2)
            try:
                from wandb_mcp_server.analytics import get_analytics_tracker
                from wandb_mcp_server.session_manager import current_session_id

                get_analytics_tracker().track_tool_call(
                    tool_name=name,
                    mcp_tool_name=name,
                    session_id=current_session_id.get(),
                    viewer_info=None,
                    params=arguments,
                    success=success,
                    error=error,
                    duration_ms=duration_ms,
                )
            except Exception as analytics_error:
                logger.debug("Tool analytics failed for %s: %s", name, analytics_error)
            finally:
                current_harness_context.reset(harness_token)


__all__ = ["InstrumentedFastMCP", "structured_result_error"]
