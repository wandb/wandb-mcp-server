"""FastMCP instrumentation at the public protocol boundary."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
from collections.abc import Sequence
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from wandb_mcp_server.admission import (
    AdmissionRejected,
    WeightedAdmissionController,
    current_tool_deadline,
    tool_cost,
)
from wandb_mcp_server.config import (
    MCP_ADMISSION_ACTOR_CAPACITY,
    MCP_ADMISSION_CONTROL_ENABLED,
    MCP_ADMISSION_PROCESS_CAPACITY,
    MCP_ADMISSION_WAIT_MS,
    MCP_HOSTED_MODE,
    MCP_TOOL_TIMEOUT_SECONDS,
)
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
    """FastMCP server with bounded dispatch and one event per public tool call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._admission_controller = (
            WeightedAdmissionController(
                actor_capacity=MCP_ADMISSION_ACTOR_CAPACITY,
                process_capacity=MCP_ADMISSION_PROCESS_CAPACITY,
                wait_timeout_seconds=MCP_ADMISSION_WAIT_MS / 1000,
            )
            if MCP_ADMISSION_CONTROL_ENABLED
            else None
        )

    def add_tool(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Run synchronous tools off-loop so admission limits real concurrency."""
        registered_fn = fn
        if not inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _threaded_tool(*fn_args: Any, **fn_kwargs: Any) -> Any:
                call = functools.partial(fn, *fn_args, **fn_kwargs)
                return await anyio.to_thread.run_sync(call, abandon_on_cancel=False)

            registered_fn = _threaded_tool
        super().add_tool(registered_fn, *args, **kwargs)

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
        lease = None
        deadline_token = None
        cost_class, weight = tool_cost(name, arguments)
        admission_outcome = "disabled"
        queue_ms = 0.0
        try:
            if self._admission_controller is not None:
                admission_started = time.monotonic()
                try:
                    lease = await self._admission_controller.acquire(
                        actor_id=self._analytics_actor_id(),
                        weight=weight,
                    )
                except AdmissionRejected as exc:
                    queue_ms = round((time.monotonic() - admission_started) * 1000, 2)
                    admission_outcome = "rejected"
                    success = False
                    error = "server_busy: tool admission wait budget exceeded"
                    raise ToolError(
                        json.dumps(
                            {
                                "error": "server_busy",
                                "message": "The MCP server is busy; retry this tool call.",
                                "retryable": True,
                                "retry_after_ms": 1000,
                            }
                        )
                    ) from exc
                queue_ms = lease.queue_ms
                admission_outcome = "admitted"

            enforce_deadline = MCP_HOSTED_MODE or self._admission_controller is not None
            if enforce_deadline:
                deadline_token = current_tool_deadline.set(time.monotonic() + MCP_TOOL_TIMEOUT_SECONDS)
            try:
                if enforce_deadline:
                    async with asyncio.timeout(MCP_TOOL_TIMEOUT_SECONDS):
                        result = await super().call_tool(name, arguments)
                else:
                    result = await super().call_tool(name, arguments)
            except TimeoutError as exc:
                success = False
                error = "tool_timeout: MCP tool execution deadline exceeded"
                raise ToolError(
                    json.dumps(
                        {
                            "error": "tool_timeout",
                            "message": "The MCP tool exceeded its execution deadline.",
                            "retryable": True,
                            "retry_after_ms": 1000,
                        }
                    )
                ) from exc
            except BaseException as exc:
                if success:
                    success = False
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
                raise
            structured_error = structured_result_error(result)
            if structured_error:
                success = False
                error = structured_error
            return result
        finally:
            if deadline_token is not None:
                current_tool_deadline.reset(deadline_token)
            if lease is not None:
                try:
                    await lease.release()
                except Exception as release_error:
                    logger.error("Tool admission release failed for %s: %s", name, release_error)
            duration_ms = round((time.monotonic() - started) * 1000, 2)
            try:
                from wandb_mcp_server.analytics import get_analytics_tracker
                from wandb_mcp_server.session_manager import current_session_id

                get_analytics_tracker().track_tool_call(
                    tool_name=name,
                    mcp_tool_name=name,
                    session_id=current_session_id.get(),
                    viewer_info=None,
                    params={
                        **arguments,
                        "cost_class": cost_class,
                        "admission_outcome": admission_outcome,
                        "queue_ms": queue_ms,
                    },
                    success=success,
                    error=error,
                    duration_ms=duration_ms,
                )
            except Exception as analytics_error:
                logger.debug("Tool analytics failed for %s: %s", name, analytics_error)
            finally:
                current_harness_context.reset(harness_token)

    @staticmethod
    def _analytics_actor_id() -> str:
        """Return a stable hashed actor without exposing the source API key."""
        try:
            from wandb_mcp_server.analytics import current_actor_id

            return current_actor_id() or "unknown"
        except Exception:
            return "unknown"


__all__ = ["InstrumentedFastMCP", "structured_result_error"]
