"""FastMCP instrumentation at the public protocol boundary."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import TextContent

from wandb_mcp_server.admission import (
    AdmissionRejected,
    WeightedAdmissionController,
    current_tool_deadline,
    tool_cost,
)
from wandb_mcp_server.api_client import (
    wandb_report_creation_failed_from_exception,
    wandb_write_outcome_unknown_from_exception,
    wandb_server_busy_from_exception,
)
from wandb_mcp_server.config import (
    MCP_ADMISSION_ACTOR_CAPACITY,
    MCP_ADMISSION_CONTROL_ENABLED,
    MCP_ADMISSION_PROCESS_CAPACITY,
    MCP_ADMISSION_WAIT_MS,
    MCP_HOSTED_MODE,
    MCP_SYNC_TOOL_WORKERS,
    MCP_TOOL_TIMEOUT_SECONDS,
)
from wandb_mcp_server.error_sanitizer import (
    MAX_EXTERNAL_ERROR_CHARS,
    sanitize_sensitive_text,
    sanitize_sensitive_value,
)
from wandb_mcp_server.harness import (
    HarnessContext,
    context_from_sdk_request,
    current_harness_context,
)
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)
_NON_IDEMPOTENT_WRITE_TOOLS = frozenset(
    {
        "create_wandb_report_tool",
        "log_analysis_to_wandb",
        "aria_send_message",
    }
)
_ARIA_TOOL_COSTS = {
    "aria_send_message": ("heavy", 4),
    "aria_get_turn": ("light", 1),
    "aria_get_turns": ("heavy", 4),
}
_TELEMETRY_RESULT_ERROR_CODES = frozenset(
    {
        "agents_api_unavailable",
        "agents_query_failed",
        "api_error",
        "artifact_inventory_unavailable",
        "auth_required",
        "authentication_failed",
        "count_failed",
        "evaluation_query_failed",
        "history_fetch_failed",
        "history_query_failed",
        "invalid_cursor",
        "invalid_input",
        "invalid_request",
        "log_failed",
        "malformed_response",
        "organization_required",
        "organization_resolution_failed",
        "out_of_memory",
        "pagination_cursor_unavailable",
        "permission_denied",
        "project_probe_failed",
        "project_query_failed",
        "query_failed",
        "query_too_large",
        "read_only_violation",
        "registry_query_failed",
        "report_creation_failed",
        "resolve_failed",
        "resource_not_found",
        "response_too_large",
        "run_not_found",
        "schema_query_failed",
        "selective_read_unavailable",
        "server_busy",
        "target_not_logged",
        "tool_timeout",
        "upstream_error",
    }
)


def _dispatch_tool_cost(name: str, arguments: dict[str, Any]) -> tuple[str, int]:
    """Return boundary costs for optional tools before the shared fallback."""
    aria_cost = _ARIA_TOOL_COSTS.get(name)
    return aria_cost if aria_cost is not None else tool_cost(name, arguments)


@dataclass
class _SyncCallState:
    futures: list[asyncio.Future[Any]] = field(default_factory=list)


_current_sync_call_state: ContextVar[_SyncCallState | None] = ContextVar(
    "mcp_sync_call_state",
    default=None,
)
_current_sync_executor: ContextVar[ThreadPoolExecutor | None] = ContextVar(
    "mcp_sync_executor",
    default=None,
)


def register_current_sync_future(future: asyncio.Future[Any]) -> None:
    """Associate externally scheduled sync work with the active tool call.

    A small number of async tools schedule blocking library work themselves.
    Registering those futures here lets admission retain the physical permit
    after protocol cancellation or timeout, just like ordinary synchronous
    tools dispatched by :meth:`add_tool`.
    """
    state = _current_sync_call_state.get()
    if state is not None:
        state.futures.append(future)


async def run_sync_in_current_tool(call: Any) -> Any:
    """Run blocking work in the active server's bounded executor.

    Async public tools use this for isolated blocking SDK sections. The
    resulting future participates in the same timeout/cancellation lease
    tracking as tools that are synchronous end-to-end.
    """
    executor = _current_sync_executor.get()
    if executor is None:
        # Local stdio and direct library callers retain the historical AnyIO
        # worker behavior. Hosted/admission-controlled dispatch installs the
        # bounded executor so timed-out work cannot be replaced indefinitely.
        return await anyio.to_thread.run_sync(call, abandon_on_cancel=False)
    context = copy_context()
    future = asyncio.get_running_loop().run_in_executor(executor, context.run, call)
    register_current_sync_future(future)
    return await asyncio.shield(future)


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


def _error_category_from_mapping(value: dict[str, Any]) -> str | None:
    """Return a bounded categorical error without copying customer text."""
    error = value.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        candidate = error.get("code") or error.get("type") or "tool_error"
    else:
        candidate = error
    return str(candidate) if candidate in _TELEMETRY_RESULT_ERROR_CODES else "tool_error"


def _telemetry_error(category: str) -> str:
    """Format a low-cardinality error so Datadog retains its error kind."""
    return f"{category}: tool failed"


def _exception_error_category(exc: BaseException) -> str:
    """Return a bounded exception class name without copying exception text."""
    candidate = type(exc).__name__
    if (
        0 < len(candidate) <= 64
        and candidate[0].isalpha()
        and all(character.isalnum() or character == "_" for character in candidate)
    ):
        return candidate
    return "exception"


def _structured_result_error_category(result: Any) -> str | None:
    """Classify a structured MCP failure for telemetry without its message."""
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        return "mcp_error"
    if isinstance(result, dict):
        category = _error_category_from_mapping(result)
        if category:
            return category
        if "result" in result:
            return _structured_result_error_category(result["result"])
        return None
    if isinstance(result, str):
        try:
            decoded = json.loads(result)
        except (TypeError, ValueError):
            return None
        return _error_category_from_mapping(decoded) if isinstance(decoded, dict) else None
    content = getattr(result, "content", None)
    if content is not None and content is not result:
        category = _structured_result_error_category(content)
        if category:
            return category
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(result, "structured_content", None)
    if structured_content is not None and structured_content is not result:
        category = _structured_result_error_category(structured_content)
        if category:
            return category
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        for block in result:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                category = _structured_result_error_category(text)
                if category:
                    return category
            category = _structured_result_error_category(block)
            if category:
                return category
    return None


def _structured_error_code(result: Any) -> str:
    """Preserve a bounded public error code when replacing oversized details."""
    if isinstance(result, dict):
        error = result.get("error")
        if isinstance(error, str) and 0 < len(error) <= 64:
            return error if error.replace("_", "").replace("-", "").isalnum() else "upstream_error"
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            if isinstance(code, str) and 0 < len(code) <= 64:
                return code if code.replace("_", "").replace("-", "").isalnum() else "upstream_error"
        nested = result.get("result")
        if nested is not None and nested is not result:
            return _structured_error_code(nested)
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(result, "structured_content", None)
    if structured_content is not None and structured_content is not result:
        return _structured_error_code(structured_content)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        for child in result:
            code = _structured_error_code(child)
            if code != "upstream_error":
                return code
    return "upstream_error"


def _bounded_error_result(result: Any) -> Any:
    """Replace an oversized structured error while retaining FastMCP's shape."""
    if len(str(result)) <= MAX_EXTERNAL_ERROR_CHARS:
        return result
    payload = {
        "error": _structured_error_code(result),
        "message": "Upstream error details exceeded the safe response limit.",
        "details_truncated": True,
    }
    return (
        [TextContent(type="text", text=json.dumps(payload, separators=(",", ":")))],
        payload,
    )


class InstrumentedFastMCP(FastMCP):
    """FastMCP server with bounded dispatch and one event per public tool call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bounded_dispatch_enabled = MCP_HOSTED_MODE or MCP_ADMISSION_CONTROL_ENABLED
        self._sync_executor = ThreadPoolExecutor(
            max_workers=MCP_SYNC_TOOL_WORKERS,
            thread_name_prefix="mcp-tool",
        )
        self._deferred_release_tasks: set[asyncio.Task[None]] = set()
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
                if not self._bounded_dispatch_enabled:
                    return await anyio.to_thread.run_sync(call, abandon_on_cancel=False)
                context = copy_context()
                future = asyncio.get_running_loop().run_in_executor(
                    self._sync_executor,
                    context.run,
                    call,
                )
                register_current_sync_future(future)
                # The executor future must survive a protocol timeout. Its
                # admission lease is released only when physical work ends.
                return await asyncio.shield(future)

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
        sync_state = _SyncCallState()
        sync_state_token = _current_sync_call_state.set(sync_state)
        sync_executor_token = _current_sync_executor.set(
            self._sync_executor if self._bounded_dispatch_enabled else None
        )
        started = time.monotonic()
        success = True
        error: str | None = None
        lease = None
        deadline_token = None
        cost_class, weight = _dispatch_tool_cost(name, arguments)
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
                except asyncio.CancelledError:
                    queue_ms = round((time.monotonic() - admission_started) * 1000, 2)
                    admission_outcome = "cancelled"
                    success = False
                    error = "cancelled: tool admission wait was cancelled"
                    raise
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
                is_write = name in _NON_IDEMPOTENT_WRITE_TOOLS
                error = (
                    "outcome_unknown: write timed out before completion was confirmed"
                    if is_write
                    else "tool_timeout: MCP tool execution deadline exceeded"
                )
                raise ToolError(
                    json.dumps(
                        (
                            {
                                "error": "outcome_unknown",
                                "message": (
                                    "The write may have completed after the MCP deadline. "
                                    "Verify W&B state before deciding whether to retry."
                                ),
                                "retryable": False,
                            }
                            if is_write
                            else {
                                "error": "tool_timeout",
                                "message": "The MCP tool exceeded its execution deadline.",
                                "retryable": True,
                                "retry_after_ms": 1000,
                            }
                        )
                    )
                ) from exc
            except BaseException as exc:
                if name in _ARIA_TOOL_COSTS:
                    # ARIA already maps its own response codes, bounded
                    # Retry-After value, and submission ambiguity. Handle that
                    # contract before the generic W&B overload classifier so a
                    # caller does not lose its ARIA turn handle or retry policy.
                    # ARIA errors can contain turn handles and customer scope;
                    # preserve the sanitized native MCP error for the caller,
                    # while keeping logs and analytics categorical.
                    success = False
                    error = "aria_error: ARIA tool returned a structured failure"
                    sanitized_message = sanitize_sensitive_text(
                        str(exc),
                        max_chars=MAX_EXTERNAL_ERROR_CHARS,
                    )
                    if sanitized_message != str(exc):
                        raise ToolError(sanitized_message) from exc
                    raise
                if name in _NON_IDEMPOTENT_WRITE_TOOLS and wandb_write_outcome_unknown_from_exception(exc):
                    success = False
                    error = "outcome_unknown: W&B did not confirm the write result"
                    raise ToolError(
                        json.dumps(
                            {
                                "error": "outcome_unknown",
                                "message": (
                                    "W&B did not confirm whether the write completed. "
                                    "Verify W&B state before deciding whether to retry."
                                ),
                                "retryable": False,
                            }
                        )
                    ) from exc
                if name == "create_wandb_report_tool" and wandb_report_creation_failed_from_exception(exc):
                    success = False
                    error = "report_creation_failed: W&B rejected the report write"
                    raise ToolError(
                        json.dumps(
                            {
                                "error": "report_creation_failed",
                                "message": "The W&B report could not be created.",
                                "retryable": False,
                            }
                        )
                    ) from exc
                if busy := wandb_server_busy_from_exception(exc):
                    success = False
                    if name in _NON_IDEMPOTENT_WRITE_TOOLS:
                        error = f"outcome_unknown: W&B did not confirm the write result (HTTP {busy.status_code})"
                        raise ToolError(
                            json.dumps(
                                {
                                    "error": "outcome_unknown",
                                    "message": (
                                        "W&B did not confirm whether the write completed. "
                                        "Verify W&B state before deciding whether to retry."
                                    ),
                                    "retryable": False,
                                    "upstream_status": busy.status_code,
                                }
                            )
                        ) from exc
                    error = f"server_busy: upstream HTTP {busy.status_code}"
                    raise ToolError(json.dumps(busy.as_dict())) from exc
                if success:
                    success = False
                    error = _telemetry_error(_exception_error_category(exc))
                sanitized_message = sanitize_sensitive_text(
                    str(exc),
                    max_chars=MAX_EXTERNAL_ERROR_CHARS,
                )
                if sanitized_message != str(exc):
                    # Do not let FastMCP serialize the original exception when
                    # it contains request credentials or an internal service
                    # address. Safe exceptions retain their original type.
                    raise ToolError(sanitized_message) from exc
                raise
            structured_error = structured_result_error(result)
            result = sanitize_sensitive_value(
                result,
                _error_context=structured_error is not None,
            )
            if structured_error:
                result = _bounded_error_result(result)
                success = False
                error = _telemetry_error(_structured_result_error_category(result) or "tool_error")
            return result
        finally:
            if deadline_token is not None:
                current_tool_deadline.reset(deadline_token)
            if lease is not None:
                pending_workers = [future for future in sync_state.futures if not future.done()]
                if pending_workers:
                    self._defer_lease_release(
                        lease=lease,
                        worker_futures=pending_workers,
                        tool_name=name,
                    )
                else:
                    try:
                        await lease.release()
                    except Exception as release_error:
                        logger.error(
                            "Tool admission release failed for %s (%s)",
                            name,
                            type(release_error).__name__,
                        )
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
                logger.debug("Tool analytics failed for %s (%s)", name, type(analytics_error).__name__)
            finally:
                _current_sync_executor.reset(sync_executor_token)
                _current_sync_call_state.reset(sync_state_token)
                current_harness_context.reset(harness_token)

    def _defer_lease_release(
        self,
        *,
        lease: Any,
        worker_futures: list[asyncio.Future[Any]],
        tool_name: str,
    ) -> None:
        """Keep the admission lease until timed-out synchronous work is done."""

        async def _release_after_workers() -> None:
            await asyncio.gather(
                *(asyncio.shield(future) for future in worker_futures),
                return_exceptions=True,
            )
            try:
                await lease.release()
            except Exception as release_error:
                logger.error(
                    "Deferred tool admission release failed for %s (%s)",
                    tool_name,
                    type(release_error).__name__,
                )

        task = asyncio.create_task(_release_after_workers())
        self._deferred_release_tasks.add(task)
        task.add_done_callback(self._deferred_release_tasks.discard)

    def shutdown_sync_executor(self) -> None:
        """Stop accepting sync work without waiting on an uncooperative SDK call."""
        self._sync_executor.shutdown(wait=False, cancel_futures=True)

    async def run_stdio_async(self) -> None:
        try:
            await super().run_stdio_async()
        finally:
            self.shutdown_sync_executor()

    async def run_streamable_http_async(self) -> None:
        try:
            await super().run_streamable_http_async()
        finally:
            self.shutdown_sync_executor()

    async def run_sse_async(self, mount_path: str | None = None) -> None:
        try:
            await super().run_sse_async(mount_path)
        finally:
            self.shutdown_sync_executor()

    @staticmethod
    def _analytics_actor_id() -> str:
        """Return a stable hashed actor without exposing the source API key."""
        try:
            from wandb_mcp_server.analytics import current_actor_id

            return current_actor_id() or "unknown"
        except Exception:
            return "unknown"


__all__ = ["InstrumentedFastMCP", "structured_result_error"]
