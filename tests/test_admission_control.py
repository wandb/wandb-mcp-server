"""Concurrency, fairness, and overload tests for MCP tool admission."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from wandb_mcp_server.admission import (
    AdmissionRejected,
    WeightedAdmissionController,
    tool_cost,
)
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP


@pytest.mark.asyncio
async def test_fifteen_same_actor_calls_never_exceed_four_units() -> None:
    controller = WeightedAdmissionController(
        actor_capacity=4,
        process_capacity=16,
        wait_timeout_seconds=1,
    )
    active = 0
    maximum = 0
    state_lock = asyncio.Lock()

    async def work() -> None:
        nonlocal active, maximum
        lease = await controller.acquire("actor-a", 1)
        try:
            async with state_lock:
                active += 1
                maximum = max(maximum, active)
            await asyncio.sleep(0.005)
        finally:
            async with state_lock:
                active -= 1
            await lease.release()

    await asyncio.gather(*(work() for _ in range(15)))

    assert maximum == 4
    assert controller.process_in_use == 0


@pytest.mark.asyncio
async def test_mixed_actors_respect_actor_and_process_capacity() -> None:
    controller = WeightedAdmissionController(
        actor_capacity=4,
        process_capacity=16,
        wait_timeout_seconds=1,
    )
    active_by_actor: dict[str, int] = {}
    maximum_by_actor: dict[str, int] = {}
    process_active = 0
    process_maximum = 0
    state_lock = asyncio.Lock()

    async def work(actor: str) -> None:
        nonlocal process_active, process_maximum
        lease = await controller.acquire(actor, 1)
        try:
            async with state_lock:
                active_by_actor[actor] = active_by_actor.get(actor, 0) + 1
                maximum_by_actor[actor] = max(
                    maximum_by_actor.get(actor, 0),
                    active_by_actor[actor],
                )
                process_active += 1
                process_maximum = max(process_maximum, process_active)
            await asyncio.sleep(0.005)
        finally:
            async with state_lock:
                active_by_actor[actor] -= 1
                process_active -= 1
            await lease.release()

    await asyncio.gather(*(work(f"actor-{index % 5}") for index in range(30)))

    assert process_maximum <= 16
    assert all(maximum <= 4 for maximum in maximum_by_actor.values())
    assert set(maximum_by_actor) == {f"actor-{index}" for index in range(5)}


@pytest.mark.asyncio
async def test_heavy_call_rejects_after_wait_budget_and_recovers() -> None:
    controller = WeightedAdmissionController(
        actor_capacity=4,
        process_capacity=16,
        wait_timeout_seconds=0.01,
    )
    lease = await controller.acquire("actor-a", 4)

    with pytest.raises(AdmissionRejected):
        await controller.acquire("actor-a", 4)

    await lease.release()
    recovered = await controller.acquire("actor-a", 4)
    await recovered.release()
    assert controller.process_in_use == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_queue_or_capacity() -> None:
    controller = WeightedAdmissionController(
        actor_capacity=1,
        process_capacity=1,
        wait_timeout_seconds=1,
    )
    lease = await controller.acquire("actor-a", 1)
    queued = asyncio.create_task(controller.acquire("actor-a", 1))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    await lease.release()
    recovered = await controller.acquire("actor-a", 1)
    await recovered.release()
    assert controller.process_in_use == 0


def test_tool_costs_are_stable_and_unknown_is_heavy() -> None:
    assert tool_cost("list_entities_tool") == ("light", 1)
    assert tool_cost("get_run_history_tool") == ("expensive", 2)
    assert tool_cost("get_run_history_tool", {"keys": [f"metric-{index}" for index in range(8)]}) == (
        "expensive",
        2,
    )
    assert tool_cost("get_run_history_tool", {"keys": [f"metric-{index}" for index in range(9)]}) == (
        "heavy",
        4,
    )
    assert tool_cost("get_run_history_tool", {"target_x": 100}) == ("heavy", 4)
    assert tool_cost("get_run_history_tool", {"min_step": 0, "max_step": 100}) == ("heavy", 4)
    assert tool_cost("list_artifact_versions_tool") == ("expensive", 2)
    assert tool_cost("list_artifact_versions_tool", {"tags": ["production"]}) == ("heavy", 4)
    assert tool_cost("list_artifact_versions_tool", {"created_after": "2026-01-01"}) == ("heavy", 4)
    assert tool_cost("query_wandb_tool", {"resource": "runs", "response_mode": "count"}) == ("light", 1)
    assert tool_cost(
        "query_wandb_tool",
        {"resource": "runs", "summary_keys": ["loss"]},
    ) == ("expensive", 2)
    assert tool_cost(
        "query_wandb_tool",
        {"resource": "runs", "include": ["summary"]},
    ) == ("heavy", 4)
    assert tool_cost("probe_project_tool") == ("heavy", 4)
    assert tool_cost("query_wandb_graphql_tool") == ("heavy", 4)
    assert tool_cost("future_unclassified_tool") == ("heavy", 4)


@pytest.mark.asyncio
async def test_public_boundary_returns_stable_busy_error(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("admission-test")
    server._admission_controller = WeightedAdmissionController(
        actor_capacity=1,
        process_capacity=1,
        wait_timeout_seconds=0.01,
    )
    started = asyncio.Event()
    finish = asyncio.Event()

    @server.tool(name="list_entities_tool")
    async def bounded_tool() -> str:
        started.set()
        await finish.wait()
        return "ok"

    first = asyncio.create_task(server.call_tool("list_entities_tool", {}))
    await started.wait()
    with pytest.raises(ToolError, match="server_busy"):
        await server.call_tool("list_entities_tool", {})
    finish.set()
    await first


@pytest.mark.asyncio
async def test_public_boundary_enforces_async_tool_deadline(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", True)
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_TOOL_TIMEOUT_SECONDS", 0.01)
    server = InstrumentedFastMCP("deadline-test")

    @server.tool(name="slow_async_tool")
    async def slow_async_tool() -> str:
        await asyncio.sleep(1)
        return "late"

    with pytest.raises(ToolError, match="tool_timeout"):
        await server.call_tool("slow_async_tool", {})


@pytest.mark.asyncio
async def test_public_boundary_maps_upstream_rate_limit_without_retry(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("upstream-rate-limit-test")

    @server.tool(name="list_entities_tool")
    async def rate_limited_tool() -> str:
        error = RuntimeError("HTTP 429 Too Many Requests")
        error.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": "3"},
            reason="Too Many Requests",
            text="",
        )
        raise error

    with pytest.raises(ToolError) as caught:
        await server.call_tool("list_entities_tool", {})

    assert '"error": "server_busy"' in str(caught.value)
    assert '"retry_after_ms": 3000' in str(caught.value)


@pytest.mark.asyncio
async def test_write_timeout_reports_unknown_outcome_and_is_not_retryable(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", True)
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_TOOL_TIMEOUT_SECONDS", 0.01)
    server = InstrumentedFastMCP("write-timeout-test")

    @server.tool(name="create_wandb_report_tool")
    def slow_write() -> str:
        time.sleep(0.05)
        return "created"

    with pytest.raises(ToolError) as caught:
        await server.call_tool("create_wandb_report_tool", {})

    rendered = str(caught.value)
    assert '"error": "outcome_unknown"' in rendered
    assert '"retryable": false' in rendered
    assert "retry_after" not in rendered
    await asyncio.sleep(0.06)


@pytest.mark.asyncio
async def test_write_rate_limit_reports_unknown_outcome_and_is_not_retryable(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("write-rate-limit-test")

    @server.tool(name="log_analysis_to_wandb")
    async def rate_limited_write() -> None:
        error = RuntimeError("HTTP 429 Too Many Requests")
        error.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": "3"},
            reason="Too Many Requests",
            text="",
        )
        raise error

    with pytest.raises(ToolError) as caught:
        await server.call_tool("log_analysis_to_wandb", {})

    rendered = str(caught.value)
    assert '"error": "outcome_unknown"' in rendered
    assert '"retryable": false' in rendered
    assert "retry_after" not in rendered


@pytest.mark.asyncio
async def test_cancelled_admission_wait_is_recorded_as_cancelled(monkeypatch) -> None:
    tracker = MagicMock()
    monkeypatch.setattr(
        "wandb_mcp_server.analytics.get_analytics_tracker",
        lambda: tracker,
    )
    server = InstrumentedFastMCP("admission-cancel-analytics-test")
    server._admission_controller = WeightedAdmissionController(
        actor_capacity=1,
        process_capacity=1,
        wait_timeout_seconds=1,
    )
    started = asyncio.Event()
    finish = asyncio.Event()

    @server.tool(name="list_entities_tool")
    async def held_tool() -> str:
        started.set()
        await finish.wait()
        return "ok"

    first = asyncio.create_task(server.call_tool("list_entities_tool", {}))
    await started.wait()
    queued = asyncio.create_task(server.call_tool("list_entities_tool", {}))
    await asyncio.sleep(0.01)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    cancelled_events = [
        call.kwargs
        for call in tracker.track_tool_call.call_args_list
        if call.kwargs["params"]["admission_outcome"] == "cancelled"
    ]
    assert len(cancelled_events) == 1
    assert cancelled_events[0]["success"] is False
    assert cancelled_events[0]["error"].startswith("cancelled:")

    finish.set()
    await first


def test_sync_tools_are_registered_as_async_threaded_calls_when_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", True)
    server = InstrumentedFastMCP("threaded-test")

    @server.tool(name="sync_tool")
    def sync_tool() -> str:
        return threading.current_thread().name

    assert server._tool_manager.get_tool("sync_tool").is_async is True


@pytest.mark.asyncio
async def test_local_dispatch_keeps_sync_work_off_event_loop(monkeypatch) -> None:
    from wandb_mcp_server.instrumented_server import run_sync_in_current_tool

    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", False)
    monkeypatch.setattr(
        "wandb_mcp_server.instrumented_server.MCP_ADMISSION_CONTROL_ENABLED",
        False,
    )
    server = InstrumentedFastMCP("local-off-loop-test")
    caller_thread = threading.get_ident()
    observed_threads: list[int] = []

    @server.tool(name="local_sync_tool")
    def local_sync_tool() -> str:
        observed_threads.append(threading.get_ident())
        return "sync"

    @server.tool(name="local_async_tool")
    async def local_async_tool() -> str:
        return await run_sync_in_current_tool(lambda: observed_threads.append(threading.get_ident()) or "async")

    assert server._tool_manager.get_tool("local_sync_tool").is_async is True
    await server.call_tool("local_sync_tool", {})
    await server.call_tool("local_async_tool", {})

    assert len(observed_threads) == 2
    assert all(thread_id != caller_thread for thread_id in observed_threads)


@pytest.mark.asyncio
async def test_timed_out_sync_tool_holds_permit_until_worker_finishes(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", True)
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_TOOL_TIMEOUT_SECONDS", 0.02)
    server = InstrumentedFastMCP("sync-timeout-test")
    server._admission_controller = WeightedAdmissionController(
        actor_capacity=1,
        process_capacity=1,
        wait_timeout_seconds=0.01,
    )
    started = threading.Event()
    finish = threading.Event()
    state_lock = threading.Lock()
    active = 0
    peak = 0

    @server.tool(name="list_entities_tool")
    def blocking_tool() -> str:
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        started.set()
        finish.wait(timeout=2)
        with state_lock:
            active -= 1
        return "done"

    first = asyncio.create_task(server.call_tool("list_entities_tool", {}))
    assert await asyncio.to_thread(started.wait, 1)
    with pytest.raises(ToolError, match="tool_timeout"):
        await first

    assert server._admission_controller.process_in_use == 1
    with pytest.raises(ToolError, match="server_busy"):
        await server.call_tool("list_entities_tool", {})
    assert peak == 1

    finish.set()
    deadline = time.monotonic() + 1
    while server._admission_controller.process_in_use and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert server._admission_controller.process_in_use == 0


@pytest.mark.asyncio
async def test_cancelled_sync_tool_holds_permit_until_worker_finishes(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", True)
    server = InstrumentedFastMCP("sync-cancel-test")
    server._admission_controller = WeightedAdmissionController(
        actor_capacity=1,
        process_capacity=1,
        wait_timeout_seconds=0.01,
    )
    started = threading.Event()
    finish = threading.Event()

    @server.tool(name="list_entities_tool")
    def blocking_tool() -> str:
        started.set()
        finish.wait(timeout=2)
        return "done"

    first = asyncio.create_task(server.call_tool("list_entities_tool", {}))
    assert await asyncio.to_thread(started.wait, 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert server._admission_controller.process_in_use == 1
    with pytest.raises(ToolError, match="server_busy"):
        await server.call_tool("list_entities_tool", {})

    finish.set()
    deadline = time.monotonic() + 1
    while server._admission_controller.process_in_use and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert server._admission_controller.process_in_use == 0


@pytest.mark.asyncio
async def test_async_tool_blocking_section_uses_bounded_tracked_executor(monkeypatch) -> None:
    from wandb_mcp_server.instrumented_server import run_sync_in_current_tool

    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_HOSTED_MODE", True)
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_TOOL_TIMEOUT_SECONDS", 0.02)
    server = InstrumentedFastMCP("async-sync-section-test")
    server._admission_controller = WeightedAdmissionController(
        actor_capacity=4,
        process_capacity=4,
        wait_timeout_seconds=0.01,
    )
    started = threading.Event()
    finish = threading.Event()

    @server.tool(name="query_weave_traces_tool")
    async def tool_with_blocking_sdk_section() -> str:
        def blocking_sdk_call() -> str:
            started.set()
            finish.wait(timeout=2)
            return "done"

        return await run_sync_in_current_tool(blocking_sdk_call)

    first = asyncio.create_task(server.call_tool("query_weave_traces_tool", {}))
    assert await asyncio.to_thread(started.wait, 1)
    heartbeat = False

    async def pulse() -> None:
        nonlocal heartbeat
        await asyncio.sleep(0)
        heartbeat = True

    await asyncio.wait_for(pulse(), timeout=0.1)
    assert heartbeat
    with pytest.raises(ToolError, match="tool_timeout"):
        await first
    assert server._admission_controller.process_in_use == 4
    with pytest.raises(ToolError, match="server_busy"):
        await server.call_tool("query_weave_traces_tool", {})

    finish.set()
    deadline = time.monotonic() + 1
    while server._admission_controller.process_in_use and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert server._admission_controller.process_in_use == 0
