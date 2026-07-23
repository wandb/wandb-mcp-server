"""Concurrency, fairness, and overload tests for MCP tool admission."""

from __future__ import annotations

import asyncio
import threading

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
    assert tool_cost("get_run_history_tool", {"target_x": 100}) == ("heavy", 4)
    assert tool_cost("get_run_history_tool", {"min_step": 0, "max_step": 100}) == ("heavy", 4)
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


def test_sync_tools_are_registered_as_async_threaded_calls(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("threaded-test")

    @server.tool(name="sync_tool")
    def sync_tool() -> str:
        return threading.current_thread().name

    assert server._tool_manager.get_tool("sync_tool").is_async is True
