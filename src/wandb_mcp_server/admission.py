"""Bounded, actor-aware admission control for public MCP tool calls."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Deque, Mapping


current_tool_deadline: ContextVar[float | None] = ContextVar("mcp_tool_deadline", default=None)


class AdmissionRejected(RuntimeError):
    """Raised when a tool call cannot obtain capacity before its wait budget."""


class ToolDeadlineExceeded(TimeoutError):
    """Raised by bounded loops after the current tool deadline expires."""


def raise_if_tool_deadline_exceeded() -> None:
    """Stop bounded application loops after the public tool deadline."""
    deadline = current_tool_deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise ToolDeadlineExceeded("MCP tool execution deadline exceeded")


@dataclass(frozen=True)
class AdmissionLease:
    """A capacity lease returned by :class:`WeightedAdmissionController`."""

    controller: "WeightedAdmissionController"
    actor_id: str
    weight: int
    queue_ms: float

    async def release(self) -> None:
        await self.controller.release(self.actor_id, self.weight)


@dataclass(frozen=True)
class _Waiter:
    actor_id: str
    weight: int


class WeightedAdmissionController:
    """FIFO weighted limiter with both actor and process capacity bounds."""

    def __init__(
        self,
        *,
        actor_capacity: int,
        process_capacity: int,
        wait_timeout_seconds: float,
    ) -> None:
        if actor_capacity < 1 or process_capacity < 1:
            raise ValueError("admission capacities must be positive")
        if actor_capacity > process_capacity:
            raise ValueError("actor admission capacity cannot exceed process capacity")
        if wait_timeout_seconds < 0:
            raise ValueError("admission wait timeout cannot be negative")
        self.actor_capacity = actor_capacity
        self.process_capacity = process_capacity
        self.wait_timeout_seconds = wait_timeout_seconds
        self._condition = asyncio.Condition()
        self._queue: Deque[_Waiter] = deque()
        self._process_in_use = 0
        self._actor_in_use: dict[str, int] = defaultdict(int)

    def _has_capacity(self, waiter: _Waiter) -> bool:
        return (
            self._process_in_use + waiter.weight <= self.process_capacity
            and self._actor_in_use[waiter.actor_id] + waiter.weight <= self.actor_capacity
        )

    async def acquire(self, actor_id: str, weight: int) -> AdmissionLease:
        """Acquire capacity in FIFO order or raise :class:`AdmissionRejected`."""
        if weight < 1 or weight > self.actor_capacity or weight > self.process_capacity:
            raise ValueError("admission weight exceeds configured capacity")
        waiter = _Waiter(actor_id=actor_id or "unknown", weight=weight)
        started = time.monotonic()
        async with self._condition:
            self._queue.append(waiter)
            try:
                async with asyncio.timeout(self.wait_timeout_seconds):
                    while self._queue[0] is not waiter or not self._has_capacity(waiter):
                        await self._condition.wait()
            except TimeoutError as exc:
                self._queue.remove(waiter)
                self._condition.notify_all()
                raise AdmissionRejected("tool admission wait budget exceeded") from exc
            except BaseException:
                self._queue.remove(waiter)
                self._condition.notify_all()
                raise

            self._queue.popleft()
            self._process_in_use += weight
            self._actor_in_use[waiter.actor_id] += weight
            self._condition.notify_all()
        return AdmissionLease(
            controller=self,
            actor_id=waiter.actor_id,
            weight=weight,
            queue_ms=round((time.monotonic() - started) * 1000, 2),
        )

    async def release(self, actor_id: str, weight: int) -> None:
        """Release a lease and wake queued callers."""
        async with self._condition:
            actor_used = self._actor_in_use.get(actor_id, 0)
            if weight > actor_used or weight > self._process_in_use:
                raise RuntimeError("admission capacity released more than once")
            self._process_in_use -= weight
            remaining = actor_used - weight
            if remaining:
                self._actor_in_use[actor_id] = remaining
            else:
                self._actor_in_use.pop(actor_id, None)
            self._condition.notify_all()

    @property
    def process_in_use(self) -> int:
        return self._process_in_use


LIGHT_TOOLS = frozenset(
    {
        "list_entities_tool",
        "query_wandb_entity_projects",
        "get_artifact_details_tool",
        "list_registries_tool",
        "list_registry_collections_tool",
        "list_wandb_automations_tool",
        "list_wandb_integrations_tool",
    }
)

EXPENSIVE_TOOLS = frozenset(
    {
        "resolve_trace_roots_tool",
        "create_wandb_report_tool",
        "log_analysis_to_wandb",
    }
)

HEAVY_TOOLS = frozenset(
    {
        "probe_project_tool",
        "compare_runs_tool",
        "diagnose_run_tool",
        "infer_trace_schema_tool",
        "summarize_evaluation_tool",
        "query_weave_traces_tool",
        "query_wandb_graphql_tool",
        "compare_artifact_versions_tool",
    }
)


def tool_cost(name: str, arguments: Mapping[str, Any] | None = None) -> tuple[str, int]:
    """Return the request-aware telemetry cost class and admission weight."""
    arguments = arguments or {}
    if name == "query_wandb_tool":
        if arguments.get("response_mode") == "count":
            return "light", 1
        resource = arguments.get("resource")
        include = set(arguments.get("include") or [])
        full_summary = "summary" in include and not arguments.get("summary_keys")
        full_config = "config" in include and not arguments.get("config_keys")
        if full_summary or full_config or include & {"system_metrics", "spec"}:
            return "heavy", 4
        if resource in {"project", "run", "sweep"} and not include:
            return "light", 1
        return "expensive", 2
    if name == "get_run_history_tool":
        if (
            arguments.get("target_x") is not None
            or arguments.get("min_step") is not None
            or arguments.get("max_step") is not None
        ):
            return "heavy", 4
        return "expensive", 2
    if name == "get_artifact_details_tool" and arguments.get("include_files"):
        return "heavy", 4
    if name == "list_artifact_versions_tool":
        if arguments.get("created_after") or arguments.get("created_before"):
            return "heavy", 4
        return "expensive", 2
    if name == "count_weave_traces_tool":
        return "light", 1
    if name in LIGHT_TOOLS:
        return "light", 1
    if name in HEAVY_TOOLS:
        return "heavy", 4
    if name in EXPENSIVE_TOOLS:
        return "expensive", 2
    return "heavy", 4


__all__ = [
    "AdmissionLease",
    "AdmissionRejected",
    "ToolDeadlineExceeded",
    "WeightedAdmissionController",
    "current_tool_deadline",
    "raise_if_tool_deadline_exceeded",
    "tool_cost",
]
