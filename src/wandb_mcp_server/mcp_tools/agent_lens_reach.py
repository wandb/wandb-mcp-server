"""Explain an empty Agent Lens instrumentation check from what reached the trace server.

A zero-span health report would read as healthy. These helpers ask the trace
server whether agent spans or Weave calls arrived, and otherwise list the
exporter checks in order.
"""

from __future__ import annotations

import json
from functools import partial
from typing import Optional

from pydantic import ValidationError

from wandb_mcp_server.api_client import raise_for_wandb_server_busy
from wandb_mcp_server.config import WF_TRACE_SERVER_URL, structured_error
from wandb_mcp_server.instrumented_server import run_sync_in_current_tool
from wandb_mcp_server.mcp_tools.agent_lens_doctor_models import AgentSpan, AgentSpanPage, DoctorScope, NoSpansReport
from wandb_mcp_server.mcp_tools.agents import query_agent_spans
from wandb_mcp_server.mcp_tools.count_traces import count_traces
from wandb_mcp_server.tool_results import ToolErrorJSON, tool_error_json
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

LATEST_CONVERSATION = "latest"
AGENTS_OTLP_ENDPOINT = f"{WF_TRACE_SERVER_URL.rstrip('/')}/agents/otel/v1/traces"
INELIGIBLE_SPANS_DIAGNOSIS = (
    "Agent spans arrived in this window, but none is eligible for instrumentation checks. "
    "Evaluation traces and Weave Signals spans are excluded."
)
REACH_CHECKS = (
    f"OTLP exporter endpoint is {AGENTS_OTLP_ENDPOINT}.",
    "Exporter sends a W&B API key in a wandb-api-key header.",
    "Tracer resource sets wandb.entity and wandb.project to this project.",
    "App flushes or shuts down its tracer provider before exiting, so batched spans are sent.",
)


async def no_spans_report(scope: DoctorScope, *, has_agent_spans: bool | None = None) -> str:
    """Explain an empty check using agent span presence or Weave calls in range."""
    if has_agent_spans is None:
        any_span = await newest_agent_span(scope)
        if isinstance(any_span, ToolErrorJSON):
            return any_span

        has_agent_spans = any_span is not None

    if has_agent_spans:
        diagnosis = _filter_diagnosis(scope) if scope.filtered else INELIGIBLE_SPANS_DIAGNOSIS
        report = NoSpansReport(verdict="no_spans", scope=scope.as_dict(), span_count=0, diagnosis=diagnosis)
        return json.dumps(report)

    calls = await _count_calls(scope)
    report = NoSpansReport(
        verdict="no_spans",
        scope=scope.as_dict(),
        span_count=0,
        diagnosis=_reach_diagnosis(calls),
        calls_in_window=calls,
    )
    if not calls:
        report["checks"] = list(REACH_CHECKS)

    return json.dumps(report)


async def newest_agent_span(scope: DoctorScope) -> Optional[AgentSpan] | ToolErrorJSON:
    """Return the newest agent span in `scope`'s range."""
    response = await run_sync_in_current_tool(
        partial(
            query_agent_spans,
            scope.entity_name,
            scope.project_name,
            started_after=scope.start_at,
            started_before=scope.end_at,
            sort_by="started_at",
            sort_direction="desc",
            limit=1,
        )
    )
    if isinstance(response, ToolErrorJSON):
        return response

    try:
        page = AgentSpanPage.model_validate_json(response)
    except ValidationError:
        error = structured_error("agents_query_failed", "The Agents API returned an unexpected span response.")
        return tool_error_json(error)

    return page.spans[0] if page.spans else None


def _filter_diagnosis(scope: DoctorScope) -> str:
    """Explain agent spans that exist in range but miss `scope`'s conversation or trace filter."""
    if scope.conversation_id == LATEST_CONVERSATION:
        return (
            "Agent spans arrived in this window, but the newest agent span carries no conversation id. "
            "Set gen_ai.conversation.id on each turn's root span, or call without conversation_id to check "
            "the whole project."
        )

    return (
        "This project has agent spans in this window, but no eligible span matches this conversation_id or "
        "trace_id, so spans are reaching Agent Lens. Check that id, pass start_at and end_at covering when it "
        'ran, or pass conversation_id="latest" and omit trace_id. Evaluation traces and Weave Signals spans are excluded.'
    )


def _reach_diagnosis(calls: Optional[int]) -> str:
    """Explain zero agent spans from the Weave call count in the same range."""
    if calls is None:
        return "No agent spans arrived, and counting Weave calls failed. Work through `checks` in order."
    if calls == 0:
        return "Nothing reached this project in this window. Work through `checks` in order."

    return (
        f"No agent spans arrived, but {calls} Weave call(s) did, so this exporter targets the legacy "
        f"/otel/v1/traces endpoint. Point OTLP export at {AGENTS_OTLP_ENDPOINT} and set resource "
        "attributes wandb.entity and wandb.project."
    )


async def _count_calls(scope: DoctorScope) -> Optional[int]:
    """Count Weave calls started in `scope`'s range, or None on non-overload failures."""
    time_range = {"time_range": {"start": scope.start_at, "end": scope.end_at}}
    try:
        count = await run_sync_in_current_tool(partial(count_traces, scope.entity_name, scope.project_name, time_range))
    except Exception as error:
        raise_for_wandb_server_busy(error, require_overload_hint=False)
        logger.warning("Weave calls check failed (%s)", type(error).__name__)
        return None

    return count
