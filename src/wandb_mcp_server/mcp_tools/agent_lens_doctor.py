"""MCP tool that ranks a project's Agent Lens instrumentation failures for a fix loop.

Wraps `GET /instrumentation/health` (`internal/api/instrumentation.go` of
github.com/wandb/agent-lens). An empty check goes to `no_spans_report` so zero
spans never read as healthy.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pydantic import ValidationError

from wandb_mcp_server.config import structured_error
from wandb_mcp_server.mcp_tools.agent_lens import fetch_agent_lens_json, parse_timestamp
from wandb_mcp_server.mcp_tools.agent_lens_doctor_models import (
    DoctorScope,
    HealthIssue,
    HealthReport,
    HealthSummary,
    IssueSummary,
    Verdict,
)
from wandb_mcp_server.mcp_tools.agent_lens_reach import LATEST_CONVERSATION, newest_agent_span, no_spans_report
from wandb_mcp_server.tool_results import ToolErrorJSON, tool_error_json

HEALTH_PATH = "/instrumentation/health"
HEALTH_PARAMS = frozenset({"start_at", "end_at", "conversation_id", "trace_id"})
TOOL_NAME = "diagnose_instrumentation"
MAX_HEALTH_WINDOW = timedelta(days=7)
PROJECT_LOOKBACK = timedelta(hours=24)
FILTERED_LOOKBACK = MAX_HEALTH_WINDOW
SEVERITY_RANK = {"must": 0, "should": 1}
SHOWN_GROUPS = 3
SHOWN_FAILING_SPANS = 20
SETTLE_MINUTES = 10
DIAGNOSE_INSTRUMENTATION_TOOL_DESCRIPTION = f"""Check a project's agent tracing against Agent Lens instrumentation rules and rank what to fix.

<when_to_use>
Use while instrumenting an agent app with OpenTelemetry for Agent Lens. Loop
without a human reading traces:

1. Run your app or its tests so it emits spans.
2. Call this tool, usually with conversation_id="latest" to judge only your
   newest run. A project-wide call also counts older spans in its window.
3. On verdict "failing", fix issues[0] (must-fix rules rank first), rerun
   your app, and call again.
4. On verdict "settling", some traces are still open or recent.
   Trace rules judge a trace only after {SETTLE_MINUTES} quiet minutes
   (`unsettled_trace_count`); call again after that.
5. Stop on verdict "clean".

Span rules update as soon as spans arrive. Verdict "no_spans" means nothing
was checked. Read its `note` or follow its `diagnosis` and any `checks`.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
conversation_id : str, optional
    Check one conversation's traces. Pass "latest" to use the conversation id
    from the newest agent span in range.
trace_id : str, optional
    Check one trace. Omit this field when using conversation_id="latest".
start_at, end_at : str, optional
    RFC 3339 bounds, supplied together, at most 7 days apart. Default: last
    7 days with a conversation or trace filter, else last 24 hours.
include_failing_spans : bool, optional
    Add up to {SHOWN_FAILING_SPANS} failing span ids per failing rule (default false).

Returns
-------
JSON with "verdict" ("clean", "settling", "failing", or "no_spans"), "scope",
span/trace/unsettled counts, "failing_rule_counts" {{"must", "should"}},
"passing_rule_count", and "issues": only failing rules, ranked, each with
"code", "severity", "rule_type", "title", "impact", "fix", failing/applicable
counts, {SHOWN_GROUPS} "top_failing_groups", and up to 3 "examples"
{{"trace_id", "conversation_id"}}. A "no_spans" result includes a "note" when
no rule applied to the received spans, or a "diagnosis" when the span count is zero.
It includes "calls_in_window" only if no agent spans reached the project in the
window. Ordered exporter "checks" appear only when that count is zero or unknown.
"""


async def diagnose_instrumentation(
    entity_name: str,
    *,
    project_name: str,
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    start_at: Optional[str] = None,
    end_at: Optional[str] = None,
    include_failing_spans: bool = False,
) -> str:
    """Rank the Agent Lens instrumentation rules a project's agent spans fail."""
    for name, value in (("conversation_id", conversation_id), ("trace_id", trace_id)):
        if value is not None and not value.strip():
            error = structured_error("agent_lens_invalid_request", f"{name} must not be blank.")
            return tool_error_json(error)

    if conversation_id == LATEST_CONVERSATION and trace_id is not None:
        error = structured_error("agent_lens_invalid_request", 'Omit trace_id when using conversation_id="latest".')
        return tool_error_json(error)

    try:
        start, end = _health_window(start_at, end_at, filtered=bool(conversation_id or trace_id))
    except ValueError as error:
        return tool_error_json(structured_error("agent_lens_invalid_request", str(error)))

    scope = DoctorScope(entity_name, project_name, _iso(start), _iso(end), conversation_id, trace_id)
    if conversation_id == LATEST_CONVERSATION:
        newest = await newest_agent_span(scope)
        if isinstance(newest, ToolErrorJSON):
            return newest
        if newest is None:
            return await no_spans_report(scope, has_agent_spans=False)
        if not newest.conversation_id:
            return await no_spans_report(scope, has_agent_spans=True)
        scope = replace(scope, conversation_id=newest.conversation_id)

    payload = await fetch_agent_lens_json(
        HEALTH_PATH,
        tool_name=TOOL_NAME,
        method="GET",
        entity_name=entity_name,
        project_name=project_name,
        track_params={"entity_name": entity_name, "project_name": project_name, "filtered": scope.filtered},
        params={key: value for key, value in scope.as_dict().items() if key in HEALTH_PARAMS},
    )
    if isinstance(payload, ToolErrorJSON):
        # Agent Lens answers 404 when a conversation or trace filter matches no eligible span.
        if scope.filtered and json.loads(payload).get("error") == "agent_lens_unavailable":
            return await no_spans_report(scope)
        return payload

    try:
        report = HealthReport.model_validate(payload.get("data"))
    except ValidationError:
        return tool_error_json(
            structured_error("agent_lens_query_failed", "Agent Lens returned an unexpected instrumentation report.")
        )
    if report.span_count == 0:
        return await no_spans_report(scope)

    summary = _summarize(report, scope, include_failing_spans=include_failing_spans)

    return json.dumps(summary)


def _summarize(report: HealthReport, scope: DoctorScope, *, include_failing_spans: bool) -> HealthSummary:
    """Keep only failing rules, musts first and then by failing share, with passing rules as a count."""
    failing = sorted((issue for issue in report.issues if issue.failing_count > 0), key=_issue_rank)
    must_count = sum(1 for issue in failing if issue.severity == "must")
    passing_count = sum(1 for issue in report.issues if issue.applicable_count > 0 and issue.failing_count == 0)

    issues = []
    for issue in failing:
        summary = IssueSummary(
            code=issue.code,
            severity=issue.severity,
            rule_type=issue.rule_type,
            title=issue.title,
            impact=issue.impact,
            fix=issue.fix,
            failing_count=issue.failing_count,
            applicable_count=issue.applicable_count,
            group_type=issue.breakdown.group_type,
            top_failing_groups=[group.model_dump() for group in issue.breakdown.failing_groups[:SHOWN_GROUPS]],
            examples=[example.model_dump() for example in issue.examples],
        )
        if include_failing_spans:
            summary["failing_spans"] = [span.model_dump() for span in issue.failing_spans[:SHOWN_FAILING_SPANS]]
        issues.append(summary)

    result = HealthSummary(
        verdict=_verdict(failing, report.unsettled_trace_count, passing_count),
        scope=scope.as_dict(),
        span_count=report.span_count,
        trace_count=report.trace_count,
        unsettled_trace_count=report.unsettled_trace_count,
        failing_rule_counts={"must": must_count, "should": len(failing) - must_count},
        passing_rule_count=passing_count,
        issues=issues,
    )
    if report.unsettled_trace_count:
        result["note"] = (
            f"{report.unsettled_trace_count} trace(s) are still settling. Trace rules judge a trace only after "
            f"{SETTLE_MINUTES} quiet minutes, so call again before treating trace rules as passing."
        )
    elif result["verdict"] == "no_spans":
        result["note"] = "Agent Lens received the spans, but no instrumentation rule applied, so nothing was checked."

    return result


def _verdict(failing: List[HealthIssue], unsettled_trace_count: int, passing_count: int) -> Verdict:
    """Choose a verdict from failures, unsettled traces, and passing rules."""
    if failing:
        return "failing"

    if unsettled_trace_count:
        return "settling"

    if passing_count == 0:
        return "no_spans"

    return "clean"


def _issue_rank(issue: HealthIssue) -> tuple[int, float, int]:
    """Sort key: must before should, then larger failing share, then larger failing count."""
    share = issue.failing_count / issue.applicable_count

    return SEVERITY_RANK[issue.severity], -share, -issue.failing_count


def _health_window(start_at: Optional[str], end_at: Optional[str], *, filtered: bool) -> tuple[datetime, datetime]:
    """Parse `start_at`/`end_at`, or end now and look back `FILTERED_LOOKBACK` or `PROJECT_LOOKBACK`."""
    if start_at is None and end_at is None:
        end = datetime.now(timezone.utc)
        lookback = FILTERED_LOOKBACK if filtered else PROJECT_LOOKBACK
        return end - lookback, end
    if start_at is None or end_at is None:
        raise ValueError("start_at and end_at must be supplied together")

    start = parse_timestamp(start_at, "start_at")
    end = parse_timestamp(end_at, "end_at")
    if end <= start:
        raise ValueError("start_at and end_at must define a nonempty time range")
    if end - start > MAX_HEALTH_WINDOW:
        raise ValueError("start_at and end_at must be at most 7 days apart")

    return start, end


def _iso(moment: datetime) -> str:
    """Format `moment` as RFC 3339 UTC with a Z suffix."""
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
