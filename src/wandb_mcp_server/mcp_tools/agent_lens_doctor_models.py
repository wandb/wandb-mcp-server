"""Response shapes for `diagnose_agent_lens_instrumentation_tool`.

The pydantic models parse `GET /instrumentation/health` from
`internal/api/instrumentation_response.go` of github.com/wandb/agent-lens; the
TypedDicts are what the tool returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel

Verdict = Literal["clean", "settling", "failing", "no_spans"]
Severity = Literal["must", "should"]


class HealthGroup(BaseModel):
    name: str
    failing_count: int
    applicable_count: int


class HealthBreakdown(BaseModel):
    group_type: str
    failing_groups: List[HealthGroup]


class HealthExample(BaseModel):
    trace_id: str
    conversation_id: str


class HealthFailingSpan(BaseModel):
    trace_id: str
    span_id: str
    conversation_id: str


class HealthIssue(BaseModel):
    code: str
    severity: Severity
    rule_type: Literal["span", "trace"]
    title: str
    impact: str
    fix: str
    applicable_count: int
    failing_count: int
    examples: List[HealthExample]
    failing_spans: List[HealthFailingSpan]
    breakdown: HealthBreakdown


class HealthReport(BaseModel):
    span_count: int
    trace_count: int
    unsettled_trace_count: int
    issues: List[HealthIssue]


class AgentSpan(BaseModel):
    conversation_id: str = ""


class AgentSpanPage(BaseModel):
    spans: List[AgentSpan]


class IssueSummary(TypedDict):
    code: str
    severity: Severity
    rule_type: str
    title: str
    impact: str
    fix: str
    failing_count: int
    applicable_count: int
    group_type: str
    top_failing_groups: List[Dict[str, object]]
    examples: List[Dict[str, object]]
    failing_spans: NotRequired[List[Dict[str, object]]]


class HealthSummary(TypedDict):
    verdict: Verdict
    scope: Dict[str, str]
    span_count: int
    trace_count: int
    unsettled_trace_count: int
    failing_rule_counts: Dict[Severity, int]
    passing_rule_count: int
    issues: List[IssueSummary]
    note: NotRequired[str]


class NoSpansReport(TypedDict):
    verdict: Verdict
    scope: Dict[str, str]
    span_count: int
    diagnosis: str
    passing_rule_count: NotRequired[int]
    calls_in_window: NotRequired[Optional[int]]
    checks: NotRequired[List[str]]


@dataclass(frozen=True)
class DoctorScope:
    entity_name: str
    project_name: str
    start_at: str
    end_at: str
    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None

    @property
    def filtered(self) -> bool:
        return bool(self.conversation_id or self.trace_id)

    def as_dict(self) -> Dict[str, str]:
        """Return set fields, used for both health query parameters and the reported scope."""
        fields = {
            "entity_name": self.entity_name,
            "project_name": self.project_name,
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "start_at": self.start_at,
            "end_at": self.end_at,
        }

        return {key: value for key, value in fields.items() if value is not None}
