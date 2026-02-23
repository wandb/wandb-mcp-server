"""Evaluation tests for the trace-analyst skill."""

import pytest

from .conftest import TRACE_SCENARIOS
from .scorers import EfficiencyScorer, ToolSelectionScorer, WorkflowOrderScorer


def _simulate_trace_skill(scenario: dict) -> dict:
    """Simulate the trace-analyst skill executing a scenario.

    Replace this with actual MCP client invocation once the skill
    runner is implemented.
    """
    return {
        "tools_called": scenario["expected_tools"],
        "workflow_steps": scenario["expected_workflow"],
        "total_tool_calls": len(scenario["expected_tools"]) + 1,
        "response_text": "Trace analysis complete.",
    }


@pytest.mark.parametrize(
    "scenario",
    TRACE_SCENARIOS,
    ids=[s["id"] for s in TRACE_SCENARIOS],
)
def test_tool_selection(scenario):
    output = _simulate_trace_skill(scenario)
    scorer = ToolSelectionScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["correct"], f"Missing tools: {result['missing_tools']}"


@pytest.mark.parametrize(
    "scenario",
    TRACE_SCENARIOS,
    ids=[s["id"] for s in TRACE_SCENARIOS],
)
def test_workflow_order(scenario):
    output = _simulate_trace_skill(scenario)
    scorer = WorkflowOrderScorer()
    result = scorer.score(output=output, expected_workflow=scenario["expected_workflow"])
    assert result["correct_order"], f"Missed steps: {result['missed_steps']}"


@pytest.mark.parametrize(
    "scenario",
    TRACE_SCENARIOS,
    ids=[s["id"] for s in TRACE_SCENARIOS],
)
def test_efficiency(scenario):
    output = _simulate_trace_skill(scenario)
    scorer = EfficiencyScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["efficient"], f"Too many calls: {result['total_calls']} > {result['max_allowed']}"
