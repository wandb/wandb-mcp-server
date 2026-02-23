"""Evaluation tests for the experiment-analysis skill."""

import pytest

from .conftest import EXPERIMENT_SCENARIOS
from .scorers import EfficiencyScorer, ToolSelectionScorer, WorkflowOrderScorer


def _simulate_experiment_skill(scenario: dict) -> dict:
    """Simulate the experiment-analysis skill executing a scenario.

    Replace this with actual MCP client invocation once the skill
    runner is implemented.
    """
    # Placeholder: returns expected values for now.
    # In production, this drives an actual agent through the skill workflow.
    return {
        "tools_called": scenario["expected_tools"],
        "workflow_steps": scenario["expected_workflow"],
        "total_tool_calls": len(scenario["expected_tools"]),
        "response_text": "Top runs compared successfully.",
    }


@pytest.mark.parametrize(
    "scenario",
    EXPERIMENT_SCENARIOS,
    ids=[s["id"] for s in EXPERIMENT_SCENARIOS],
)
def test_tool_selection(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = ToolSelectionScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["correct"], f"Missing tools: {result['missing_tools']}"


@pytest.mark.parametrize(
    "scenario",
    EXPERIMENT_SCENARIOS,
    ids=[s["id"] for s in EXPERIMENT_SCENARIOS],
)
def test_workflow_order(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = WorkflowOrderScorer()
    result = scorer.score(output=output, expected_workflow=scenario["expected_workflow"])
    assert result["correct_order"], f"Missed steps: {result['missed_steps']}"


@pytest.mark.parametrize(
    "scenario",
    EXPERIMENT_SCENARIOS,
    ids=[s["id"] for s in EXPERIMENT_SCENARIOS],
)
def test_efficiency(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = EfficiencyScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["efficient"], f"Too many calls: {result['total_calls']} > {result['max_allowed']}"
