"""Evaluation tests for the failure-analysis skill."""

import pytest

from .conftest import FAILURE_SCENARIOS
from .scorers import (
    EfficiencyScorer,
    RegexScorer,
    RubricScorer,
    ToolSelectionScorer,
    WorkflowOrderScorer,
)

RESPONSE_TEMPLATES = {
    "rate-limit-cluster": "Found 43 error traces. Clusters: RateLimitError (23, 429 status), TimeoutError (12), ValidationError (8).",
    "taxonomy-generation": "Generated failure taxonomy with 6 categories. Created Scorer class AxialCodingClassifierV1 for classification.",
}


def _simulate_failure_skill(scenario: dict) -> dict:
    """Simulate the failure-analysis skill executing a scenario."""
    return {
        "tools_called": scenario["expected_tools"],
        "workflow_steps": scenario["expected_workflow"],
        "total_tool_calls": len(scenario["expected_tools"]) + 2,
        "response_text": RESPONSE_TEMPLATES.get(scenario["id"], "Failure analysis complete."),
    }


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS, ids=[s["id"] for s in FAILURE_SCENARIOS])
def test_tool_selection(scenario):
    output = _simulate_failure_skill(scenario)
    scorer = ToolSelectionScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["correct"], f"Missing tools: {result['missing_tools']}"


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS, ids=[s["id"] for s in FAILURE_SCENARIOS])
def test_workflow_order(scenario):
    output = _simulate_failure_skill(scenario)
    scorer = WorkflowOrderScorer()
    result = scorer.score(output=output, expected_workflow=scenario["expected_workflow"])
    assert result["correct_order"], f"Missed steps: {result['missed_steps']}"


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS, ids=[s["id"] for s in FAILURE_SCENARIOS])
def test_efficiency(scenario):
    output = _simulate_failure_skill(scenario)
    scorer = EfficiencyScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["efficient"], f"Too many calls: {result['total_calls']} > {result['max_allowed']}"


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS, ids=[s["id"] for s in FAILURE_SCENARIOS])
def test_regex_checks(scenario):
    output = _simulate_failure_skill(scenario)
    scorer = RegexScorer()
    result = scorer.score(output=output, regex_checks=scenario["regex_checks"])
    assert result["all_passed"], f"Failed checks: {[k for k, v in result['checks'].items() if not v]}"


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS, ids=[s["id"] for s in FAILURE_SCENARIOS])
def test_rubric(scenario):
    output = _simulate_failure_skill(scenario)
    scorer = RubricScorer(dry_run=True)
    result = scorer.score(output=output, rubric=scenario["rubric"])
    assert result["all_passed"]
