"""Evaluation tests for the experiment-analysis skill."""

import pytest

from .conftest import EXPERIMENT_SCENARIOS
from .scorers import (
    EfficiencyScorer,
    RegexScorer,
    RubricScorer,
    ToolSelectionScorer,
    WorkflowOrderScorer,
    run_custom_scorers,
)

RESPONSE_TEMPLATES = {
    "compare-top-runs": "Compared top 5 runs by eval_loss. Run A: 0.12, Run B: 0.15, Run C: 0.18.",
    "best-run": "Run 'finetune-v3' had the best accuracy at 94.2%.",
    "create-report": "Report created: https://wandb.ai/team/project/reports/Comparison--abc123",
}


def _simulate_experiment_skill(scenario: dict) -> dict:
    """Simulate the experiment-analysis skill executing a scenario."""
    return {
        "tools_called": scenario["expected_tools"],
        "workflow_steps": scenario["expected_workflow"],
        "total_tool_calls": len(scenario["expected_tools"]),
        "response_text": RESPONSE_TEMPLATES.get(scenario["id"], "Analysis complete."),
    }


@pytest.mark.parametrize("scenario", EXPERIMENT_SCENARIOS, ids=[s["id"] for s in EXPERIMENT_SCENARIOS])
def test_tool_selection(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = ToolSelectionScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["correct"], f"Missing tools: {result['missing_tools']}"


@pytest.mark.parametrize("scenario", EXPERIMENT_SCENARIOS, ids=[s["id"] for s in EXPERIMENT_SCENARIOS])
def test_workflow_order(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = WorkflowOrderScorer()
    result = scorer.score(output=output, expected_workflow=scenario["expected_workflow"])
    assert result["correct_order"], f"Missed steps: {result['missed_steps']}"


@pytest.mark.parametrize("scenario", EXPERIMENT_SCENARIOS, ids=[s["id"] for s in EXPERIMENT_SCENARIOS])
def test_efficiency(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = EfficiencyScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["efficient"], f"Too many calls: {result['total_calls']} > {result['max_allowed']}"


@pytest.mark.parametrize("scenario", EXPERIMENT_SCENARIOS, ids=[s["id"] for s in EXPERIMENT_SCENARIOS])
def test_regex_checks(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = RegexScorer()
    result = scorer.score(output=output, regex_checks=scenario["regex_checks"])
    assert result["all_passed"], f"Failed checks: {[k for k, v in result['checks'].items() if not v]}"


@pytest.mark.parametrize("scenario", EXPERIMENT_SCENARIOS, ids=[s["id"] for s in EXPERIMENT_SCENARIOS])
def test_rubric(scenario):
    output = _simulate_experiment_skill(scenario)
    scorer = RubricScorer(dry_run=True)
    result = scorer.score(output=output, rubric=scenario["rubric"])
    assert result["all_passed"]


CUSTOM_SCORER_SCENARIOS = [s for s in EXPERIMENT_SCENARIOS if "custom_scorers" in s]


@pytest.mark.parametrize("scenario", CUSTOM_SCORER_SCENARIOS, ids=[s["id"] for s in CUSTOM_SCORER_SCENARIOS])
def test_custom_scorers(scenario):
    output = _simulate_experiment_skill(scenario)
    results = run_custom_scorers(output, scenario["custom_scorers"])
    for r in results:
        assert r["passed"], f"Custom scorer {r['scorer_name']} failed: {r['details']}"
