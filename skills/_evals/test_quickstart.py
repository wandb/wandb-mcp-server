"""Evaluation tests for the quickstart skill.

Tests both code-generation scenarios (mock) and live verification scenarios
that reference the seed project. Live scenarios simulate an agent that uses
MCP tools to verify traces exist.
"""

import pytest

from .conftest import EVAL_SEED_ENTITY, EVAL_SEED_PROJECT, QUICKSTART_SCENARIOS
from .scorers import OutputQualityScorer, RegexScorer, RubricScorer, ToolSelectionScorer

MOCK_RESPONSES = {
    "openai-app": 'import weave\nweave.init("my-project")\nimport openai',
    "langchain-app": 'import weave\nweave.init("my-project")\nfrom langchain_openai import ChatOpenAI',
    "custom-app": 'import weave\nweave.init("my-project")\n\n@weave.op()\ndef my_fn(): ...',
    "verify-traces-live": (
        f"I checked your project {EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT} and found "
        f"20 total traces. You can view them at: "
        f"https://wandb.ai/{EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}/weave/traces"
    ),
    "instrument-and-verify-live": (
        f"Here's how to add Weave tracing:\n\n"
        f"```python\nimport weave\nweave.init(\"{EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}\")\n```\n\n"
        f"I verified traces exist -- found 20 traces in {EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}."
    ),
}


def _simulate_quickstart_skill(scenario: dict) -> dict:
    """Simulate the quickstart skill executing a scenario.

    For code-gen scenarios: returns preset code snippets.
    For live scenarios: returns mock MCP tool responses.
    """
    scenario_id = scenario["id"]
    response = MOCK_RESPONSES.get(scenario_id, "")

    tools = scenario.get("expected_tools", [])
    workflow = scenario.get("expected_workflow", ["detect_framework", "add_init", "verify"])

    return {
        "tools_called": tools,
        "workflow_steps": workflow,
        "total_tool_calls": len(tools),
        "response_text": response,
    }


CODE_GEN_SCENARIOS = [s for s in QUICKSTART_SCENARIOS if "expected_output_contains" in s]
LIVE_SCENARIOS = [s for s in QUICKSTART_SCENARIOS if "expected_tools" in s]


@pytest.mark.parametrize("scenario", CODE_GEN_SCENARIOS, ids=[s["id"] for s in CODE_GEN_SCENARIOS])
def test_output_contains_expected(scenario):
    output = _simulate_quickstart_skill(scenario)
    scorer = OutputQualityScorer()
    result = scorer.score(
        output=output,
        expected_output_contains=scenario["expected_output_contains"],
    )
    assert result["contains_all"], f"Missing content: {[k for k, v in result['matches'].items() if not v]}"


@pytest.mark.parametrize("scenario", QUICKSTART_SCENARIOS, ids=[s["id"] for s in QUICKSTART_SCENARIOS])
def test_regex_checks(scenario):
    output = _simulate_quickstart_skill(scenario)
    scorer = RegexScorer()
    result = scorer.score(output=output, regex_checks=scenario["regex_checks"])
    assert result["all_passed"], f"Failed checks: {[k for k, v in result['checks'].items() if not v]}"


@pytest.mark.parametrize("scenario", LIVE_SCENARIOS, ids=[s["id"] for s in LIVE_SCENARIOS])
def test_tool_selection(scenario):
    output = _simulate_quickstart_skill(scenario)
    scorer = ToolSelectionScorer()
    result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
    assert result["correct"], f"Missing tools: {result['missing_tools']}"


@pytest.mark.parametrize("scenario", LIVE_SCENARIOS, ids=[s["id"] for s in LIVE_SCENARIOS])
def test_rubric(scenario):
    output = _simulate_quickstart_skill(scenario)
    scorer = RubricScorer(dry_run=True)
    result = scorer.score(output=output, rubric=scenario["rubric"])
    assert result["all_passed"]
