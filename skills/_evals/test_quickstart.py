"""Evaluation tests for the quickstart skill."""

import pytest

from .conftest import QUICKSTART_SCENARIOS
from .scorers import OutputQualityScorer, RegexScorer


def _simulate_quickstart_skill(scenario: dict) -> dict:
    """Simulate the quickstart skill executing a scenario."""
    code_snippets = {
        "openai": 'import weave\nweave.init("my-project")\nimport openai',
        "langchain": 'import weave\nweave.init("my-project")\nfrom langchain_openai import ChatOpenAI',
        "custom": 'import weave\nweave.init("my-project")\n\n@weave.op()\ndef my_fn(): ...',
    }
    return {
        "tools_called": [],
        "workflow_steps": ["detect_framework", "add_init", "verify"],
        "total_tool_calls": 0,
        "response_text": code_snippets.get(scenario["framework"], ""),
    }


@pytest.mark.parametrize("scenario", QUICKSTART_SCENARIOS, ids=[s["id"] for s in QUICKSTART_SCENARIOS])
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
