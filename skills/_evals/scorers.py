"""Reusable Weave Scorer classes for evaluating MCP skill execution quality.

Each scorer is a `weave.Scorer` subclass that can be versioned, published,
and used with `weave.Evaluation` for tracked comparisons.
"""

from typing import Any

import weave


class ToolSelectionScorer(weave.Scorer):
    """Score whether the agent selected the correct MCP tools for a given task.

    Checks that the predicted tool calls are a superset of the expected tools.
    """

    @weave.op()
    def score(self, output: dict[str, Any], expected_tools: list[str]) -> dict[str, Any]:
        """Score tool selection accuracy.

        Args:
            output: Model output containing a "tools_called" list.
            expected_tools: List of tool names that should have been called.

        Returns:
            Dict with 'correct' bool, 'precision', 'recall', and 'missing_tools'.
        """
        tools_called = set(output.get("tools_called", []))
        expected = set(expected_tools)

        hits = tools_called & expected
        missing = expected - tools_called
        extra = tools_called - expected

        precision = len(hits) / len(tools_called) if tools_called else 0.0
        recall = len(hits) / len(expected) if expected else 1.0

        return {
            "correct": missing == set(),
            "precision": precision,
            "recall": recall,
            "missing_tools": list(missing),
            "extra_tools": list(extra),
        }


class WorkflowOrderScorer(weave.Scorer):
    """Score whether the agent followed the prescribed workflow step order.

    Checks that the observed workflow steps are in the correct relative order,
    allowing for extra intermediate steps.
    """

    @weave.op()
    def score(self, output: dict[str, Any], expected_workflow: list[str]) -> dict[str, Any]:
        """Score workflow ordering.

        Args:
            output: Model output containing a "workflow_steps" list.
            expected_workflow: Ordered list of expected workflow step names.

        Returns:
            Dict with 'correct_order' bool, 'steps_completed' ratio, and details.
        """
        actual_steps = output.get("workflow_steps", [])

        completed = []
        idx = 0
        for expected_step in expected_workflow:
            for i in range(idx, len(actual_steps)):
                if actual_steps[i] == expected_step:
                    completed.append(expected_step)
                    idx = i + 1
                    break

        steps_ratio = len(completed) / len(expected_workflow) if expected_workflow else 1.0

        return {
            "correct_order": completed == expected_workflow,
            "steps_completed": steps_ratio,
            "completed_steps": completed,
            "missed_steps": [s for s in expected_workflow if s not in completed],
        }


class EfficiencyScorer(weave.Scorer):
    """Score the efficiency of skill execution by tool call count.

    Penalizes excessive tool calls (>2x expected) and rewards minimal calls.
    """

    max_calls_multiplier: float = 2.0

    @weave.op()
    def score(self, output: dict[str, Any], expected_tools: list[str]) -> dict[str, Any]:
        """Score execution efficiency.

        Args:
            output: Model output containing "total_tool_calls" count.
            expected_tools: Expected tool list (length = minimum calls).

        Returns:
            Dict with 'efficient' bool and 'call_ratio'.
        """
        total_calls = output.get("total_tool_calls", 0)
        min_calls = len(expected_tools)
        max_calls = int(min_calls * self.max_calls_multiplier)

        return {
            "efficient": total_calls <= max_calls,
            "call_ratio": total_calls / min_calls if min_calls > 0 else 0.0,
            "total_calls": total_calls,
            "min_expected": min_calls,
            "max_allowed": max_calls,
        }


class OutputQualityScorer(weave.Scorer):
    """Score output quality by checking for expected content strings.

    Simple substring-match scorer. For production use, replace with
    an LLM-as-judge scorer.
    """

    @weave.op()
    def score(self, output: dict[str, Any], expected_output_contains: list[str]) -> dict[str, Any]:
        """Score output quality via substring matching.

        Args:
            output: Model output containing a "response_text" string.
            expected_output_contains: Substrings that should appear in the response.

        Returns:
            Dict with 'contains_all' bool and per-substring results.
        """
        response = output.get("response_text", "")

        results = {}
        for substring in expected_output_contains:
            results[substring] = substring in response

        return {
            "contains_all": all(results.values()),
            "match_ratio": sum(results.values()) / len(results) if results else 1.0,
            "matches": results,
        }
