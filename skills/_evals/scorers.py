"""Reusable Weave Scorer classes for evaluating MCP skill execution quality.

Each scorer is a `weave.Scorer` subclass that can be versioned, published,
and used with `weave.Evaluation` for tracked comparisons.

Scorer types:
- ToolSelectionScorer: Did the agent pick the right MCP tools?
- WorkflowOrderScorer: Did it follow the prescribed step sequence?
- EfficiencyScorer: How many tool calls were needed?
- OutputQualityScorer: Does the output contain expected substrings?
- RubricScorer: LLM-as-judge evaluation against rubric items.
- RegexScorer: Pattern matching for verifiable outputs.
"""

import re
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


class RegexScorer(weave.Scorer):
    """Score output by checking regex patterns against the response text.

    Each check has an id, pattern, and optional description. Inspired by
    the improver repo's `core.scorers.regex::RegexScorer` which validates
    task outputs against expected numerical ranges and structural patterns.
    """

    @weave.op()
    def score(self, output: dict[str, Any], regex_checks: list[dict[str, str]]) -> dict[str, Any]:
        """Score output against a list of regex patterns.

        Args:
            output: Model output containing a "response_text" string.
            regex_checks: List of dicts, each with 'id', 'pattern', and optional 'description'.

        Returns:
            Dict with 'all_passed' bool, 'pass_ratio', and per-check results.
        """
        response = output.get("response_text", "")

        results = {}
        for check in regex_checks:
            check_id = check["id"]
            pattern = check["pattern"]
            results[check_id] = bool(re.search(pattern, response))

        passed = sum(results.values())
        total = len(results)

        return {
            "all_passed": passed == total,
            "pass_ratio": passed / total if total > 0 else 1.0,
            "checks": results,
        }


class RubricScorer(weave.Scorer):
    """Score output by evaluating against rubric items using an LLM judge.

    Each rubric item has an id and text description of what to check.
    The scorer formats them into a prompt and asks an LLM to judge each.
    Inspired by the improver repo's `core.scorers.rubric::RubricScorer`.

    For unit testing, set `dry_run=True` to skip the LLM call and return
    all items as passed.
    """

    dry_run: bool = False

    @weave.op()
    def score(self, output: dict[str, Any], rubric: list[dict[str, str]]) -> dict[str, Any]:
        """Score output against rubric items.

        Args:
            output: Model output containing a "response_text" string.
            rubric: List of dicts, each with 'id' and 'text' describing the criterion.

        Returns:
            Dict with 'all_passed' bool, 'pass_ratio', and per-item verdicts.
        """
        response = output.get("response_text", "")

        if self.dry_run or not response:
            items = {item["id"]: True for item in rubric}
            return {
                "all_passed": True,
                "pass_ratio": 1.0,
                "items": items,
            }

        try:
            import json

            from openai import OpenAI

            rubric_text = "\n".join(
                f"- {item['id']}: {item['text']}" for item in rubric
            )

            client = OpenAI()
            llm_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluator. Given a response and rubric items, "
                            "judge each item as passed or failed. Return ONLY valid JSON: "
                            '{"items": [{"id": "...", "verdict": "pass"|"fail", "notes": "..."}]}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Response:\n{response}\n\nRubric:\n{rubric_text}",
                    },
                ],
                temperature=0,
            )

            text = llm_response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)

            items = {}
            for item in result.get("items", []):
                items[item["id"]] = item.get("verdict", "fail") == "pass"

            passed = sum(items.values())
            total = len(items)

            return {
                "all_passed": passed == total,
                "pass_ratio": passed / total if total > 0 else 1.0,
                "items": items,
            }

        except Exception as e:
            return {
                "all_passed": False,
                "pass_ratio": 0.0,
                "items": {},
                "error": str(e),
            }
