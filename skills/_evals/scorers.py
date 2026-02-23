"""Reusable Weave Scorer classes for evaluating MCP skill execution quality.

Each scorer is a `weave.Scorer` subclass that can be versioned, published,
and used with `weave.Evaluation` for tracked comparisons.

Generic scorers (shared across all skills):
- ToolSelectionScorer: Did the agent pick the right MCP tools?
- WorkflowOrderScorer: Did it follow the prescribed step sequence?
- EfficiencyScorer: How many tool calls were needed?
- OutputQualityScorer: Does the output contain expected substrings?
- RubricScorer: LLM-as-judge evaluation against rubric items.
- RegexScorer: Pattern matching for verifiable outputs.

Custom skill-specific scorers:
- Defined per-skill in scenario dicts under "custom_scorers" key.
- Each custom scorer is a callable(output: dict) -> dict with "passed" bool.
- Enables domain-specific validation (e.g., "code is valid Python",
  "taxonomy has >= 3 categories", "trace count is a real number").

To add a custom scorer for a skill, see the CUSTOM SCORER GUIDE at the
bottom of this file.
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


# ---------------------------------------------------------------------------
# Custom skill-specific scorers
# ---------------------------------------------------------------------------
# These scorers encode domain knowledge unique to a specific skill.
# They are registered in scenario dicts under "custom_scorers": [scorer_instance, ...]
# and picked up by both pytest tests and the run_evals.py orchestrator.
#
# HOW TO ADD A CUSTOM SCORER:
# 1. Define a weave.Scorer subclass below (or in a separate file).
# 2. Add it to the relevant scenarios in conftest.py under "custom_scorers".
# 3. The orchestrator and test files will automatically pick it up.
# ---------------------------------------------------------------------------


class ValidPythonScorer(weave.Scorer):
    """Quickstart-specific: check that the response contains syntactically valid Python.

    Extracts code from markdown fenced blocks (```python ... ```) and runs
    compile() to check for syntax errors. Useful for the quickstart skill
    which must produce runnable instrumentation code.
    """

    @weave.op()
    def score(self, output: dict[str, Any]) -> dict[str, Any]:
        """Check if the response contains valid Python code.

        Args:
            output: Model output containing a "response_text" string.

        Returns:
            Dict with 'passed' bool, 'code_blocks_found', and optional 'error'.
        """
        response = output.get("response_text", "")

        blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if not blocks:
            if "import weave" in response and "weave.init" in response:
                blocks = [response]
            else:
                return {"passed": True, "code_blocks_found": 0, "note": "No code blocks to validate"}

        errors = []
        for i, block in enumerate(blocks):
            try:
                compile(block.strip(), f"<block_{i}>", "exec")
            except SyntaxError as e:
                errors.append({"block": i, "error": str(e), "line": e.lineno})

        return {
            "passed": len(errors) == 0,
            "code_blocks_found": len(blocks),
            "syntax_errors": errors,
        }


class TaxonomyCoverageScorer(weave.Scorer):
    """Failure-analysis-specific: check that the taxonomy has sufficient categories.

    Verifies the response mentions at least `min_categories` distinct error
    categories, which is the core output of the failure-analysis skill.
    """

    min_categories: int = 3

    @weave.op()
    def score(self, output: dict[str, Any]) -> dict[str, Any]:
        """Check taxonomy coverage in the response.

        Args:
            output: Model output containing a "response_text" string.

        Returns:
            Dict with 'passed' bool, 'categories_found', and the category list.
        """
        response = output.get("response_text", "").lower()

        known_categories = [
            "rate_limit", "rate limit", "ratelimit", "429",
            "timeout", "timed out",
            "validation", "validationerror",
            "auth", "authentication", "unauthorized", "403", "401",
            "parsing", "json", "jsondecodeerror",
            "type_error", "typeerror", "attributeerror",
            "context_length", "token limit", "max_tokens",
            "refusal", "i cannot", "content filter",
            "hallucination",
            "empty_output", "empty output", "none",
            "infrastructure", "server error", "500", "503",
        ]

        found = set()
        for cat in known_categories:
            if cat in response:
                normalized = cat.split()[0].replace("_", "").lower()
                found.add(normalized)

        return {
            "passed": len(found) >= self.min_categories,
            "categories_found": len(found),
            "categories": sorted(found),
            "min_required": self.min_categories,
        }


class TraceCountAccuracyScorer(weave.Scorer):
    """Trace-analyst-specific: check that reported trace counts are plausible.

    Verifies the response contains numeric trace counts and they look
    reasonable (not zero, not absurdly high).
    """

    @weave.op()
    def score(self, output: dict[str, Any]) -> dict[str, Any]:
        """Check trace count plausibility.

        Args:
            output: Model output containing a "response_text" string.

        Returns:
            Dict with 'passed' bool and extracted counts.
        """
        response = output.get("response_text", "")

        count_patterns = re.findall(r"(\d[\d,]*)\s*(?:traces|calls|total|error|success)", response, re.IGNORECASE)
        counts = []
        for match in count_patterns:
            try:
                counts.append(int(match.replace(",", "")))
            except ValueError:
                continue

        if not counts:
            return {"passed": False, "counts_found": 0, "note": "No trace counts found in response"}

        all_plausible = all(0 < c < 10_000_000 for c in counts)

        return {
            "passed": all_plausible and len(counts) > 0,
            "counts_found": len(counts),
            "counts": counts,
        }


class MetricComparisonScorer(weave.Scorer):
    """Experiment-analysis-specific: check that metrics are compared across runs.

    Verifies the response contains numeric metric values and mentions
    multiple runs, indicating a proper comparison was performed.
    """

    @weave.op()
    def score(self, output: dict[str, Any]) -> dict[str, Any]:
        """Check metric comparison quality.

        Args:
            output: Model output containing a "response_text" string.

        Returns:
            Dict with 'passed' bool, 'metrics_found', 'runs_mentioned'.
        """
        response = output.get("response_text", "")

        metric_values = re.findall(r"\d+\.\d+", response)
        run_mentions = re.findall(r"(?:run|Run|model)\s*[\w-]+", response)

        return {
            "passed": len(metric_values) >= 2 and len(run_mentions) >= 2,
            "metrics_found": len(metric_values),
            "runs_mentioned": len(run_mentions),
        }


def run_custom_scorers(output: dict[str, Any], custom_scorers: list) -> list[dict]:
    """Run a list of custom scorer instances against an output.

    Args:
        output: The standardized agent output dict.
        custom_scorers: List of weave.Scorer instances.

    Returns:
        List of dicts with 'scorer_name', 'passed', 'details'.
    """
    results = []
    for scorer in custom_scorers:
        name = scorer.__class__.__name__
        try:
            result = scorer.score(output=output)
            results.append({
                "scorer_name": name,
                "passed": result.get("passed", False),
                "details": result,
            })
        except Exception as e:
            results.append({
                "scorer_name": name,
                "passed": False,
                "details": {"error": str(e)},
            })
    return results
