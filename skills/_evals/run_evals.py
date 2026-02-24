"""CLI orchestrator for MCP skill evaluations.

Runs skill scenarios against real coding agents (Claude Code, Codex) via
their CLIs, scores the results with Weave Scorers, and displays results
in the terminal or a Textual TUI.

Usage:
    # Run quickstart evals with Claude Code (default profile)
    python -m skills._evals.run_evals --skill quickstart --runner claude

    # Run hackathon-flavored evals for Mistral Worldwide Hackathon
    python -m skills._evals.run_evals --skill all --runner claude --profile hackathon

    # Seed hackathon data and run with TUI
    python -m skills._evals.run_evals --skill quickstart --runner mock --profile hackathon --seed --tui

    # Dry run (mock agent, no CLI needed)
    python -m skills._evals.run_evals --skill quickstart --runner mock
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

EVAL_ENTITY = os.environ.get("MCP_LOGS_WANDB_ENTITY", "a-sh0ts")
EVAL_PROJECT = os.environ.get("MCP_EVAL_PROJECT", "mcp-skill-evals")

AVAILABLE_SKILLS = ["quickstart", "trace", "experiment", "failure"]
AVAILABLE_PROFILES = ["default", "hackathon"]


@dataclass
class ScoreResult:
    """Result from a single scorer on a single scenario."""

    scorer_name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    """Complete evaluation record for one scenario + runner combination."""

    scenario_id: str
    skill: str
    runner_name: str
    user_request: str
    agent_response: str
    tools_called: list[str]
    workflow_steps: list[str]
    total_tool_calls: int
    duration_ms: int
    exit_code: int
    error: str | None
    scores: list[ScoreResult] = field(default_factory=list)
    passed: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


def load_scenarios(skill: str, profile: str = "default") -> list[dict]:
    """Load scenarios for a given skill and profile.

    Args:
        skill: Skill name or "all".
        profile: Profile name ("default" or "hackathon").

    Returns:
        List of scenario dicts with "_skill" key injected.
    """
    from skills._evals.conftest import PROFILES

    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Choose from: {list(PROFILES.keys())}")

    mapping = PROFILES[profile]

    if skill == "all":
        all_scenarios = []
        for name, scenarios in mapping.items():
            for s in scenarios:
                s["_skill"] = name
                all_scenarios.append(s)
        return all_scenarios

    scenarios = mapping.get(skill)
    if not scenarios:
        raise ValueError(f"Unknown skill: {skill}. Choose from: {list(mapping.keys())}")
    for s in scenarios:
        s["_skill"] = skill
    return scenarios


def get_runner(runner_name: str):
    """Get an agent runner by name."""
    if runner_name == "claude":
        from skills._evals.runners.claude_runner import ClaudeRunner
        return ClaudeRunner()
    elif runner_name == "codex":
        from skills._evals.runners.codex_runner import CodexRunner
        return CodexRunner()
    elif runner_name == "mock":
        return MockRunner()
    else:
        raise ValueError(f"Unknown runner: {runner_name}. Choose from: claude, codex, mock")


class MockRunner:
    """Mock runner for testing the eval pipeline without a real agent CLI."""

    name = "mock"

    def run(self, prompt: str, skill_name: str, timeout: int = 120):
        from skills._evals.runners.base import AgentResult

        time.sleep(0.1)

        mock_responses = {
            "quickstart": 'import weave\nweave.init("my-project")\n\n@weave.op()\ndef my_fn(): ...',
            "trace": "Found 15,234 total traces. 12,100 successful, 3,134 errors (20.6% error rate).",
            "experiment": "Compared top 5 runs by eval_loss. Run A: 0.12, Run B: 0.15.",
            "failure": "Found 43 error traces. Clusters: RateLimitError (23), TimeoutError (12).",
        }
        mock_tools = {
            "quickstart": [],
            "trace": ["count_weave_traces_tool", "query_weave_traces_tool"],
            "experiment": ["query_wandb_tool"],
            "failure": ["count_weave_traces_tool", "query_weave_traces_tool"],
        }

        return AgentResult(
            response_text=mock_responses.get(skill_name, "Mock response"),
            tools_called=mock_tools.get(skill_name, []),
            workflow_steps=["mock_step"],
            total_tool_calls=len(mock_tools.get(skill_name, [])),
            duration_ms=100,
            exit_code=0,
        )


def score_result(scenario: dict, agent_result) -> list[ScoreResult]:
    """Score an agent result against all applicable scorers for a scenario."""
    from skills._evals.runners.base import AgentResult
    from skills._evals.scorers import (
        EfficiencyScorer,
        OutputQualityScorer,
        RegexScorer,
        RubricScorer,
        ToolSelectionScorer,
        WorkflowOrderScorer,
    )

    output = {
        "response_text": agent_result.response_text,
        "tools_called": agent_result.tools_called,
        "workflow_steps": agent_result.workflow_steps,
        "total_tool_calls": agent_result.total_tool_calls,
    }

    scores = []

    if "regex_checks" in scenario:
        scorer = RegexScorer()
        result = scorer.score(output=output, regex_checks=scenario["regex_checks"])
        scores.append(ScoreResult(
            scorer_name="regex",
            passed=result["all_passed"],
            details=result,
        ))

    if "rubric" in scenario:
        scorer = RubricScorer(dry_run=True)
        result = scorer.score(output=output, rubric=scenario["rubric"])
        scores.append(ScoreResult(
            scorer_name="rubric",
            passed=result["all_passed"],
            details=result,
        ))

    if "expected_tools" in scenario:
        scorer = ToolSelectionScorer()
        result = scorer.score(output=output, expected_tools=scenario["expected_tools"])
        scores.append(ScoreResult(
            scorer_name="tool_selection",
            passed=result["correct"],
            details=result,
        ))

        eff_scorer = EfficiencyScorer()
        eff_result = eff_scorer.score(output=output, expected_tools=scenario["expected_tools"])
        scores.append(ScoreResult(
            scorer_name="efficiency",
            passed=eff_result["efficient"],
            details=eff_result,
        ))

    if "expected_workflow" in scenario:
        scorer = WorkflowOrderScorer()
        result = scorer.score(output=output, expected_workflow=scenario["expected_workflow"])
        scores.append(ScoreResult(
            scorer_name="workflow_order",
            passed=result["correct_order"],
            details=result,
        ))

    if "expected_output_contains" in scenario:
        scorer = OutputQualityScorer()
        result = scorer.score(output=output, expected_output_contains=scenario["expected_output_contains"])
        scores.append(ScoreResult(
            scorer_name="output_quality",
            passed=result["contains_all"],
            details=result,
        ))

    if "custom_scorers" in scenario:
        from skills._evals.scorers import run_custom_scorers

        custom_results = run_custom_scorers(output, scenario["custom_scorers"])
        for cr in custom_results:
            scores.append(ScoreResult(
                scorer_name=cr["scorer_name"],
                passed=cr["passed"],
                details=cr["details"],
            ))

    return scores


def run_eval_batch(
    scenarios: list[dict],
    runner,
    skill: str,
    timeout: int = 120,
    on_record: callable = None,
) -> list[EvalRecord]:
    """Run all scenarios through a runner and score results.

    Args:
        scenarios: List of scenario dicts from conftest.py.
        runner: An AgentRunner instance.
        skill: Skill name being evaluated.
        timeout: Per-scenario timeout in seconds.
        on_record: Callback(record) called after each scenario completes.

    Returns:
        List of EvalRecord results.
    """
    records = []

    for scenario in scenarios:
        scenario_skill = scenario.get("_skill", skill)
        scenario_id = scenario["id"]

        if on_record:
            on_record(EvalRecord(
                scenario_id=scenario_id,
                skill=scenario_skill,
                runner_name=runner.name,
                user_request=scenario["user_request"],
                agent_response="",
                tools_called=[],
                workflow_steps=[],
                total_tool_calls=0,
                duration_ms=0,
                exit_code=-1,
                error="running...",
            ))

        agent_result = runner.run(
            prompt=scenario["user_request"],
            skill_name=scenario_skill,
            timeout=timeout,
        )

        scores = score_result(scenario, agent_result)
        all_passed = all(s.passed for s in scores) if scores else False

        record = EvalRecord(
            scenario_id=scenario_id,
            skill=scenario_skill,
            runner_name=runner.name,
            user_request=scenario["user_request"],
            agent_response=agent_result.response_text,
            tools_called=agent_result.tools_called,
            workflow_steps=agent_result.workflow_steps,
            total_tool_calls=agent_result.total_tool_calls,
            duration_ms=agent_result.duration_ms,
            exit_code=agent_result.exit_code,
            error=agent_result.error,
            scores=scores,
            passed=all_passed,
        )

        records.append(record)
        if on_record:
            on_record(record)

    return records


def print_results_table(records: list[EvalRecord]):
    """Print eval results as a rich table to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Skill Eval Results", show_lines=True)

        table.add_column("Scenario", style="cyan")
        table.add_column("Skill", style="blue")
        table.add_column("Runner", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Duration", justify="right")
        table.add_column("Tools", style="dim")
        table.add_column("Scores", style="dim")

        for r in records:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            if r.error and r.error != "running...":
                status = f"[red]ERROR: {r.error[:30]}[/red]"

            score_summary = ", ".join(
                f"{'✓' if s.passed else '✗'}{s.scorer_name}"
                for s in r.scores
            )

            table.add_row(
                r.scenario_id,
                r.skill,
                r.runner_name,
                status,
                f"{r.duration_ms}ms",
                ", ".join(r.tools_called) or "-",
                score_summary or "-",
            )

        console.print(table)

        passed = sum(1 for r in records if r.passed)
        total = len(records)
        console.print(f"\n[bold]{passed}/{total} passed[/bold]")

    except ImportError:
        print("\n=== Eval Results ===")
        for r in records:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {r.scenario_id} [{r.runner_name}]: {status} ({r.duration_ms}ms)")
        passed = sum(1 for r in records if r.passed)
        print(f"\n{passed}/{len(records)} passed")


def log_to_weave(records: list[EvalRecord]):
    """Log eval records to Weave for tracking across versions."""
    try:
        import weave

        weave.init(f"{EVAL_ENTITY}/{EVAL_PROJECT}")

        for record in records:
            record_dict = {
                "scenario_id": record.scenario_id,
                "skill": record.skill,
                "runner": record.runner_name,
                "passed": record.passed,
                "duration_ms": record.duration_ms,
                "tools_called": record.tools_called,
                "total_tool_calls": record.total_tool_calls,
                "scores": {s.scorer_name: s.passed for s in record.scores},
                "error": record.error,
            }

        weave.finish()
    except Exception as e:
        print(f"Warning: Could not log to Weave: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run MCP skill evaluations")
    parser.add_argument("--skill", default="quickstart",
                        choices=["quickstart", "trace", "experiment", "failure", "all"],
                        help="Skill to evaluate (default: quickstart)")
    parser.add_argument("--runner", default="mock",
                        choices=["claude", "codex", "mock", "all"],
                        help="Agent runner to use (default: mock)")
    parser.add_argument("--profile", default="default",
                        choices=AVAILABLE_PROFILES,
                        help="Scenario profile (default: default, hackathon: Mistral hackathon)")
    parser.add_argument("--seed", action="store_true",
                        help="Seed the eval project before running")
    parser.add_argument("--tui", action="store_true",
                        help="Use Textual TUI for interactive display")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-scenario timeout in seconds (default: 120)")
    parser.add_argument("--weave-log", action="store_true",
                        help="Log results to Weave")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Write results to a JSON file")

    args = parser.parse_args()

    if args.seed:
        import subprocess
        seed_cmd = [sys.executable, "-m", "skills._evals.seed_project", "--profile", args.profile]
        subprocess.run(seed_cmd, check=True)
        print()

    scenarios = load_scenarios(args.skill, profile=args.profile)
    print(f"Loaded {len(scenarios)} scenarios for skill={args.skill} profile={args.profile}")

    runners = []
    if args.runner == "all":
        for name in ["claude", "codex"]:
            try:
                runners.append(get_runner(name))
            except RuntimeError as e:
                print(f"Warning: {name} runner unavailable: {e}")
    else:
        runners.append(get_runner(args.runner))

    if args.tui:
        try:
            from skills._evals.tui import run_tui
            run_tui(scenarios, runners, timeout=args.timeout)
            return
        except ImportError:
            print("Warning: textual not installed, falling back to table output")
            print("Install with: uv pip install 'textual>=3.0.0'")

    all_records = []
    for runner in runners:
        print(f"\nRunning {len(scenarios)} scenarios with {runner.name}...")
        records = run_eval_batch(
            scenarios, runner, args.skill, timeout=args.timeout,
        )
        all_records.extend(records)

    print_results_table(all_records)

    if args.weave_log:
        log_to_weave(all_records)

    if args.json_output:
        output = []
        for r in all_records:
            d = {
                "scenario_id": r.scenario_id,
                "skill": r.skill,
                "runner": r.runner_name,
                "passed": r.passed,
                "duration_ms": r.duration_ms,
                "exit_code": r.exit_code,
                "error": r.error,
                "tools_called": r.tools_called,
                "scores": [{
                    "name": s.scorer_name,
                    "passed": s.passed,
                } for s in r.scores],
            }
            output.append(d)

        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.json_output}")

    passed = sum(1 for r in all_records if r.passed)
    sys.exit(0 if passed == len(all_records) else 1)


if __name__ == "__main__":
    main()
