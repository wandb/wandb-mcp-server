# Skills Evaluation Framework

Python-native evaluation framework for MCP skills, built on Weave's `Evaluation` and `Scorer` classes. Supports both mock unit tests (pytest) and live agent evals (Claude Code CLI, Codex CLI) with a Textual TUI.

## Architecture

```
_evals/
  conftest.py          # Shared fixtures, scenarios, env var config
  scorers.py           # Reusable Weave Scorer classes (6 scorers)
  seed_project.py      # Seeds W&B project with sample runs + Weave traces
  run_evals.py         # CLI orchestrator (scenarios -> runners -> scorers -> display)
  tui.py               # Textual TUI for interactive eval debugging
  run.sh               # Bash convenience script
  runners/
    base.py            # AgentRunner ABC, AgentResult dataclass, parsers
    claude_runner.py   # Claude Code CLI runner (claude -p --mcp-config)
    codex_runner.py    # Codex CLI runner (codex exec --full-auto)
  test_experiment.py   # Experiment-analysis skill evals
  test_trace.py        # Trace-analyst skill evals
  test_quickstart.py   # Quickstart skill evals (code-gen + live verification)
  test_failure.py      # Failure-analysis skill evals
```

## Quick Start

### Mock evals (no API keys needed)

```bash
# Run all skill evals with mock agent
pytest skills/_evals/ -v

# Run quickstart evals with TUI
./skills/_evals/run.sh quickstart mock
```

### Live evals (requires API keys)

```bash
# Set API keys
export WANDB_API_KEY=...
export ANTHROPIC_API_KEY=...  # for Claude Code
export OPENAI_API_KEY=...     # for Codex

# Seed the eval project with sample data (idempotent)
python -m skills._evals.seed_project

# Run quickstart evals with Claude Code + TUI
./skills/_evals/run.sh quickstart claude

# Run all skills with both runners
./skills/_evals/run.sh all all --seed

# Run without TUI (table output)
./skills/_evals/run.sh quickstart claude --no-tui

# JSON output for CI
python -m skills._evals.run_evals --skill quickstart --runner claude --json-output results.json
```

## Agent Runners

### Claude Code (`claude`)

Invokes `claude -p` (print mode) with MCP config pointing to the local W&B MCP server. Uses `--output-format stream-json` for structured output parsing.

Requirements: `claude` CLI >= 2.0, `WANDB_API_KEY`

### Codex (`codex`)

Invokes `codex exec` (non-interactive mode) with MCP config.

Requirements: `npx @openai/codex` >= 0.100, `WANDB_API_KEY`, `OPENAI_API_KEY`

### Mock (`mock`)

Returns preset responses per skill. No CLI or API keys needed. Used for testing the eval pipeline itself.

## Scorers

| Scorer | What it checks |
|--------|---------------|
| ToolSelectionScorer | Did the agent pick the right MCP tools? |
| WorkflowOrderScorer | Did it follow the prescribed step sequence? |
| EfficiencyScorer | How many tool calls were needed? |
| OutputQualityScorer | Does the output contain expected substrings? |
| RegexScorer | Pattern matching for verifiable outputs |
| RubricScorer | LLM-as-judge evaluation against rubric items |

## TUI

The Textual TUI (`--tui` flag) shows:
- **DataTable** (left): Scenarios with live status, duration, tools, scores
- **RichLog** (right): Full agent output for selected scenario
- **Footer**: Pass/fail counts, keyboard bindings (q=quit, Enter=detail)

## Seed Project

`seed_project.py` creates real W&B data for live evals:
- 5 W&B runs with loss/accuracy/eval_loss metrics
- 20 Weave traces with success/error mix

Configure via env vars:
- `MCP_EVAL_SEED_ENTITY` (default: `a-sh0ts`)
- `MCP_EVAL_SEED_PROJECT` (default: `mcp-skill-eval-seed`)

## Adding Evals for a New Skill

1. Add scenarios to `conftest.py` (same schema as existing)
2. Create `test_{skill}.py` with `_simulate_{skill}_skill()` and parametrized tests
3. The seed project, runners, scorers, TUI, and orchestrator are all skill-agnostic

## Design Decisions

- **Python-native over YAML** -- scorers are debuggable Python classes, not config
- **Weave-native** -- all results logged to W&B for visual comparison across versions
- **Pytest + CLI dual mode** -- mock tests for CI, live agent evals for development
- **CLI-based runners** -- tests real end-to-end behavior, not just API calls
- **Textual TUI** -- interactive debugging without leaving the terminal
