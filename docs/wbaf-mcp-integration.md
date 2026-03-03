# Making MCP a Benchmarked Path in WandBAgentFactory

**Status:** Proposal for alignment meeting
**Prerequisite:** WBAF `codex-mcp` agent config already exists (`agents/codex/codex-mcp/config.yaml`)

---

## What exists today

WBAF already has a `codex-mcp` agent that connects to the MCP server:

```yaml
# WBAF: agents/codex/codex-mcp/config.yaml
extends: codex
env_from_host:
  - OPENAI_API_KEY
  - WANDB_API_KEY
mcp_servers:
  wandb:
    url: https://mcp.withwandb.com/mcp
    bearer_token_env_var: WANDB_API_KEY
```

This agent can already run against every WBAF task. But nobody is running it yet. The existing evals (`evals/all.yaml`) specify tasks but no agents -- agents are passed at runtime (`--agent improver-bare`).

WBAF has 30+ tasks that query W&B/Weave data. All of them should be answerable via MCP tools because the tools cover the same data surface:

| WBAF task category | MCP tools that can answer it |
|----|---|
| `count-*` (evals, traces, errors, llm-calls) | `count_weave_traces_tool` |
| `project-summary`, `list-root-ops` | `query_weave_traces_tool` (metadata_only) |
| `compare-project-traces`, `eval-*` | `query_weave_traces_tool` (with filters) |
| `email-*` (training analysis) | `query_wandb_tool` (GQL) |
| `soul/models-*` (run comparison) | `query_wandb_tool` + `create_wandb_report_tool` |

---

## Proposal 1: MCP-vs-SDK eval config

Create `evals/mcp-vs-sdk.yaml` that runs the same tasks with both agent backends:

```yaml
# WBAF: evals/mcp-vs-sdk.yaml
agents:
  - codex               # SDK path: wandb.Api() + weave SDK
  - codex-mcp           # MCP path: tool-calling via MCP server

skills:
  codex: [wandb-data-analysis]
  codex-mcp: [wandb-mcp]        # MCP-aware skill (to be authored)

tasks:
  - count-evals
  - count-traces
  - count-error-traces
  - count-llm-calls
  - project-summary
  - list-root-ops
  - compare-project-traces
  - eval-success-rates
  - eval-latency-ranking

scorers: [rubric, regex]
trials: 1
```

Run: `uv run python -m core.run_eval evals/mcp-vs-sdk.yaml`

This directly measures: does MCP tool-calling perform as well as SDK code execution for the same analytical queries? Results land in Weave (`wandb/wb-agent-frozen`) for comparison.

---

## Proposal 2: MCP-specific tasks

Some MCP capabilities don't have SDK equivalents. Add tasks that exercise these:

### Task: `mcp-report-creation`

```yaml
# WBAF: tasks/mcp-report-creation/task.yaml
instruction: |
  Create a W&B Report comparing the top 5 runs in wandb/weave-improver1
  by total token usage. Include a title and description.
  Return the report URL.

level: L1

scoring:
  scorers: [rubric, regex]
  rubric:
    - id: created_report
      text: Successfully created a W&B Report with a valid URL.
      must_pass: true
    - id: comparison_content
      text: Report compares runs by token usage.
      weight: 2.0
  regex_checks:
    - id: url_present
      pattern: 'https://wandb\.ai/.+/reports/.+'
      description: Output contains a W&B Report URL
```

### Task: `mcp-multi-tool-orchestration`

```yaml
# WBAF: tasks/mcp-multi-tool-orchestration/task.yaml
instruction: |
  In wandb/weave-improver1, find the evaluation with the most errors.
  Then drill into its child traces to identify the most common error type.
  Report: eval name, error count, most common error message, and 2 example call IDs.

level: L2

scoring:
  scorers: [rubric, regex]
  rubric:
    - id: found_eval
      text: Identified the evaluation with the most errors.
      must_pass: true
    - id: drilled_into_children
      text: Queried child traces of that evaluation.
      weight: 2.0
    - id: error_analysis
      text: Identified the most common error type with example call IDs.
      weight: 2.0
```

### Task: `mcp-wandbot-integration`

```yaml
# WBAF: tasks/mcp-wandbot-integration/task.yaml
instruction: |
  A user asks: "How do I log images to W&B from a PyTorch training loop?"
  Use the W&B support bot to get the answer, then provide a code example.

level: L0

scoring:
  scorers: [rubric]
  rubric:
    - id: used_wandbot
      text: Queried the W&B support bot for the answer.
      must_pass: true
    - id: code_example
      text: Provided a working code example with wandb.log and wandb.Image.
      weight: 2.0
```

---

## Proposal 3: Add `codex-mcp` to agent-skills CI

The `eval-skills.yml` workflow in agent-skills already evaluates skills by copying them into WBAF and running with a single agent. Extend it to test both SDK and MCP paths:

```yaml
# agent-skills: .github/workflows/eval-skills.yml (proposed change)
# In the "Run evaluation" step, run both agents:

- name: Run evaluation (SDK path)
  run: |
    cd wbaf
    uv run python -m core.run_eval evals/ci-skill-eval.yaml \
      --agent codex --weave-parallelism 15

- name: Run evaluation (MCP path)
  run: |
    cd wbaf
    uv run python -m core.run_eval evals/ci-skill-eval.yaml \
      --agent codex-mcp --weave-parallelism 15
```

This means every skill PR to agent-skills gets benchmarked on both paths. If a skill breaks MCP but not SDK (or vice versa), CI catches it.

---

## What this gives us

1. **Empirical answer** to "are MCP tools as good as SDK for analytics?" -- not opinion, data.
2. **Regression detection** when MCP tool code changes break task pass rates.
3. **Shared language** with Zubin: both paradigms measured in the same framework, same scorers, same Weave project.
4. **GTC readiness** via manual eval run (`--agent codex-mcp`). Automation is a fast follow.
