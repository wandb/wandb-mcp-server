# CI/Eval Pipeline: Connecting WBAF Evals to MCP Server CI

**Status:** Proposal for alignment meeting
**Depends on:** PR #21 (wandb-mcp-server CI), WBAF `eval-skills.yml` workflow

---

## Current state: two disconnected CI systems

```
wandb-mcp-server CI (PR #21)          WBAF CI (eval-skills.yml)
┌───────────────────────┐              ┌───────────────────────┐
│ Trigger: push to main │              │ Trigger: push to main │
│ Runs: ruff + pytest   │              │   in agent-skills     │
│ Tests: unit only      │              │ Runs: skill evals     │
│ No eval coverage      │              │   via codex agent     │
│                       │              │ No MCP path tested    │
└───────────────────────┘              └───────────────────────┘
```

Neither system tests MCP tools against real analytical tasks. Unit tests verify tool code doesn't crash; they don't verify that an agent can use the tools to answer "how many evals are in this project?"

---

## Proposed: cross-repo eval trigger

When MCP tool code changes, trigger WBAF evals against the live MCP server:

```
wandb-mcp-server                       WBAF
┌───────────────────┐                  ┌───────────────────────┐
│ push to main      │  workflow_       │ eval-skills.yml       │
│ src/wandb_mcp_    │  dispatch        │ agent: codex-mcp      │
│ server/mcp_tools/ ├────────────────> │ tasks: L0 + L1 subset │
│                   │                  │ results -> Weave      │
└───────────────────┘                  └───────────────────────┘
```

### wandb-mcp-server workflow

```yaml
# .github/workflows/eval-mcp-tools.yml
name: Eval MCP tools via WBAF

on:
  push:
    branches: [main]
    paths:
      - 'src/wandb_mcp_server/mcp_tools/**'
      - 'src/wandb_mcp_server/weave_api/**'
  workflow_dispatch:

jobs:
  trigger-wbaf-eval:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger WBAF eval
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.WBAF_PAT }}
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: 'wandb',
              repo: 'WandBAgentFactory',
              workflow_id: 'eval-skills.yml',
              ref: 'main',
              inputs: {
                agent: 'codex-mcp',
                eval_config: 'evals/mcp-ci.yaml',
                skills: 'wandb-mcp',
              }
            });
```

### WBAF eval config for MCP CI

```yaml
# WBAF: evals/mcp-ci.yaml
# Lightweight eval for CI -- fast tasks only (L0 + L1)
tasks:
  - count-evals
  - count-traces
  - count-error-traces
  - count-llm-calls
  - list-root-ops
  - project-summary
  - eval-success-rates

scorers: [rubric, regex]
trials: 1
```

---

## Results flow

```
MCP server      Cloud Logging    BigQuery     Hex dashboard
  tool calls ──> structured ───> analytics ──> tool health
                 JSON logs       events        per-tool P95
                                               success rate

WBAF eval       Weave traces     Weave UI     task pass rate
  agent runs ──> full trace ────> project ───> by agent type
                 with scores     wb-agent-     SDK vs MCP
                                 frozen
```

Two data streams, joinable by timestamp window:
- **MCP analytics** (BigQuery): per-tool latency, success/failure, params, session_id
- **WBAF eval results** (Weave): per-task pass/fail, agent trajectory, skill used, scorer breakdown

Cross-joining them gives: "when the agent called `query_weave_traces_tool` with these params, did the task pass?" This is tool-level signal that guides which tools need improvement.

---

## GTC readiness

The full CI pipeline is not needed for GTC. The manual version is:

```bash
# In WBAF repo
uv run python -m core.run_eval evals/mcp-vs-sdk.yaml \
  --agent codex-mcp \
  --weave-parallelism 5
```

Check results in Weave at `wandb/wb-agent-frozen`. Compare `codex-mcp` pass rates against `codex` (SDK) pass rates for the same tasks.

If MCP pass rate >= SDK pass rate on L0/L1 tasks, MCP tools are GTC-ready.
If not, the failing tasks point to specific tools that need improvement.

---

## Automation timeline

| Phase | What | When |
|-------|------|------|
| Manual | Run `codex-mcp` eval in WBAF, inspect Weave | Now |
| Semi-auto | GitHub Action in wandb-mcp-server triggers WBAF eval on tool changes | Post-GTC |
| Full auto | Results parsed, badge updated, Slack notification on regression | Sprint 4+ |

The manual phase is sufficient for GTC. Each subsequent phase reduces human-in-the-loop but the signal quality is the same.
