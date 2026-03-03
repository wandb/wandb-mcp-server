# CI, Eval, and Deployment Pipeline

**Status:** Proposal for alignment meeting
**Audience:** MCP + Skills stakeholders

---

## Overview

```
Local dev          PR              Merge to main       Deploy
┌─────────┐       ┌─────────┐     ┌──────────────┐    ┌──────────────┐
│ MCP local│       │ ruff    │     │ test-repo CI │    │ test-repo    │
│ + ngrok  │──PR──>│ pytest  │──┬─>│ smoke tests  │    │ deploy.yml   │
│ + WBAF   │       │ unit    │  │  │ integration  │    │ workflow_    │
│ run_task │       └─────────┘  │  └──────────────┘    │ dispatch     │
└─────────┘                     │  ┌──────────────┐    │              │
                                └─>│ WBAF eval    │    │ ref=<sha>    │
                                   │ codex-mcp    │    │ env=prod     │
                                   │ -> Weave     │    │ -> approve   │
                                   └──────────────┘    └──────────────┘
```

Four stages, each with increasing confidence:
1. **Local dev** -- fast iteration on tools and skills with real eval feedback
2. **PR CI** -- lint + unit tests gate merge
3. **Merge to main** -- integration tests + WBAF eval against production MCP
4. **Deploy** -- manual approval from wandb-mcp-server-test, pins exact SHA

---

## 1. Local dev loop

The inner loop for iterating on MCP tools and skills. No deployment required.

### Start the MCP server locally

```bash
# In wandb-mcp-server/
uv run wandb_mcp_server --transport http --port 8000
```

The server starts on `http://localhost:8000` with your local tool changes. Analytics events log to stdout as structured JSON.

### Expose to WBAF's Modal sandbox

WBAF runs agents in Modal containers (remote). They can't reach your localhost. Use ngrok to create a public tunnel:

```bash
ngrok http 8000
# Output: https://abc123.ngrok-free.app -> http://localhost:8000
```

### Create a local WBAF agent config

```yaml
# WandBAgentFactory/agents/codex/codex-mcp-local/config.yaml
extends: codex
env_from_host:
  - OPENAI_API_KEY
  - WANDB_API_KEY
mcp_servers:
  wandb:
    url: https://abc123.ngrok-free.app/mcp    # <-- your ngrok URL
    bearer_token_env_var: WANDB_API_KEY
```

This is identical to `agents/codex/codex-mcp/config.yaml` except `url` points at your tunnel instead of `https://mcp.withwandb.com/mcp`.

### Run a single task

```bash
# In WandBAgentFactory/
uv run python -W ignore -m core.run_task \
  --agent codex-mcp-local \
  --task count-evals
```

This runs one WBAF task using your local MCP server. Output shows:
- Agent messages and tool calls
- Scorer results (pass/fail)
- Weave trace link (full trajectory at `wandb/wb-agent-frozen`)

On the MCP server side you see analytics events in stdout:
```json
{"event_type": "tool_call", "tool_name": "count_weave_traces_tool", "session_id": "sess_abc...", ...}
```

### Run a full eval

```bash
uv run python -W ignore -m core.run_eval evals/all.yaml \
  --agent codex-mcp-local \
  --weave-parallelism 5
```

Results land in Weave. Compare against SDK baseline:

```bash
uv run python -W ignore -m core.run_eval evals/all.yaml \
  --agent codex \
  --weave-parallelism 5
```

### Iterate

Change tool code in `wandb-mcp-server/src/wandb_mcp_server/mcp_tools/` -> restart the server -> re-run the task. No redeploy, no Docker build.

For skills: edit SKILL.md in `WandBAgentFactory/skills/` -> re-run the task. WBAF loads skills fresh each run.

---

## 2. PR-level CI

When you open a PR against `main` in wandb-mcp-server:

**Automatic** (from `hackathon/ci-tests` branch workflow, once merged):
```yaml
# wandb-mcp-server/.github/workflows/ci.yml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    # ruff check + ruff format --check + pytest tests/ -x -v
```

This catches lint violations, import errors, and unit test regressions in <60 seconds.

**Manual** (for significant tool changes): trigger a WBAF eval via `workflow_dispatch` against your branch. Useful when changing tool response formats or adding new tools.

---

## 3. Merge to main

Two things fire when a PR merges to `main`:

### wandb-mcp-server-test CI

The private test repo's CI ([`.github/workflows/ci.yml`](https://github.com/wandb/wandb-mcp-server-test/blob/main/.github/workflows/ci.yml)) triggers on push to its own main. It installs `wandb-mcp-server@main` from the public repo and runs:

1. **Build + import verification** (Python 3.11 + 3.12 matrix)
2. **Server smoke tests**: start with `MCP_AUTH_DISABLED=true`, hit `/health`, MCP `initialize`, `tools/list` (expects 6 tools)
3. **Integration tests** (push to main only): real `WANDB_API_KEY`, `pytest tests/ -x -v -m "not slow"` -- tests actual W&B API calls

If the test repo tracks the public repo's main (via `WANDB_MCP_SERVER_REF: "main"`), this fires automatically when the public repo updates.

### WBAF eval trigger (proposed)

Add a workflow in wandb-mcp-server that dispatches a WBAF eval when tool code changes:

```yaml
# wandb-mcp-server/.github/workflows/eval-mcp-tools.yml
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

WBAF eval config for CI (fast tasks only):

```yaml
# WandBAgentFactory/evals/mcp-ci.yaml
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

Results land in Weave (`wandb/wb-agent-frozen`). If pass rates drop below baseline, the tool change broke something.

---

## 4. Deployment approval

Production deployment is controlled entirely from **wandb-mcp-server-test** (the private repo). This is the release gate.

### Existing flow

The test repo already has [`deploy.yml`](https://github.com/wandb/wandb-mcp-server-test/blob/hackathon/deployment-infra/.github/workflows/deploy.yml):

- **Trigger**: `workflow_dispatch` only (manual)
- **Input `mcp_server_ref`**: SHA, tag, or branch from wandb-mcp-server. Resolved to exact SHA via `git ls-remote`.
- **Input `environment`**: currently `production` only
- **Target**: Cloud Run (`wandb-mcp-production`, `us-central1`, `wandb-mcp-server` service)
- **Post-deploy checks**: health, auth enforcement (401 for unauthed), authenticated MCP initialize

### How to deploy

1. Go to [wandb-mcp-server-test Actions](https://github.com/wandb/wandb-mcp-server-test/actions/workflows/deploy.yml)
2. Click "Run workflow"
3. Set `mcp_server_ref` to the SHA you want to deploy (e.g. `fa67cfb` from the PR #19 fix)
4. Set `environment` to `production`
5. If GitHub environment protection rules are configured, an approval prompt appears
6. Deploy runs: Docker build (Chainguard base) -> Cloud Run -> verify

### Approval gate

The `deploy` job uses `environment: ${{ inputs.environment }}`. GitHub environment protection rules can require:
- Reviewer approval (1+ people)
- Wait timer (e.g. 5 minutes after CI passes)
- Branch restrictions (only deploy from `main`)

Configure these in: GitHub repo settings -> Environments -> `production` -> Protection rules.

---

## 5. Adding staging

The test repo is set up for this but doesn't have it yet.

### What exists

- `deploy.yml` has an `environment` input but only `production` is listed
- `e2e_test.py` already has a `STAGING_URL` constant (`https://wandb-mcp-server-staging-778262415675.us-central1.run.app`)
- The Dockerfile and `app.py` are environment-agnostic

### Proposed changes to deploy.yml

```yaml
# wandb-mcp-server-test/.github/workflows/deploy.yml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Target environment"
        required: true
        default: "staging"      # <-- default to staging, not prod
        type: choice
        options:
          - staging             # <-- new
          - production
      mcp_server_ref:
        description: "wandb-mcp-server git ref"
        required: false
        default: "main"
```

Map environment to Cloud Run service name:

```yaml
    env:
      SERVICE_NAME: ${{ inputs.environment == 'staging' && 'wandb-mcp-server-staging' || 'wandb-mcp-server' }}
```

### Staging workflow

1. Merge tool changes to `main` in wandb-mcp-server
2. Deploy to staging: `workflow_dispatch` with `environment=staging`, `mcp_server_ref=main`
3. Run WBAF eval against staging:
   ```yaml
   # WandBAgentFactory/agents/codex/codex-mcp-staging/config.yaml
   extends: codex
   mcp_servers:
     wandb:
       url: https://wandb-mcp-server-staging-778262415675.us-central1.run.app/mcp
       bearer_token_env_var: WANDB_API_KEY
   ```
   ```bash
   uv run python -m core.run_eval evals/mcp-ci.yaml --agent codex-mcp-staging
   ```
4. Check Weave results. If pass rates look good, deploy to production.
5. Deploy to production: `workflow_dispatch` with `environment=production`, `mcp_server_ref=<same-sha>`

---

## Results flow

Two data streams from every eval run:

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

Cross-join by timestamp window gives tool-level signal: "when the agent called `query_weave_traces_tool` with these params, did the task pass?"

For local dev: the MCP server logs to stdout (analytics JSON), and WBAF logs to Weave. Both are visible in real time.

---

## Automation timeline

| Phase | What | Effort | When |
|-------|------|--------|------|
| Now | Local dev loop (ngrok + WBAF `run_task`) | Zero code changes | Today |
| Now | Manual WBAF eval (`run_eval --agent codex-mcp`) | Zero code changes | Today |
| Sprint 3 | PR-level CI (ruff + pytest) on wandb-mcp-server | Merge PR #21 | This week |
| Sprint 3 | Staging environment in deploy.yml | ~20 lines YAML | This week |
| Sprint 4 | Cross-repo WBAF eval trigger on merge to main | GitHub Action + PAT | Post-GTC |
| Sprint 4+ | Badge, Slack notification on regression | Parse eval results | Later |

The local dev loop and manual eval are usable today with zero code changes. Each subsequent phase adds automation but the signal quality is the same.
