# Skills Index

This is the main entrypoint for using W&B skills with coding agents.

## Purpose

Use this package to help a coding agent:
1. find the correct W&B project context,
2. run tracked evals with stable schemas,
3. inspect traces and failures,
4. publish comparable dashboards,
5. run an RCA-driven self-eval loop with human-gated fix promotion.

This was designed from a real loop executed in jupyBot and is intended to be reusable across agents.

## Required Environment

Set these before running any workflow:
1. `WANDB_API_KEY` (required)
2. `WANDB_ENTITY` (recommended)
3. `WANDB_PROJECT` (recommended)
4. `WANDB_BASE_URL` (optional for dedicated/on-prem)
5. `WEAVE_PROJECT` (optional, for explicit Weave routing)

If `WANDB_ENTITY` or `WANDB_PROJECT` are missing, start with `wandb-projects` to resolve them.

## How This Skill Pack Should Be Used

### Mode A: Install a single capability

Use this when you only need one task (for example traces or eval setup).

```bash
npx skills add wandb/wandb-mcp-server --skill wandb-traces
npx skills add wandb/wandb-mcp-server --skill wandb-evals
```

### Mode B: Run full coding-agent self-eval loop

Use this when you want RCA -> fix -> re-eval iterations.

```bash
npx skills add wandb/wandb-mcp-server --skill coding-agent-self-eval
```

Then follow the ordered references listed in:
`coding-agent-self-eval/references/`

## Step Order For Full Loop

1. [wandb-projects](./wandb-projects/SKILL.md)
   Resolve entity/project and block execution if unresolved.
2. [wandb-runs](./wandb-runs/SKILL.md)
   Start a run with stable naming, tags, and version metadata.
3. [wandb-traces](./wandb-traces/SKILL.md)
   Capture and query trace evidence for debugging and RCA.
4. [wandb-evals](./wandb-evals/SKILL.md)
   Run question-level scoring with canonical correctness fields.
5. [wandb-reports](./wandb-reports/SKILL.md)
   Publish dashboards/tables for run comparison and failure analysis.
6. [coding-agent-self-eval](./coding-agent-self-eval/SKILL.md)
   Execute human-gated fix promotion and next-run iteration flow.

## Agent Discovery Checklist

Before running eval loops on a new agent, collect:
1. agent entrypoint file,
2. tool files invoked by the agent,
3. eval runner location,
4. scorer implementation location,
5. run artifact output path.

Do not start comparison runs until these are confirmed.

## Expected Artifacts For Reliable RCA

Per run, ensure you can access:
1. predictions rows (question-level outputs),
2. failure rows (incorrect-only rows),
3. trace mapping (`question_id` -> trace/call IDs),
4. run summary metrics (`correct`, `failed`, `accuracy`),
5. version metadata (`git_sha`, prompt/tool version, slice metadata).

## Troubleshooting And Fallbacks

### Project or entity looks wrong

1. Re-run `wandb-projects`.
2. Validate with MCP project listing.
3. If MCP fails, use env values and require explicit user confirmation.

### Traces missing or incomplete

1. Confirm tracing is enabled in run code path.
2. Query root traces first, then narrow.
3. If MCP trace transport fails, use local artifact trace index as fallback.

### Dashboard metrics do not match run outputs

1. Treat question-level rows as canonical source.
2. Recompute aggregate metrics from rows.
3. Regenerate RCA and republish dashboard for the same run.
4. Do not approve new fixes until data is reconciled.

### A fix improves one slice but regresses another

1. Mark as potential overfit.
2. Require human decision (`accepted`, `deferred`, `rejected`).
3. Record decision and rationale before next full eval.
