---
name: wandb-runs
description: Standardize W&B run lifecycle and logging. Use when creating or updating experiment runs with consistent naming, tags, config snapshots, and comparable metrics across iterations.
---

# W&B Runs

Create comparable runs with stable naming and schema.

## Execute

1. Start each run with a deterministic name pattern (for example `run_<n>` plus optional slice metadata).
2. Log immutable context at run start:
   - code version (`git_sha`)
   - prompt/tool version
   - dataset slice (`offset`, `limit`)
   - model identifier
3. Log per-question metrics with explicit step indexing.
4. Log run-level summary metrics at completion (`accuracy`, `correct`, `total`, error rates).
5. Apply canonical tags (for example `baseline`, `fix-batch`, `agent-vX`).
6. Keep key names stable between runs; avoid renaming metrics mid-series.

## Fallback Order

1. Use W&B SDK and MCP validation for run metadata sanity checks.
2. If behavior differs from expectation, check official W&B run logging docs.
3. If still unclear, inspect local SDK/source usage in the project codebase.

## Output Contract

Return run metadata that can be joined to RCA/report pipelines:

```json
{
  "run_id": "<wandb-run-id>",
  "run_name": "run_<n>",
  "git_sha": "<commit>",
  "slice": {"offset": 0, "limit": 100}
}
```
