# Project Bootstrap

Resolve correct W&B coordinates before any eval, trace query, or dashboard step.

## Purpose

Use this reference to reliably determine:
1. `entity`
2. `project`
3. auth/runtime readiness

This should work for any existing agent repository (LangGraph, Mastra, custom).

## Inputs

1. Optional user-provided `entity` and `project`
2. Repo env files (for example `.env`, `.env.local`, framework config)
3. W&B MCP tools

## Preflight (Mandatory)

1. Confirm runtime is available (`python` or `node`, depending on agent stack).
2. Confirm `WANDB_API_KEY` exists.
3. Capture optional defaults: `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_BASE_URL`, `WEAVE_PROJECT`.

Stop if runtime or API key is missing.

## Procedure

1. Load env/config defaults.
2. If `entity` is missing or uncertain, query MCP for available projects by entity.
3. Resolve exact `project` under that entity.
4. Validate project exists and is accessible.
5. Persist resolved context in a runtime artifact file for downstream steps.

## Output Contract

Write a single context JSON (path configurable by repo). Minimum fields:

```json
{
  "entity": "<resolved-entity>",
  "project": "<resolved-project>",
  "base_url": "<optional>",
  "resolved_at": "<iso-timestamp>",
  "source": "env|mcp|mixed",
  "validated": true
}
```

## Guardrails

1. Do not run eval loops before context is validated.
2. Do not silently guess entity/project when multiple candidates exist.
3. Keep one canonical context file per session and refresh timestamp on re-resolve.

## Fallbacks

1. If MCP project listing fails:
   - use env values only with explicit user confirmation.
2. If env and MCP disagree:
   - treat MCP as source of truth and record reconciliation note.
