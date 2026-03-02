---
name: wandb-projects
description: Resolve and validate Weights & Biases project context before automation. Use when a coding agent needs the correct W&B entity/project values, needs to verify project existence, or needs to avoid logging to the wrong workspace.
---

# W&B Projects

Resolve project context before any run, eval, trace query, or report.

## Execute

1. Read environment values first (`WANDB_ENTITY`, `WANDB_PROJECT`, optional `WEAVE_PROJECT`).
2. Validate the entity exists by querying W&B MCP project listing.
3. Confirm the target project name is present under the resolved entity.
4. Fail fast if entity or project cannot be validated; do not guess silently.
5. Persist resolved values into run metadata used by downstream automation.

## Fallback Order

1. Use W&B MCP tools first (`query_wandb_entity_projects` and related queries).
2. If MCP is unavailable, use official W&B docs for entity/project validation workflow.
3. If docs are insufficient, inspect local library code and env-loading code paths.

## Output Contract

Return a compact context object:

```json
{
  "entity": "<resolved-entity>",
  "project": "<resolved-project>",
  "source": "mcp|env|docs-fallback",
  "validated": true
}
```
