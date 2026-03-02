---
name: mcp-project-bootstrap
description: Resolve correct W&B entity/project context before running eval or RCA, using MCP tools and local config, then persist normalized runtime context.
---

# MCP Project Bootstrap

Use this skill when you need to confirm or recover W&B project coordinates (`entity`, `project`, run IDs) before running eval, dashboard publish, or trace analysis.

## Inputs

1. Optional user-provided `entity` and `project`.
2. Local `.env` and `analytics-agent/.env`.
3. MCP W&B tools access.

## Procedure

1. Read local env defaults first.
2. If entity is missing or uncertain, query MCP projects for the user/team entity.
3. Confirm project exists (`jupybot` here).
4. Persist resolved runtime context in one JSON file for downstream scripts.

## Required Output

Write `analytics-agent/outputs/runtime/project_context.json` with:
1. `entity`
2. `project`
3. `resolved_at`
4. `source` (`env`, `mcp`, or `mixed`)

## Guardrails

1. Never run eval/publish scripts until entity/project are confirmed.
2. If MCP returns transport errors, fall back to env + explicit user confirmation.
3. Keep one canonical project context file per session and update timestamp on refresh.

## Common Commands

```powershell
# read env files
Get-Content .env
Get-Content analytics-agent/.env

# verify project exists via MCP tool:
# query_wandb_entity_projects(entity="<entity>")
```
