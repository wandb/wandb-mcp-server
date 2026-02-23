---
name: experiment-analysis
description: Compare W&B experiment runs, analyze metrics, and create reports. Use when the user asks to "compare runs", "analyze experiments", "which run performed best", "summarize my training", or "create a report".
---

# Experiment Analysis

Compare, analyze, and report on W&B experiment runs using the MCP tools.

## When to Use

- User asks about run comparisons, metrics, or experiment results
- User wants to find the best run by a specific metric
- User wants a report summarizing experiments

## Workflow

### Step 1: Identify the Project

If the user hasn't specified entity/project, find it:

```
Use query_wandb_entity_projects with the user's entity name.
If unknown, ask: "What's your W&B entity (username or team name)?"
```

### Step 2: Query Runs

Use `query_wandb_tool` with a GraphQL query. Always:

- Limit results with `first:` (start with 10, increase if needed)
- Request only needed fields: `name`, `displayName`, `state`, `summaryMetrics`, `config`, `createdAt`
- Use `order: "-summary_metrics.{metric}"` to sort by the metric of interest
- Use `filters` to narrow scope (by state, date range, tags)

Consult `references/gql-patterns.md` for tested query patterns.

### Step 3: Analyze Results

Extract and compare:

- Key metrics from `summaryMetrics` (this is a JSON string -- parse it)
- Config differences between top runs
- Training progression indicators (if `historyLineCount` available)

Present findings as a clear comparison table.

### Step 4: Create Report (if requested)

Use `create_wandb_report_tool` to create a shareable W&B Report:

- Structure with H1 title, H2 sections
- Include a comparison table of top runs
- Add analysis narrative
- Always return the report URL to the user

## Important Rules

1. **Use `summaryMetrics` not `metrics`** -- the `Run` type has `summaryMetrics` (JSONString), not a `metrics` field
2. **Always limit queries** -- never request all runs without `first:` parameter
3. **Parse JSON strings** -- `summaryMetrics` and `config` return JSON strings, not objects
4. **Sort server-side** -- use `order` parameter instead of fetching all and sorting client-side
5. **Prefer MCP tools over Python scripts** -- always use `query_wandb_tool` instead of writing `wandb.Api()` code

## Troubleshooting

- `"Cannot query field 'step'"` -- Use `historyLineCount` or `summaryMetrics._step` instead
- `"Cannot query field 'metrics'"` -- Use `summaryMetrics` (JSONString field)
- Empty results -- Check entity/project name, try `query_wandb_entity_projects` first
- Timeout on large projects -- Add `filters` or reduce `first:` limit
