---
name: experiment-analysis
description: Compare W&B experiment runs, analyze metrics, and create reports. Use when the user asks to "compare runs", "analyze experiments", "which run performed best", "summarize my training", "create a report", "show me run configs", or "rank my models". Do NOT use for Weave trace analysis (use trace-analyst) or for error debugging (use failure-analysis).
metadata:
  author: wandb
  version: 0.1.0
  mcp-server: wandb-mcp-server
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

> **MCP_TOOL_GAP**: For detailed metric history (training curves, loss over time),
> the improver repo uses `run.history(samples=200, keys=["loss"])` and
> `run.scan_history(keys=[...])`. The GQL `sampledHistory` query via
> `query_wandb_tool` partially covers this, but there is no MCP tool for
> streaming full history. A future `run_history_tool` would close this gap.

### Step 4: Create Report (if requested)

Use `create_wandb_report_tool` to create a shareable W&B Report:

- Structure with H1 title, H2 sections
- Include a comparison table of top runs
- Add analysis narrative
- Always return the report URL to the user

> **MCP_TOOL_GAP**: The current `create_wandb_report_tool` accepts basic report
> structure but lacks support for structured Runset filters (`expr.Config(...)`,
> `expr.Summary(...)`, `expr.Tags().isin([...])`). The improver repo uses the
> `wandb.apis.reports` SDK for this. A future `report_structured_filter_tool`
> would enable richer reports via MCP.

## Important Rules

1. **Use `summaryMetrics` not `metrics`** -- the `Run` type has `summaryMetrics` (JSONString), not a `metrics` field
2. **Always limit queries** -- never request all runs without `first:` parameter
3. **Parse JSON strings** -- `summaryMetrics` and `config` return JSON strings, not objects
4. **Sort server-side** -- use `order` parameter instead of fetching all and sorting client-side
5. **Prefer MCP tools over Python scripts** -- always use `query_wandb_tool` instead of writing `wandb.Api()` code
6. **Avoid printing full metric lists** -- summarize with aggregates, not raw dumps

## Troubleshooting

- `"Cannot query field 'step'"` -- Use `historyLineCount` or `summaryMetrics._step` instead
- `"Cannot query field 'metrics'"` -- Use `summaryMetrics` (JSONString field)
- Empty results -- Check entity/project name, try `query_wandb_entity_projects` first
- Timeout on large projects -- Add `filters` or reduce `first:` limit

## Appendix: SDK Fallback Patterns

When MCP tools don't yet support a capability, users with the `wandb` SDK installed
can use these battle-tested patterns from the improver repo. These are documented
here for reference until equivalent MCP tools are built.

### Run History Sampling (SDK)

```python
import wandb

api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_name}")

# Sampled history (fast, returns pandas DataFrame)
hist = run.history(samples=200, keys=["loss", "accuracy", "_step"], pandas=True)
print(hist.tail())

# Full history streaming (for large runs)
for i, row in enumerate(run.scan_history(keys=["loss", "accuracy"], page_size=1000)):
    if i >= 5:
        break
    print(row)
```

### W&B Reports with Structured Filters (SDK)

```python
from wandb.apis import reports as wr
import wandb_workspaces.expr as expr

filters = [
    expr.Config("model") == "gpt-4",
    expr.Summary("accuracy") >= 0.9,
    expr.Metric("State") == "finished",
]

runset = wr.Runset(entity=entity, project=project, name="Top runs", filters=filters)
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["loss"]),
        wr.BarPlot(title="Accuracy", metrics=["accuracy"], orientation="v"),
    ],
)

report = wr.Report(
    entity=entity, project=project,
    title="Experiment Analysis",
    width="fixed",
    blocks=[wr.H1(text="Results"), plots],
)
report.save(draft=True)
```

**Key SDK gotchas** (from improver):
- Avoid dot-paths like `config.lr` in filter strings; use `Config("lr")` instead
- `query` in Runset is a regex on run name, not a filter language
- For explicit run IDs, use `Metric("name").isin([...])`, not `ID`
- Preflight string filters with `expr.expr_to_filters(...)` to verify keys

## Future MCP Tools Needed

| Capability | SDK Call | Proposed MCP Tool |
|---|---|---|
| Stream/sample run history | `run.history()` / `run.scan_history()` | `run_history_tool` |
| Structured report filters | `wandb.apis.reports` with `expr.*` | `report_structured_filter_tool` |
