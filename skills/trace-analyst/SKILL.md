---
name: trace-analyst
description: Analyze Weave traces to understand LLM application behavior, summarize evaluations, debug failures, and identify patterns. Use when the user asks to "analyze traces", "summarize evaluation", "what happened in this trace", "why did it fail", or "show me trace patterns".
---

# Trace Analyst

Analyze Weave traces for LLM application debugging, evaluation summarization, and production behavior understanding.

## When to Use

- User asks about trace behavior, success rates, or error patterns
- User wants to understand an evaluation's results
- User asks "why did this fail" or "what happened"
- User wants a summary of their application's behavior

## Workflow

### Step 1: Understand the Scale

Always start by counting -- never pull traces blind:

```
Use count_weave_traces_tool with entity_name and project_name.
This tells you how many traces exist before you query any.
```

### Step 2: Get the Shape (Metadata First)

Use `query_weave_traces_tool` with `metadata_only=True`:

```
This returns: op distribution, status summary, time range, token counts.
No individual traces are fetched -- fast and cheap.
```

### Step 3: Targeted Trace Queries

Based on what you learned, pull specific traces with minimal columns:

**For overview questions**: Request `id`, `op_name`, `status`, `latency_ms`, `started_at`

**For failure analysis**: Filter `status: error`, request `exception`, `op_name`, `started_at`

**For evaluation drill-down**:
1. Find evaluation traces: `op_name_contains: "Evaluation.evaluate"`
2. Get children: filter by `parent_id` of the evaluation trace
3. Pull child traces in batches with specific columns

### Step 4: Synthesize Findings

Present a structured summary:
- Success/error rates
- Latency distribution (min, max, median if calculable)
- Top operations by frequency
- Common error patterns (if errors exist)
- Cost summary (if costs tracked)

## Field Priority

When the response is large, the system may truncate. Fields are prioritized:

| Priority | Fields | Always available |
|----------|--------|-----------------|
| HIGH | id, op_name, status, latency_ms, exception, started_at, ended_at | Yes |
| MEDIUM | attributes, summary, costs, feedback | Truncated at L2 |
| LOW | inputs, output | Dropped first |

## Important Rules

1. **Count before querying** -- use `count_weave_traces_tool` first
2. **Start with metadata** -- `metadata_only=True` gives you the overview cheaply
3. **Request minimal columns** -- specify `columns` parameter, never pull everything
4. **Never request inputs/output on first call** -- these are large, pull only when needed
5. **Use filters** -- `status`, `op_name_contains`, `time_range`, `parent_id`
6. **For evaluations** -- always filter by `parent_id` to stay within one eval
7. **Prefer MCP tools** -- use `query_weave_traces_tool`, not `weave.init()` + Python code
