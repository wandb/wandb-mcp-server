---
name: trace-analyst
description: Analyze Weave traces to understand LLM application behavior, summarize evaluations, debug failures, and identify patterns. Use when the user asks to "analyze traces", "summarize evaluation", "what happened in this trace", "show me trace patterns", "how is my app performing", or "latency breakdown". Do NOT use for W&B experiment run comparison (use experiment-analysis) or for building failure taxonomies (use failure-analysis).
metadata:
  author: wandb
  version: 0.1.0
  mcp-server: wandb-mcp-server
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
count_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project"
)
→ Returns: {"total_traces": 15234}
```

### Step 2: Get the Shape (Metadata First)

```
query_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project",
    metadata_only=True
)
→ Returns: op distribution, status summary, time range, token counts.
   No individual traces fetched -- fast and cheap.
```

### Step 3: Targeted Trace Queries

Based on what you learned, pull specific traces with minimal columns.

**For overview questions:**
```
query_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project",
    columns=["id", "op_name", "status", "latency_ms", "started_at"],
    limit=50
)
```

**For failure analysis:**
```
query_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project",
    status="error",
    columns=["id", "op_name", "exception", "started_at", "latency_ms"],
    limit=50
)
```

**For evaluation drill-down:**

Step A: Find the evaluation root trace:
```
query_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project",
    op_name_contains="Evaluation.evaluate",
    columns=["id", "op_name", "status", "started_at"],
    limit=10
)
```

Step B: Get child traces using the parent_id from Step A:
```
query_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project",
    parent_id="<eval_trace_id_from_step_A>",
    columns=["id", "op_name", "status", "latency_ms", "exception"],
    limit=100
)
```

> **MCP_TOOL_GAP**: The improver repo uses `Eval.from_call_id().model_calls()` for
> first-class eval call retrieval. A dedicated `eval_calls_tool` MCP tool would
> simplify this two-step parent_id pattern.

Step C: For individual call details, pull inputs/output only on specific traces:
```
query_weave_traces_tool(
    entity_name="my-team",
    project_name="my-project",
    trace_ids=["<specific_call_id>"],
    columns=["id", "op_name", "inputs", "output", "exception", "attributes"]
)
```

### Step 4: Synthesize Findings

Present a structured summary:
- Success/error rates
- Latency distribution (min, max, median if calculable)
- Top operations by frequency
- Common error patterns (if errors exist)
- Cost summary (if costs tracked)

## Common op_name Values

| op_name | What it is |
|---------|-----------|
| `openai.chat.completions.create` | OpenAI chat call |
| `anthropic.messages.create` | Anthropic call |
| `Evaluation.evaluate` | Weave evaluation root |
| `Evaluation.predict_and_score` | Individual eval sample |
| `litellm.completion` | LiteLLM call |
| Custom `@weave.op()` names | User-defined operations |

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

## Troubleshooting

- Zero traces returned -- verify entity/project name with `count_weave_traces_tool` first
- Truncated response -- reduce `limit`, request fewer columns, or filter more narrowly
- Can't find evaluation children -- ensure you're using `parent_id`, not `trace_id`
- Slow query -- add `time_range` filter to narrow the window
