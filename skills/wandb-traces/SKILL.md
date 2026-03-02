---
name: wandb-traces
description: Query and analyze Weave traces for debugging and RCA. Use when investigating tool-call behavior, latency/errors, or question-level failure evidence from agent runs.
---

# W&B Traces

Use Weave traces as primary evidence for failure analysis.

## Execute

1. Start with filtered trace metadata before pulling full payloads.
2. Query top-level traces first (`trace_roots_only`) to orient volume and status.
3. Narrow by run/question identifiers and error status before deep retrieval.
4. Extract minimal columns required for RCA:
   - `id`, `trace_id`, `op_name`, `status`, `latency_ms`, `exception`
   - selected `inputs`/`output` fields relevant to failure
5. Save trace evidence references alongside each RCA item.
6. Prefer small, iterative queries over one large full-data query.

## Fallback Order

1. Use W&B MCP Weave tools (`count_weave_traces_tool`, `query_weave_traces_tool`).
2. If query semantics are unclear, use official Weave docs.
3. If needed, inspect local instrumentation code and generated trace schema.

## Output Contract

For each failure, persist:

```json
{
  "question_id": "<id>",
  "run_id": "<wandb-run-id>",
  "trace_refs": [{"call_id": "<id>", "op_name": "<op>"}],
  "trace_summary": "<short evidence summary>"
}
```
