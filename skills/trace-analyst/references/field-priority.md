# Weave Trace Field Reference

Field categories used by the semantic-aware truncation system (processors.py).

## High-Signal Fields (always preserved)

These fields survive all truncation levels:

- `id` -- unique call identifier
- `op_name` -- the operation that was called (e.g., `openai.chat.completions.create`)
- `display_name` -- human-friendly name
- `trace_id` -- groups calls in the same logical trace
- `parent_id` -- parent-child relationship
- `started_at` -- UTC timestamp of call start
- `ended_at` -- UTC timestamp of call end
- `status` -- `success` | `error`
- `latency_ms` -- computed duration in milliseconds
- `exception` -- error details (null on success)

## Medium-Signal Fields (trimmed under budget pressure)

These are preserved until L2 truncation (deep-trim values >200 chars):

- `attributes` -- model name, temperature, token params (from trace attributes)
- `summary` -- aggregate stats (token counts, costs)
- `costs` -- detailed cost breakdown
- `feedback` -- human ratings or automated scores
- `wb_run_id` -- W&B run association
- `wb_user_id` -- user who triggered the trace

## Low-Signal Fields (dropped first)

These are large payload fields, dropped at L1:

- `inputs` -- full input to the operation (prompts, messages, etc.)
- `output` -- full output from the operation (completions, etc.)

## Query Column Strategy

For initial analysis, request only HIGH fields:

```
columns: ["id", "op_name", "status", "latency_ms", "started_at", "exception"]
```

For drill-down on specific traces:

```
columns: ["id", "op_name", "inputs", "output", "status", "exception"]
```

## Common op_name Values

- `openai.chat.completions.create` -- OpenAI chat calls
- `anthropic.messages.create` -- Anthropic calls
- `Evaluation.evaluate` -- Weave evaluation root
- `Evaluation.predict_and_score` -- Individual eval sample
- Custom `@weave.op()` decorated functions
