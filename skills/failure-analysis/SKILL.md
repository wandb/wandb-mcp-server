---
name: failure-analysis
description: Investigate errors and failures in Weave traces and W&B runs. Use when the user asks "why did this fail", "what errors are happening", "debug my pipeline", "cluster my failures", "generate a taxonomy", or "create a scorer for my errors".
---

# Failure Analysis

Investigate, cluster, and taxonomize failures in Weave traces. Build automated scorers from discovered patterns using Weave's evaluation framework.

## When to Use

- User reports errors in their LLM application
- User wants to understand failure patterns
- User wants to build an automated taxonomy scorer
- User wants to run a "scoring backfill" on past traces

## Workflow

### Step 1: Quantify the Problem

```
1. count_weave_traces_tool -> total traces
2. query_weave_traces_tool with metadata_only=True -> status distribution
3. Note error rate: errors / total
```

### Step 2: Sample Error Traces

Pull error traces with key diagnostic fields:

```
query_weave_traces_tool:
  status: "error"
  columns: ["id", "op_name", "exception", "started_at", "latency_ms"]
  limit: 50
```

### Step 3: Cluster Errors

Group exceptions by pattern:

1. **By exception type** -- `TypeError`, `ValueError`, `APIError`, etc.
2. **By op_name** -- which operations fail most
3. **By time bucket** -- error spikes (hourly/daily)
4. **By error message similarity** -- deduplicate "same root cause" errors

Present a summary:
```
Error Cluster 1: "RateLimitError" -- 23 occurrences, all in openai.chat.completions.create
Error Cluster 2: "ValidationError" -- 12 occurrences, in my_pipeline.parse_output
Error Cluster 3: "TimeoutError" -- 8 occurrences, spread across all ops
```

### Step 4: Drill Down

For each cluster, pull 2-3 representative traces with full payloads:

```
query_weave_traces_tool:
  trace_ids: [specific IDs from Step 2]
  columns: ["id", "op_name", "inputs", "output", "exception", "attributes"]
```

Identify:
- Root cause (input validation, API limits, model hallucination, etc.)
- Whether the error is recoverable
- Suggested fix

### Step 5: Generate Taxonomy (Scorer Creation)

This step implements the Intelligent Trace Analysis pattern from the PRD.

**Auto-generate a taxonomy** from the sampled failures:

```python
import weave

@weave.op()
def failure_taxonomy_scorer(output: str, exception: str, op_name: str) -> dict:
    """Score traces into failure categories.

    Categories discovered from error sampling:
    - rate_limit: API rate limit errors
    - validation: Input/output validation failures
    - timeout: Network or API timeouts
    - model_error: Model-level failures (hallucination, refusal)
    - infrastructure: System-level failures
    - unknown: Unclassified errors
    """
    # LLM-based classification using the taxonomy
    category = classify_error(exception, op_name)  # implement with LLM call
    severity = assess_severity(exception)  # "critical" | "warning" | "info"
    return {
        "category": category,
        "severity": severity,
        "recoverable": category in ("rate_limit", "timeout"),
    }
```

**Persist the scorer** as a Weave object so users can iterate:

```python
weave.publish(failure_taxonomy_scorer, name="failure-taxonomy-v1")
```

### Step 6: Scoring Backfill

Run the scorer across historical traces:

```python
dataset = weave.Dataset(name="error-traces", rows=sampled_error_traces)

evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[failure_taxonomy_scorer],
)
results = asyncio.run(evaluation.evaluate(identity_fn))
```

### Step 7: Human-in-the-Loop Iteration

Users can then:
1. Review auto-generated taxonomy in Weave UI
2. Edit the scoring prompt / taxonomy categories
3. Re-publish as `failure-taxonomy-v2`
4. Re-run backfill with new scorer version
5. Create per-category scorers for finer analysis

## Important Rules

1. **Always quantify first** -- know the error rate before diving into individual traces
2. **Sample, don't fetch all** -- 50 error traces is enough to identify clusters
3. **Time-bucket analysis** -- error spikes suggest deployment or external service issues
4. **Severity tiers** -- not all errors are equal; rate limits vs. data corruption
5. **Persist taxonomy as a Weave object** -- enables versioning and iteration
6. **Use `weave.Evaluation`** -- not custom loops, for proper trace lineage

## Troubleshooting

- Few errors but user reports issues -- check for silent failures (status: success but bad output). Use a quality scorer instead.
- Errors only in specific time window -- check for deployment changes or upstream API outages
- Same exception, different root cause -- drill into `inputs` to distinguish
