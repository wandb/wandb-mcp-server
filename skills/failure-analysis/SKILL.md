---
name: failure-analysis
description: Investigate errors and failures in Weave traces, cluster them into categories, and build automated taxonomy scorers. Use when the user asks "why did this fail", "what errors are happening", "debug my pipeline", "cluster my failures", "generate a taxonomy", "create a scorer for my errors", "error analysis", or "scoring backfill". Do NOT use for general trace overview (use trace-analyst) or for comparing experiment runs (use experiment-analysis).
metadata:
  author: wandb
  version: 0.1.0
  mcp-server: wandb-mcp-server
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

### Step 5: Open Coding -- Write Failure Notes

> **MCP_TOOL_GAP**: Steps 5-7 require the Weave Python SDK (`weave`, `weave.flow.scorer.Scorer`).
> There is no MCP tool yet for running scorers on traces or reading feedback.
> Future MCP tools needed: `run_scorer_tool`, `read_feedback_tool`, `eval_calls_tool`.

This follows the Open Coding -> Axial Coding methodology from qualitative research.

Create a scorer that journals what went wrong for each failing call. Use an LLM to analyze the trace and produce a free-text note. Only run analysis on failing calls to reduce cost.

```python
import json
import weave
from openai import OpenAI
from weave.flow.scorer import Scorer

class OpenCodingNoteV1(Scorer):
    name: str = "open_coding_note_v1"

    @weave.op
    def score(self, output, *, failure=None):
        if not (
            isinstance(failure, dict)
            and isinstance(failure.get("output"), dict)
            and failure["output"].get("passed") is False
        ):
            return {"text": "(passed or unscored)"}

        record_text = (
            f"Op: {output.get('op_name', 'unknown')}\n"
            f"Exception: {output.get('exception', 'none')}\n"
            f"Output summary: {str(output.get('output', ''))[:500]}"
        )

        client = OpenAI()
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": "Analyze this failed eval record. Write a concise failure note (2-4 sentences)."},
                {"role": "user", "content": record_text},
            ],
            store=False,
        )
        return {"text": response.output_text}
```

Run on a small sample first, then expand:

```python
from weave_tools.weave_api import init, Eval

init("entity/project")
ev = Eval.from_call_id("YOUR_EVAL_CALL_ID")
calls = ev.model_calls()
sample = calls.limit(200)

failing = sample.filter(lambda c: (
    isinstance(c.feedback("scorer_passfail", default=None), dict)
    and c.feedback("scorer_passfail")["output"]["passed"] is False
))

for prog in failing.limit(10).run_scorer(
    OpenCodingNoteV1(),
    feedback_kwargs={"failure": "scorer_passfail"},
    max_concurrent=3,
):
    print(prog.status, prog.call_id)
```

### Step 6: Axial Coding -- Classify Into Taxonomy

Create a second scorer that reads the open-coding notes and assigns taxonomy labels.

```python
from typing import ClassVar

class AxialCodingClassifierV1(Scorer):
    name: str = "axial_coding_classifier_v1"
    TAXONOMY: ClassVar[list[str]] = [
        "rate_limit", "auth_error", "timeout", "context_length",
        "validation", "parsing", "type_error",
        "refusal", "hallucination", "empty_output",
        "infrastructure", "unknown",
    ]

    @weave.op
    def score(self, output, *, note=None):
        if not isinstance(note, dict) or note.get("text", "").startswith("(passed"):
            return {"labels": [], "primary": "skipped", "rationale": "Not a failure"}

        prompt = (
            f"Given this failure note:\n{note['text']}\n\n"
            f"Classify into one or more of: {self.TAXONOMY}\n"
            "Return JSON: {\"labels\": [...], \"primary\": \"...\", \"rationale\": \"...\"}"
        )

        client = OpenAI()
        response = client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": prompt}],
            store=False,
        )

        text = response.output_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)

        valid_labels = [l for l in result.get("labels", []) if l in self.TAXONOMY]
        if not valid_labels:
            valid_labels = ["unknown"]
        result["labels"] = valid_labels
        if result.get("primary") not in self.TAXONOMY:
            result["primary"] = valid_labels[0]
        return result
```

Chain it by passing the open-coding notes as `feedback_kwargs`:

```python
for prog in calls.run_scorer(
    AxialCodingClassifierV1(),
    feedback_kwargs={"note": OpenCodingNoteV1().name},
    max_concurrent=6,
):
    pass
```

### Step 7: Summarize and Iterate

Count failures by taxonomy label and provide actionable recommendations:

```python
from collections import Counter

label_counts = Counter()
primary_counts = Counter()

for call in calls:
    cls = call.feedback("axial_coding_classifier_v1", default=None)
    if not isinstance(cls, dict):
        continue
    for label in cls.get("labels", []):
        label_counts[label] += 1
    primary = cls.get("primary")
    if isinstance(primary, str):
        primary_counts[primary] += 1

print("PRIMARY:", primary_counts.most_common())
print("MULTI:", label_counts.most_common())
```

**Human-in-the-loop iteration:**
1. Review taxonomy results in Weave UI
2. Edit the `TAXONOMY` list or LLM prompt in `AxialCodingClassifierV1`
3. Re-run with `run_scorer()` -- new feedback versions stack, old ones are preserved
4. Create per-category scorers for deeper analysis on the largest clusters

## Important Rules

1. **Always quantify first** -- know the error rate before diving into individual traces
2. **Sample, don't fetch all** -- 50 error traces is enough to identify clusters
3. **Time-bucket analysis** -- error spikes suggest deployment or external service issues
4. **Severity tiers** -- not all errors are equal; rate limits vs. data corruption
5. **Use `Scorer` subclasses, not bare `@weave.op()` functions** -- they integrate with `run_scorer()` and feedback chaining
6. **Use `weave.Evaluation`** -- not custom loops, for proper trace lineage

## Pydantic v2 and run_scorer Gotchas

- Pydantic v2 requires `name: str = "..."` (no untyped override)
- Class constants need `ClassVar[...]` annotation
- Access `name` via an instance (e.g., `OpenCodingNoteV1().name`) to avoid `AttributeError` on the class
- `run_scorer` only accepts `feedback_kwargs` (no `additional_scorer_kwargs`)
- `Progress` from `run_scorer` uses `.status` and `.error` (no `.exception`)
- If the LLM returns fenced JSON, strip fences before `json.loads`
- Validate labels against your taxonomy; map unknowns to `"unknown"` to keep counts clean
- If you set `max_output_tokens`, keep it >= 16 to satisfy API minimums

## Troubleshooting

- Few errors but user reports issues -- check for silent failures (status: success but bad output). Use a quality scorer instead.
- Errors only in specific time window -- check for deployment changes or upstream API outages
- Same exception, different root cause -- drill into `inputs` to distinguish
- `AttributeError: 'OpenCodingNoteV1' has no attribute 'name'` -- access on instance, not class

## Future MCP Tools Needed

The following SDK capabilities are used in Steps 5-7 and should become MCP tools:

| Capability | SDK Call | Proposed MCP Tool |
|---|---|---|
| Run a scorer on traces | `calls.run_scorer(MyScorer())` | `run_scorer_tool` |
| Read feedback from calls | `call.feedback("scorer_name")` | `read_feedback_tool` |
| Get eval model calls | `Eval.from_call_id().model_calls()` | `eval_calls_tool` |
