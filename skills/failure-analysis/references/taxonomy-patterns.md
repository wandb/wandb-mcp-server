# Failure Taxonomy Patterns

Common error taxonomies discovered across LLM applications. Use as starting points for auto-generated taxonomies.

## Standard Taxonomy Categories

### API-Level Errors

| Category | Exception Patterns | Severity | Recoverable |
|----------|-------------------|----------|-------------|
| rate_limit | `RateLimitError`, `429`, `quota exceeded` | warning | Yes (retry with backoff) |
| auth_error | `AuthenticationError`, `401`, `403`, `InvalidAPIKey` | critical | No (config issue) |
| timeout | `TimeoutError`, `ReadTimeout`, `504` | warning | Yes (retry) |
| context_length | `context_length_exceeded`, `max_tokens` | warning | Yes (truncate input) |
| service_unavailable | `ServiceUnavailableError`, `503`, `overloaded` | warning | Yes (retry) |

### Application-Level Errors

| Category | Exception Patterns | Severity | Recoverable |
|----------|-------------------|----------|-------------|
| validation | `ValidationError`, `pydantic`, `schema` | warning | Yes (fix input) |
| parsing | `JSONDecodeError`, `KeyError`, `IndexError` | warning | Yes (retry with prompt fix) |
| type_error | `TypeError`, `AttributeError` | critical | No (code bug) |
| assertion | `AssertionError` | critical | No (code bug) |

### Model-Level Errors

| Category | Indicators | Severity | Recoverable |
|----------|------------|----------|-------------|
| refusal | `"I cannot"`, `"I'm sorry"`, content_filter | warning | Maybe (rephrase) |
| hallucination | Factually incorrect output (needs ground truth) | warning | Yes (retry/RAG) |
| empty_output | `output is None`, `output == ""` | warning | Yes (retry) |
| malformed_output | Structured output doesn't match schema | warning | Yes (retry with stricter prompt) |

## Severity Levels

- **critical**: Requires immediate attention, likely a code or configuration bug
- **warning**: Transient or recoverable, may need monitoring
- **info**: Expected behavior (e.g., content filter on intentionally adversarial input)

## Time-Bucketing Strategy

When analyzing error patterns over time:

1. Query with `time_range` to get temporal distribution
2. Bucket into hourly windows for spike detection
3. Compare error rate per bucket against baseline
4. Spikes often correlate with: deployments, API provider incidents, traffic surges

## Scorer Prompt Template

Use this as a starting point for LLM-based failure classification:

```
Given this trace failure:
- Operation: {op_name}
- Exception: {exception}
- Input summary: {input_summary}

Classify into exactly one category from: {taxonomy_categories}
Also assess severity: critical | warning | info
And whether the error is recoverable: true | false

Respond as JSON: {"category": "...", "severity": "...", "recoverable": ...}
```
