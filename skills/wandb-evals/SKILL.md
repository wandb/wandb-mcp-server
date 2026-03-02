---
name: wandb-evals
description: Configure and run W&B-backed evaluations for agents. Use when executing benchmark slices, logging question-level correctness, and scoring model outputs for run-to-run comparison.
---

# W&B Evals

Evaluate the whole agent, not isolated tool functions.

## Execute

1. Define the eval slice deterministically (`dataset`, `offset`, `limit`).
2. Log question-level eval outcomes:
   - `question`
   - `prediction`
   - `gold` or expected output
   - `is_correct` (canonical boolean)
   - error category when incorrect
3. Track both step-level metrics and final aggregate metrics.
4. Keep scorer logic versioned; include scorer version in run config.
5. Persist failure rows for RCA (`failures.jsonl` or table-equivalent artifact).
6. Validate that aggregate accuracy equals derived accuracy from question rows.

## Fallback Order

1. Use W&B/Weave eval tooling and project scorer scripts.
2. If scorer behavior is ambiguous, check official W&B eval docs.
3. If unresolved, inspect local eval runner/scorer code directly.

## Output Contract

Produce both:

1. Question-level table with `is_correct`.
2. Run-level summary:

```json
{
  "correct": 53,
  "total": 100,
  "accuracy": 0.53
}
```
