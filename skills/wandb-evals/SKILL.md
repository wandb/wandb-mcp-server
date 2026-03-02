---
name: wandb-evals
description: Configure and run W&B-backed evaluations for agents. Use when executing benchmark slices, logging question-level correctness, and scoring model outputs for run-to-run comparison.
---

# W&B Evals

Evaluate the whole agent, not isolated tool functions.

## Execute

1. Define eval source first:
   - benchmark slice (`dataset`, `split`, `offset`, `limit`), or
   - custom eval dataset with stable IDs and expected outputs.
2. If no dataset exists, create an initial test set before iteration runs.
3. Log question-level eval outcomes:
   - `question`
   - `prediction`
   - `gold` or expected output
   - `is_correct` (canonical boolean)
   - error category when incorrect
4. Track both step-level metrics and final aggregate metrics.
5. Keep scorer logic versioned; include scorer version and dataset version in run config.
6. Persist failure rows for RCA (`failures.jsonl` or table-equivalent artifact).
7. Validate that aggregate accuracy equals derived accuracy from question rows.
8. Prefer dual scoring when needed:
   - strict score (anchor metric),
   - optional judge score (non-scalar/narrative outputs).

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
