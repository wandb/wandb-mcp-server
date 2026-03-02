# W&B Evals Setup

Configure eval inputs, scorer contract, and logging schema before iteration runs.

## Purpose

Make eval runs comparable and RCA-ready across versions.

## Eval Source Branch

Choose one:
1. existing benchmark (define deterministic slice), or
2. custom eval dataset (build test set + scorer rubric).

If no benchmark exists, create a custom eval set with stable IDs and expected outputs before running RCA loops.

## Eval Setup Checklist

1. Define scoring mode:
   - strict score (required anchor metric)
   - judge score (optional, recommended for non-scalar outputs)
2. Define dataset metadata:
   - dataset name/version
   - split/slice (`offset`, `limit`, or explicit sample IDs)
3. Define run config fields:
   - `agent_version`
   - `prompt_version`
   - `model`
   - `scorer_version`
   - `dataset_version`
   - `iteration`
4. Define per-question log fields:
   - `question_id`
   - `prediction`
   - `expected`
   - `is_correct`
   - optional `judge_score`
   - `trace_id`

## Scoring Contract Guidance

1. Keep strict score as continuity anchor.
2. Add judge score only with explicit versioned rubric.
3. If contract changes, mark new run variant and avoid direct apples-to-apples comparison without note.

## Guardrails

1. Do not change scorer logic mid-run.
2. Do not compare runs with different dataset/scorer contracts without explicit comparability note.
3. Ensure aggregate metrics can be recomputed from question rows.
