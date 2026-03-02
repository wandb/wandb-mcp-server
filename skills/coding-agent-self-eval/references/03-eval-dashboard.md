# Eval Execution And Dashboard

Run comparable eval iterations and publish canonical dashboard outputs.

## Prerequisite

Define eval source before run:
1. benchmark slice (`dataset`, `split`, `offset`, `limit`), or
2. custom eval dataset with stable question IDs and scorer contract.

Do not run dashboard comparisons without a defined eval source.

## Run Naming Standard

1. Human label: `run_1`, `run_2`, `run_3`, ...
2. Optional detailed execution name.
3. Always apply label tags after run completion.

## Procedure

1. Execute eval with explicit slice or dataset version.
2. Capture run ID and URL.
3. Generate RCA for the run.
4. Apply stable run label (`run_n`).
5. Publish dashboard tables and summary metrics.
6. Materialize run-centric folder under configurable output root:
   - metadata
   - observability artifacts
   - RCA artifacts.

## Required Dashboard Outputs

1. `eval/questions_total`
2. `eval/questions_correct`
3. `eval/questions_failed`
4. `eval/final_accuracy`
5. optional `eval/judge_accuracy` (if judge scorer is enabled)
6. question-level eval table (question, prediction, expected, correctness)
7. RCA failure table (failure category + trace references)

## Reconciliation Rule (Mandatory)

If dashboard values conflict with local artifacts:
1. recompute metrics from question rows,
2. treat recomputed values as canonical,
3. regenerate RCA and republish dashboard,
4. compare runs only after reconciliation.

## Guardrails

1. Do not compare runs from different datasets/slices without explicit metadata.
2. Keep metric keys stable across runs.
3. Preserve run labels and avoid relabeling historical runs.
