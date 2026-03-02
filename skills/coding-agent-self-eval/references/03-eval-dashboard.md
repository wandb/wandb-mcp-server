---
name: eval-dashboard
description: Run benchmark slices, publish canonical W&B dashboard metrics/tables, and enforce stable run naming/labeling for iteration comparison.
---

# Eval Dashboard

Use this skill to execute benchmark runs and publish comparable dashboard outputs.

## Run Naming Standard

1. Human label: `run_1`, `run_2`, `run_3`, ...
2. Optional detailed name for local execution.
3. Always apply label tags after run completion.

## Benchmark Run Procedure

1. Run eval with explicit `--limit` and `--offset`.
2. Capture run URL and run ID from output.
3. Generate RCA for that run.
4. Label run with `run_n`.
5. Publish canonical dashboard tables/summary.
6. Materialize run-centric folder:
   - `analytics-agent/outputs/runs/run_n/`
   - include run README + metadata + observability + RCA files.

## Commands

```powershell
# run eval slice
& .\.venv\Scripts\python.exe analytics-agent/eval/runner.py --limit 100 --offset 100 --run-name run_3

# generate RCA
& .\.venv\Scripts\python.exe analytics-agent/eval/rca_from_run.py --run-id <run_id>

# label
& .\.venv\Scripts\python.exe analytics-agent/eval/label_runs.py --set <run_id>=run_3

# publish dashboard
& .\.venv\Scripts\python.exe analytics-agent/eval/publish_run_dashboard.py --run-id <run_id> --run-name "run 3" --run-label run_3
```

## Required Dashboard Outputs

1. `eval/questions_total`
2. `eval/questions_correct`
3. `eval/questions_failed`
4. `eval/final_accuracy`
5. `dashboard/question_eval_table`
6. `dashboard/rca_failures_table`

## Guardrails

1. Do not compare runs from different slices without marking slice metadata.
2. If command times out locally, confirm whether W&B sync still completed before retrying.
3. Keep canonical summary keys consistent across all runs.
4. Keep demo data under `analytics-agent/outputs/runs/` so judges can navigate by run label.
