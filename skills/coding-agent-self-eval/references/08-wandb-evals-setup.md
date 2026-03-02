---
name: wandb-evals-setup
description: Configure W&B/Weave evaluation setup for benchmark runs, including scorer strategy, run config schema, and required eval logs for comparison.
---

# W&B Evals Setup

Use this skill before running benchmark slices when you need consistent evaluator behavior and run-to-run comparability.

## Purpose

Define exactly how questions are scored and what eval metadata is logged so RCA and dashboards are reliable.

## Eval Setup Checklist

1. Confirm scoring mode:
   - strict scalar score (`exact/numeric`)
   - optional judge score (`text/narrative`)
2. Confirm benchmark slice:
   - `offset`, `limit`, split id
3. Confirm run config payload includes:
   - `agent_version`
   - `prompt_version`
   - `model`
   - `run_variant`
   - `iteration`
4. Confirm per-question logs include:
   - `question_id`
   - `correct` / `exec_accuracy`
   - `running_accuracy`
   - `trace_id`
   - `db_id`

## Baseline Scoring Contract (Current Project)

1. Gold value extraction: first row, first column from gold SQL.
2. Agent answer uses `answer_value` as primary scored field.
3. This is strict and can under-score multi-row narrative answers.

## Recommended Next Scoring Upgrade

1. Keep strict score for continuity.
2. Add judge score for non-scalar outputs.
3. Log both:
   - `eval/strict_accuracy`
   - `eval/judge_accuracy`

## Commands

```powershell
# benchmark run (uses current scorer contract)
& .\.venv\Scripts\python.exe analytics-agent/eval/runner.py --limit 100 --offset 0 --run-name run_2

# publish canonical eval dashboard metrics
& .\.venv\Scripts\python.exe analytics-agent/eval/publish_run_dashboard.py --run-id <run_id> --run-name "run 2" --run-label run_2
```

## Guardrails

1. Do not change scoring contract mid-comparison without marking run metadata.
2. If scorer logic changes, create a new run variant and document in run README.
3. Keep strict score as anchor metric when adding judge-based evaluation.
