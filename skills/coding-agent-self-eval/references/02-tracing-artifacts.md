---
name: tracing-artifacts
description: Configure and validate full evaluation observability for agent runs, including trace mapping artifacts, predictions/failures JSONL, and dashboard-friendly tables.
---

# Tracing Artifacts

Use this skill to ensure agent executions are traceable and debuggable across runs.

## Scope

1. Per-question prediction logging.
2. Failure logging.
3. Trace-to-question mapping.
4. Notebook/tool evidence capture.

## Required Artifacts

Under `analytics-agent/outputs/observability/<run_id>/`:
1. `predictions.jsonl`
2. `failures.jsonl`
3. `trace_index.jsonl`
4. `notebooks.jsonl` (when enabled)

## Procedure

1. Start observability session with stable run config (`phase`, `group`, `agent_version`, `prompt_version`).
2. For each question:
   - run agent
   - extract trace metadata
   - log row via observability logger
3. Finish session and log artifact bundle to W&B.
4. Verify row counts align with processed questions.

## Validation Checklist

1. `predictions.jsonl` row count equals evaluated questions.
2. Every prediction row has `question_id`, `trace_id`, `wandb_run_id`.
3. `failures.jsonl` contains all non-true correctness rows.
4. W&B run summary includes `questions_total`, `questions_correct`, `final_accuracy`.

## Guardrails

1. Keep default mode lean; enable heavy payloads only when required.
2. Do not change row schema without updating downstream dashboard/RCA scripts.
3. Preserve stable key names for cross-run joins.
