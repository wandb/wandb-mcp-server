# Tracing And Artifacts

Ensure every eval run is traceable and RCA-ready.

## Purpose

Capture minimal but sufficient observability for:
1. question-level scoring,
2. trace-backed RCA,
3. cross-run comparison.

## Required Artifacts Per Run

Use a configurable run artifact root. Minimum files:
1. `predictions.jsonl`
2. `failures.jsonl`
3. `trace_index.jsonl`
4. `notebooks.jsonl` (optional, only when notebook/tool evidence is enabled)

## Required Row Fields

### predictions rows

1. `question_id`
2. `prediction` or `answer_value`
3. `expected` or gold field
4. `is_correct`
5. `wandb_run_id`
6. `trace_id`
7. `call_id`
8. `agent_version`
9. `prompt_version`

### trace index rows

1. `question_id`
2. `trace_id`
3. `call_id`
4. `op_name`
5. `status`
6. optional timing fields (`started_at`, `ended_at`, `latency_ms`)

## Procedure

1. Start run with stable metadata (`agent_version`, `prompt_version`, dataset slice/version).
2. For each eval question:
   - run the full agent,
   - capture prediction + correctness,
   - capture trace IDs and selected metadata.
3. Write `failures.jsonl` as strict subset where `is_correct=false`.
4. Log run-level summary metrics and upload artifacts.

## Validation Checklist

1. `predictions.jsonl` count equals evaluated question count.
2. `failures.jsonl` count equals number of incorrect rows.
3. Every failed row has trace reference (`trace_id` or `call_id`).
4. Run summary metrics match row-derived totals.
5. Prompt budget fields exist (`prompt_chars`, `prompt_tokens_est`, `prompt_budget_ok`) when supported.

## Guardrails

1. Keep default mode lean; enable heavy payloads only when needed.
2. Do not rename core keys between runs.
3. Do not skip `failures.jsonl`; it is required for RCA speed.

## Fallbacks

1. If MCP trace retrieval fails (transport/decode issues):
   - use local `trace_index.jsonl` + run artifacts as primary RCA source.
2. If trace IDs are missing:
   - fail run quality gate and rerun with tracing enabled.
