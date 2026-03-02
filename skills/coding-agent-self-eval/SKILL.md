---
name: coding-agent-self-eval
description: Execute the full jupyBot-style self-evaluation loop for coding agents (Cursor, Claude Code, Codex) using W&B runs, traces, evals, dashboards, RCA, and human-gated fix promotion.
---

# Coding Agent Self Eval

Use this as a standalone workflow skill for an end-to-end RCA -> fix -> re-eval loop.

## Core Policy

No new version and no new full eval unless fix decisions are user-approved and recorded.

## Full Workflow

1. Resolve W&B project context (`entity`, `project`) from env + MCP.
2. Run benchmark slice with explicit `--limit` and `--offset`.
3. Ensure observability artifacts are written:
   - `predictions.jsonl`
   - `failures.jsonl`
   - `trace_index.jsonl`
   - `notebooks.jsonl` (when enabled)
4. Generate per-question RCA for failed questions.
5. Propose fixes with evidence links and RCA tags.
6. Run prompt governance check for `prompt_update` fixes.
7. Record human decision (`accepted` / `deferred` / `rejected`).
8. Implement only accepted fixes and commit version.
9. Run next eval slice and publish dashboard with stable `run_n` labeling.
10. Update run-centric data folders and question history.

## Required Mappings Per Run

1. `git_sha` (agent version)
2. `prompt_version`
3. `run_id`
4. `run_label` (`run_1`, `run_2`, ...)
5. slice metadata (`offset`, `limit`)

## Required Outputs

1. `fix_registry.jsonl`
2. `fix_judgement.jsonl`
3. `analytics-agent/outputs/runs/run_n/metadata.json`
4. `analytics-agent/outputs/runs/run_n/README.md`
5. `analytics-agent/outputs/runs/run_n/observability/*`
6. `analytics-agent/outputs/runs/run_n/rca/*`

## Commands (Reference)

```powershell
# benchmark run
& .\.venv\Scripts\python.exe analytics-agent/eval/runner.py --limit 100 --offset 100 --run-name run_3

# generate RCA
& .\.venv\Scripts\python.exe analytics-agent/eval/rca_from_run.py --run-id <run_id>

# propose fix
& .\.venv\Scripts\python.exe analytics-agent/eval/fix_registry.py propose --fix-id fix-XXXX --rca-tag tool_design --change-type <type> --description "<desc>" --run-id <run_id>

# prompt governance (for prompt updates)
& .\.venv\Scripts\python.exe analytics-agent/eval/prompt_governance.py --rca-tag prompt_update --pattern-failure-count <n> --pattern-threshold 5

# record human approval
& .\.venv\Scripts\python.exe analytics-agent/eval/fix_registry.py decide --fix-id fix-XXXX --decision accepted --rationale "User approved after RCA review"

# label and publish dashboard
& .\.venv\Scripts\python.exe analytics-agent/eval/label_runs.py --set <run_id>=run_3
& .\.venv\Scripts\python.exe analytics-agent/eval/publish_run_dashboard.py --run-id <run_id> --run-name "run 3" --run-label run_3

# rebuild question longitudinal history
& .\.venv\Scripts\python.exe analytics-agent/eval/question_history.py
```

## Overfitting Risk Rubric

1. Low: repeated across multiple DBs/questions and aligns with contract/tool bug.
2. Medium: repeated but concentrated in one domain/slice.
3. High: isolated one-off query pattern, likely brittle patch.

## Data Mismatch Fallback (Use When Metrics Conflict)

If W&B charts, local summary, and run artifacts disagree:

1. Treat `predictions.jsonl` + scorer output as source of truth for canonical correctness.
2. Recompute `correct`, `failed`, `accuracy` from per-question rows.
3. Regenerate RCA from the same `run_id`.
4. Re-publish dashboard tables for that `run_id`.
5. Mark run metadata with a reconciliation note before comparing to other runs.

Do not propose or approve new fixes until reconciliation is complete.

## Guardrails

1. Do not apply prompt-only fixes for isolated failures.
2. Require repeated pattern evidence before `prompt_update`.
3. Preserve stable metric keys for cross-run comparability.
4. Never overwrite historical run folders or RCA rows.
