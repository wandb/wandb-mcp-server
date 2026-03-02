---
name: rca-human-gate
description: Execute RCA workflow with mandatory human approval gate before implementing fixes or launching new eval iterations.
---

# RCA Human Gate

Use this skill when moving from observed failures to candidate fixes.

## Core Policy

No new version and no new full eval unless fix decisions are user-approved and recorded.

## RCA Procedure

1. Generate per-question RCA for target run.
2. Identify top failure categories and representative trace IDs.
3. Propose fix candidates in registry with evidence links.
4. Run governance checks for prompt fixes.
5. Request human decision (`accepted`, `deferred`, `rejected`).
6. Only implement accepted fixes.

## Commands

```powershell
# generate run RCA
& .\.venv\Scripts\python.exe analytics-agent/eval/rca_from_run.py --run-id <run_id>

# propose fix
& .\.venv\Scripts\python.exe analytics-agent/eval/fix_registry.py propose --fix-id fix-XXXX --rca-tag tool_design --change-type <type> --description "<desc>" --run-id <run_id>

# prompt governance check for prompt updates
& .\.venv\Scripts\python.exe analytics-agent/eval/prompt_governance.py --rca-tag prompt_update --pattern-failure-count <n> --pattern-threshold 5

# human decision record
& .\.venv\Scripts\python.exe analytics-agent/eval/fix_registry.py decide --fix-id fix-XXXX --decision accepted --rationale "User approved after RCA review"
```

## Overfitting Risk Rubric

1. Low: repeated across multiple DBs/questions and aligns with contract/tool bug.
2. Medium: repeated but concentrated in one domain/slice.
3. High: isolated, one-off query pattern, likely brittle patch.

## Required Tracking

1. `fix_registry.jsonl` for proposals/decisions/evidence.
2. `fix_judgement.jsonl` for question-level impact judgments.
3. Run label + agent version linkage in W&B summary.

## Human Intervention Rules

1. Human approves or defers each fix.
2. Human can require targeted slice validation before full eval.
3. Rejected/deferred fixes remain visible in registry and dashboard RCA tables.
