---
name: version-mapping
description: Maintain strict mapping between code version, prompt version, run labels, and per-question outcomes for reproducible iteration tracking.
---

# Version Mapping

Use this skill for release hygiene and reproducibility in the eval loop.

## Required Mappings

For every benchmark run, map:
1. `git_sha` (agent version)
2. `prompt_version`
3. `run_id`
4. `run_label` (`run_n`)
5. evaluation slice (`offset`, `limit`)

## Procedure

1. Commit accepted fixes before running eval.
2. Run benchmark and capture run ID/URL.
3. Label run with `run_n`.
4. Publish canonical dashboard summary.
5. Create/update `analytics-agent/outputs/runs/run_n/` with:
   - `metadata.json`
   - `README.md`
   - `observability/`
   - `rca/`
6. Build/refresh question history across runs.

## Commands

```powershell
# version check
git rev-parse --short HEAD

# run label
& .\.venv\Scripts\python.exe analytics-agent/eval/label_runs.py --set <run_id>=run_3

# publish dashboard summary and RCA tables
& .\.venv\Scripts\python.exe analytics-agent/eval/publish_run_dashboard.py --run-id <run_id> --run-name "run 3" --run-label run_3

# rebuild longitudinal history
& .\.venv\Scripts\python.exe analytics-agent/eval/question_history.py
```

## Release Guardrails

1. Do not run eval on uncommitted fix batches when comparing versions.
2. Keep `run_n` sequence unique.
3. If slice differs, mark comparability limits in report.
4. Preserve historical rows; never overwrite prior run artifacts.
5. Demo navigation should start at `analytics-agent/outputs/runs/`.
