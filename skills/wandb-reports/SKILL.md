---
name: wandb-reports
description: Build W&B dashboards and reports that explain run quality and fix impact. Use when presenting accuracy trends, failure buckets, RCA summaries, and cross-run comparisons.
---

# W&B Reports

Publish readable run evidence for humans and automation.

## Execute

1. Create a run dashboard with canonical panels:
   - overall accuracy
   - correct vs failed counts
   - sql/tool error rates
   - retry/tool-call distributions
2. Include question text, prediction, and correctness in tabular views.
3. Add RCA summary panels or linked tables for failed questions.
4. Use stable run labels (`run_1`, `run_2`, `run_3`, ...).
5. Keep chart semantics consistent between runs to avoid misleading comparisons.

## Fallback Order

1. Use W&B report APIs and MCP report creation tools.
2. If layout semantics are unclear, use official W&B reporting docs.
3. If needed, inspect existing report/dashboard scripts in the repo.

## Output Contract

Return a report artifact descriptor:

```json
{
  "run_id": "<wandb-run-id>",
  "run_label": "run_4",
  "dashboard_url": "<url>",
  "panels": ["accuracy", "failure_table", "rca_summary"]
}
```
