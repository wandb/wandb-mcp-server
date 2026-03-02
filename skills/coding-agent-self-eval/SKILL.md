---
name: coding-agent-self-eval
description: Orchestrate a self-improving coding-agent loop using W&B projects, runs, traces, evals, and reports. Use when running iterative RCA-to-fix cycles with human approval gates and version-to-run mapping.
---

# Coding Agent Self Eval

Run a full loop with evidence and controls.

## Prerequisites

Use these skills first or in parallel:
1. `wandb-projects`
2. `wandb-runs`
3. `wandb-traces`
4. `wandb-evals`
5. `wandb-reports`

## Loop

1. Resolve entity/project and validate context.
2. Execute eval slice and log question-level correctness.
3. Publish run dashboard and failure artifacts.
4. Run RCA on failed questions using traces plus code references.
5. Classify each fix candidate:
   - prompt update
   - tool design
   - architecture change
   - needs model training
6. Enforce human approval gate before implementing fixes.
7. Implement approved fixes, version code, and launch next run.
8. Compare run-to-run outcome deltas and repeat.

## Guardrails

1. Do not patch prompt blindly for isolated failures.
2. Apply prompt changes only on repeated pattern evidence.
3. Track prompt growth and enforce a hard prompt-size budget.
4. Maintain fix registry and version mapping for auditability.

## Fallback Order

1. Use W&B MCP data + local run artifacts as primary truth.
2. Use official docs only when tool behavior is uncertain.
3. Inspect local library code when docs/tool outputs conflict.
