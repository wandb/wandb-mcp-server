---
name: coding-agent-self-eval
description: Guide coding agents on how to run a full self-evaluation loop on the AI agent they are working on: define eval datasets/benchmarks, instrument W&B runs and Weave traces, execute evals, perform RCA, apply human-gated fixes, and iterate with versioned comparisons.
---

# Coding Agent Self Eval

Use this as a standalone workflow skill for an end-to-end RCA -> fix -> re-eval loop.

## Core Policy

No new version and no new full eval unless fix decisions are user-approved and recorded.

## Load Order

Read and execute these reference docs in order:

1. `references/01-mcp-project-bootstrap.md`
2. `references/02-tracing-artifacts.md`
3. `references/08-wandb-evals-setup.md`
4. `references/03-eval-dashboard.md`
5. `references/04-rca-human-gate.md`
6. `references/05-version-mapping.md`
7. `references/06-submission-packaging.md`
8. `references/07-next-iteration-evals.md`

## Data Reconciliation Rule

If data is inconsistent across W&B charts, local summaries, and artifacts:

1. Treat `predictions.jsonl` and scorer rows as canonical correctness source.
2. Recompute run metrics from question rows.
3. Regenerate RCA for the same `run_id`.
4. Re-publish dashboard tables.
5. Continue only after consistency is restored.

## Why This Skill Is Standalone

This skill does not depend on parent/base skills at runtime.
It contains the original jupyBot execution workflow through the reference files above.
