# RCA And Human Gate

Move from observed failures to controlled fix promotion.

## Core Policy

No new version and no full re-eval unless fix decisions are human-approved and recorded.

## RCA Evidence Contract (Per Fix Candidate)

Every proposed fix must include:
1. failure category
2. affected question IDs
3. trace references (`trace_id`/`call_id`)
4. root-cause hypothesis
5. code touchpoint(s)
6. proposed change type:
   - `prompt_update`
   - `tool_design`
   - `architecture_change`
   - `needs_model_training`
7. regression risk note

## Procedure

1. Generate per-question RCA from failed rows.
2. Cluster by repeated failure patterns.
3. Propose candidate fixes with evidence contract above.
4. Run governance checks for prompt-related fixes.
5. Request human decision:
   - `accepted`
   - `deferred`
   - `rejected`
6. Implement only accepted fixes.
7. Re-evaluate and track per-question transitions.

## Overfitting Risk Rubric

1. Low: repeated across multiple datasets/DBs and aligns with contract/tool issue.
2. Medium: repeated but concentrated in one domain/slice.
3. High: isolated one-off query pattern with brittle patch risk.

## Required Tracking

1. fix registry (proposals, decisions, evidence links)
2. question-level fix judgements (optional but recommended)
3. run/version linkage for each decision

## Fallbacks

1. If trace retrieval fails via MCP:
   - use local `trace_index` + failed prediction rows + logs.
2. If evidence is insufficient:
   - defer decision and request additional instrumentation before implementation.
