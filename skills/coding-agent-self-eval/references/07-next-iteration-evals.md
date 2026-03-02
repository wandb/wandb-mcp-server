# Next Iteration Planning

Plan post-baseline improvements with clear escalation logic.

## Priority Workstreams

1. Failure memory and fix retrieval
   - store recurring failure signatures
   - link to prior accepted/deferred fix outcomes
2. Eval quality upgrades for non-scalar answers
   - add optional LLM-judge scorer
   - add JSON/schema parsing before fallback text comparison
3. Dual scoring contract
   - strict score (anchor metric)
   - judge score (semantic coverage)
4. SFT escalation path
   - export persistent `needs_model_training` clusters
   - define repeatable training-data extraction rules

## Escalation Triggers

Escalate beyond prompt/tool/architecture patches when both are true:
1. plateau across at least 2 accepted fix iterations, and
2. persistent failure cluster exceeds project-defined threshold.

## Required Artifacts

1. memory store of prior failures and decisions
2. judge scorer config/version
3. strict + judge metric history
4. SFT candidate export manifest

## Exit Criteria

1. New run reports strict and judge metrics side by side.
2. Fix proposals cite memory evidence.
3. SFT trigger conditions are explicit and auditable.
