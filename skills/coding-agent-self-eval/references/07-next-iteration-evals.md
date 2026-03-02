---
name: next-iteration-evals
description: Plan and execute next-step eval improvements: failure memory, LLM-judge scoring, JSON answer parsing, and SFT escalation criteria.
---

# Next Iteration Evals

Use this skill after baseline submission when planning the next improvement cycle.

## Priority Workstreams

1. Failure memory and fix retrieval:
   - store failure signatures and prior fix outcomes
   - retrieve before proposing new fixes
2. Non-scalar eval quality:
   - add LLM-judge scorer for text responses
   - add JSON schema parsing fallback for typed answer extraction
3. Dual scoring:
   - strict exact score
   - judge score for narrative outputs
4. SFT escalation:
   - trigger when approved fix iterations plateau
   - export persistent `needs_model_training` clusters

## Exit Criteria

1. Next run reports strict + judge metrics side by side.
2. Fix proposals reference memory entries.
3. SFT trigger condition is explicit and measurable.
