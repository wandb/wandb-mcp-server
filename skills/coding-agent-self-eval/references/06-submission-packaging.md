---
name: submission-packaging
description: Build final presentation-ready README and evidence set from run artifacts, fixes, and RCA outputs.
---

# Submission Packaging

Use this skill to produce the final project presentation package quickly.

## Required Outputs

1. Root `README.md` as the main presentation entrypoint.
2. Run comparison table with slice, SHA, and accuracy.
3. Fix table with status (`implemented`, `accepted`, `deferred`).
4. Evidence links to RCA artifacts and run URLs.
5. Run-centric demo data structure under `analytics-agent/outputs/runs/run_n/`.

## Checklist

1. Confirm final run IDs and metrics from artifacts.
2. Highlight what improved and what regressed.
3. Mark cross-slice comparability limits explicitly.
4. Link skills index: `skills/skills.md`.
5. Keep backlog in separate doc, not in main presentation body.
6. Ensure each run folder has a local `README.md` with:
   - run metrics
   - RCA summary
   - applied/proposed/deferred fixes.
