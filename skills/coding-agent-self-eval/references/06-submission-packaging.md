# Evidence Packaging

Package proof that the self-eval loop actually ran and produced measurable outcomes.

## Purpose

Produce an evidence bundle that reviewers can verify without rerunning the entire project.

## Minimum Evidence Bundle

1. Run comparison table:
   - run label, run ID, dataset/slice, version ID, accuracy
2. Fix ledger:
   - proposed/accepted/deferred/rejected
3. RCA summaries per run:
   - failure categories and counts
4. Artifact links:
   - predictions/failures/trace mapping
5. dashboard/report links

## Optional Presentation Outputs

1. Root `README.md` summary
2. flowchart of eval loop
3. skill index links

## Checklist

1. Confirm run IDs and canonical metrics from artifacts.
2. Mark comparability limits when slice/scorer changes.
3. Ensure each run folder has:
   - run metadata,
   - observability files,
   - RCA outputs,
   - short run summary.
4. Keep backlog/next ideas in separate doc from evidence summary.
