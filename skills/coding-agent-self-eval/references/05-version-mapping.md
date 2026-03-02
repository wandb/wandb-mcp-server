# Version And Run Mapping

Maintain reproducibility across iterations.

## Required Mapping Fields

For every run, track:
1. `git_sha` or code version ID
2. `prompt_version`
3. `tooling_version` (if separate)
4. `scorer_version`
5. dataset version or benchmark slice (`offset`, `limit`, split)
6. `run_id`
7. `run_label` (`run_n`)
8. run URL

## Procedure

1. Commit accepted changes before running comparison eval.
2. Execute run and capture ID/URL.
3. Apply stable label (`run_n`).
4. Store run metadata in a run-centric folder.
5. Store observability and RCA artifacts under same run label.
6. Rebuild question-level history across runs.

## Recommended Run-Centric Folder Contract

Under configurable output root:
1. `run_n/metadata.json`
2. `run_n/README.md`
3. `run_n/observability/*`
4. `run_n/rca/*`

## Guardrails

1. Do not compare uncommitted local code against committed runs.
2. Keep `run_n` labels unique and monotonic.
3. Mark comparability limits when dataset/slice/scorer changes.
4. Never overwrite prior run artifacts.
