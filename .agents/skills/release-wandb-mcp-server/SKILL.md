---
name: release-wandb-mcp-server
description: Prepare, validate, audit, and hand off a public W&B MCP Server release. Use for creating or reconciling staging release branches, integrating component PRs, versioning, release notes, CI and security gates, wheel and MCP harness validation, immutable artifact evidence, release readiness reviews, or rollback preparation.
---

# Release W&B MCP Server

## Start from the release contract

1. Read `RELEASING.md`, `CONTRIBUTING.md`, and `docs/releases/v<version>.md` completely.
2. Fetch remotes and record the exact `main`, staging, and PR head SHAs. Do not use a moving branch name as evidence.
3. Confirm the semantic version and set it in `pyproject.toml`, `src/wandb_mcp_server/__init__.py`, and the root `uv.lock` entry.
4. Keep the release PR draft until every intended component is merged and the exact combined head is validated.

## Integrate reviewed work

- Retarget or recreate component PRs against `staging/<version>` only with the release owner's approval.
- Merge components in dependency order with merge commits.
- Before each merge, refresh the head, inspect the cumulative diff, rerun conflict checks and focused tests, and require current-head checks.
- Never bypass branch protection, silently drop reviewed commits, or mix unrelated deployment changes into the public release.
- After each merge, refresh remaining diffs and the release evidence.

## Validate the exact candidate

Run the commands in `RELEASING.md` on Python 3.11 and 3.12, including:

- lock validation, Ruff check and format check;
- full non-integration tests;
- W&B latest-compatibility tests;
- Bandit, Grype, and repository security checks;
- source distribution and wheel builds;
- clean installed-wheel import, server construction, and STDIO MCP sessions;
- default, read-only, and every supported feature-gated tool manifest.

Treat a manifest as a set of exact public names for the candidate, not a minimum count. Exercise representative successful calls and structured failures through the MCP protocol rather than only calling Python helpers.

## Keep release evidence safe and reproducible

- Bind staging evidence to the public SHA, wrapper SHA, immutable image digest, runtime flags, and observed tool manifest.
- Promote the exact staged digest; never rebuild the tag from a different source state.
- Keep raw credentials, prompts, customer identifiers, private repository details, service routes, and secret names out of public docs and PR artifacts.
- Run live checks only with authorized fixture credentials supplied through the environment. Use unique write fixtures and mandatory cleanup.
- Record skipped or unavailable checks as blockers or explicit residual risk; do not describe them as passing.

## Review documentation for every release

Before marking the release ready:

1. Add or update `docs/releases/v<version>.md` with outcomes, breaking changes, migration, defaults, security/privacy changes, limitations, and deployment availability.
2. Update README highlights and operator settings without embedding transient SHAs, revisions, tool counts, or deployment secrets.
3. Update `CONTRIBUTING.md`, `RELEASING.md`, and repo-local skills when the workflow itself changed.
4. Remove only documentation or helper paths that are both unreferenced and superseded.
5. Validate Markdown links, examples, and configuration names against the candidate.

## Respect approval and rollback boundaries

This skill does not authorize production. Require the repository's normal review for the public merge and explicit user approval for publishing, promotion, or customer rollout. Before any promotion, preserve the exact previous artifact and traffic/configuration state. Verify rollback restoration, not merely that a rollback command returned successfully.
