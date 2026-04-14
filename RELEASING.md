# Releasing the W&B MCP Server

## Overview

This repo contains the MCP server logic (tools, protocol handling, analytics). Deployment infrastructure lives in the private `wandb/wandb-mcp-server-test` repo. See that repo's [RELEASING.md](https://github.com/wandb/wandb-mcp-server-test/blob/main/RELEASING.md) for the full deployment pipeline.

## Version Bumping

Versions are tracked in `pyproject.toml`:

```toml
[project]
version = "0.3.0"
```

To release a new version:

1. Create a PR bumping the version in `pyproject.toml`
2. Update `__init__.py` `__version__` to match
3. Ensure all tests pass: `uv run pytest tests/ -v --tb=short`
4. Ensure lint passes: `uv run ruff check src/ tests/`
5. Merge to `main`

## What Happens After Merge

1. **Staging auto-deploys**: The test repo's `deploy-staging.yml` triggers on push to its `main`, resolves this repo's `main` to a SHA, and deploys to Cloud Run staging
2. **Nightly eval runs**: `eval.yml` runs 7 CI smoke tasks via WandBAgentFactory and updates README badges
3. **Manual promotion**: After staging is verified, a team member promotes to production via `promote-production.yml` in the test repo

## CI Workflows (this repo)

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | Push to main/staging/*, PR to main | Ruff lint + pytest on Python 3.11 + 3.12 |
| `eval.yml` | Nightly cron (7 AM UTC) + manual | Run MCP eval suite, update README badges |

## Release Checklist

- [ ] Version bumped in `pyproject.toml`
- [ ] `__init__.py` `__version__` matches
- [ ] All 401 unit tests pass
- [ ] CI green on PR
- [ ] Staging auto-deployed and healthy (14 tools, `/health` returns 200)
- [ ] Nightly eval passes (or manual eval triggered)
- [ ] Production promoted via test repo workflow
- [ ] On-prem image published via `publish-image.yml` in test repo
- [ ] Helm chart `values.yaml` image tag updated in helm-charts PR
- [ ] QA validated on at least one instance (17/17 deployment tests)

## Contacts

| Area | Person |
|---|---|
| MCP server code | Anish Shah (@ash0ts) |
| Code review | Nico (@NiWaRe) |
| Helm chart | Zachary Blasczyk |
| Infrastructure | Kevin Chen (@wandb-kc) |
