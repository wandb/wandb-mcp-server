# v0.3.0 Deployment & Version Strategy

**Date:** March 27, 2026 | **Status:** Pre-QA

## Three-Repo Version Coupling

The MCP server deployment spans three repositories with independently versioned artifacts:

| Repo | Artifact | Current Version | v0.3.0 Target |
|------|----------|----------------|---------------|
| `wandb-mcp-server` | Python package | 0.1.0 (pyproject.toml) | 0.3.0 |
| `wandb-mcp-server-test` | Docker image `wandb/mcp-server` | 0.2.0 (Docker Hub) | 0.3.0 |
| `helm-charts` | `operator-wandb` chart | 0.41.3 (main) / 0.42.0 (PR #571) | 0.42.0+ |

### Dependency Flow

```
wandb-mcp-server (Python source)
    |
    | pip install from git SHA
    v
wandb-mcp-server-test (Dockerfile + publish-image.yml)
    |
    | wandb/mcp-server:<tag> on Docker Hub
    v
helm-charts (operator-wandb values.yaml, mcp-server.image.tag)
    |
    | WeightsAndBiases CR spec.chart.version
    v
QA / Production clusters (operator reconciles)
```

## QA Pre-release (Current Phase)

Using the pre-release chart mechanism from `pr-release.yaml`:

- **Chart version:** `0.42.0-PR571-c86689bf` (published to `charts.wandb.ai`)
- **Image tag:** `wandb/mcp-server:0.2.0` (chart default)
- **Scope:** Single QA namespace via `WeightsAndBiases` CR override

### QA User Spec

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  name: wandb
spec:
  chart:
    url: https://charts.wandb.ai
    name: operator-wandb
    version: "0.42.0-PR571-c86689bf"
  values:
    mcp-server:
      install: true
    weave-trace:
      install: true
```

## v0.3.0 Promotion Checklist

### Step 1: Merge wandb-mcp-server

```bash
# Ensure all tests pass on feat/api-efficiency-audit
cd wandb-mcp-server/
uv run pytest tests/ -v --tb=short
uv run ruff check src/ tests/

# Merge to main (via PR)
gh pr merge <PR#> --merge
```

Record the merge commit SHA for the next step.

### Step 2: Publish Docker image

The `publish-image.yml` workflow in `wandb-mcp-server-test` is currently only on
`hackathon/deployment-infra` (PR #33). Before this step, PR #33 must be merged to
`main` or the workflow file cherry-picked.

```bash
# Trigger image publish (GitHub Actions UI or CLI)
cd wandb-mcp-server-test/
gh workflow run publish-image.yml \
  -f mcp_server_sha=<40-char-SHA-from-step-1> \
  -f version_tag=0.3.0
```

This publishes `wandb/mcp-server:0.3.0` and `wandb/mcp-server:latest`.

### Step 3: Update helm-charts default tag

In `charts/operator-wandb/values.yaml` (line ~2573):

```yaml
mcp-server:
  image:
    tag: "0.3.0"    # was "0.2.0"
```

Regenerate snapshots (`./snapshots.sh update`) and push.

### Step 4: Merge helm-charts PR #571

Once QA testing passes on the pre-release chart, merge PR #571 to `main`.
This triggers `release.yaml` which publishes the final `operator-wandb-0.42.0`
chart (no `-PR571-*` suffix).

### Step 5: Remove QA pre-release pin

Update the QA `WeightsAndBiases` CR to use the released chart version (or remove
the `spec.chart.version` override to use whatever the operator defaults to).

## Blockers & Prerequisites

| Blocker | Impact | Resolution |
|---------|--------|------------|
| `publish-image.yml` not on `main` in wandb-mcp-server-test | Cannot trigger image publish via GitHub Actions | Merge PR #33 or cherry-pick workflow file |
| `pyproject.toml` version still `0.1.0` | Package version mismatches release tag | Bump to `0.3.0` before merging to main |
| litellm security pin (test-only dep) | Does not affect Docker image (test extra only) | Already on main via PR #35 |

## Future: Automated Version Coupling

Options to reduce manual coordination:

1. **Renovate/Dependabot** on `helm-charts` watching `wandb/mcp-server` Docker Hub tags
2. **CI cross-repo trigger** from `publish-image.yml` that auto-opens a helm-charts PR bumping `mcp-server.image.tag`
3. **Digest pinning** in production: `wandb/mcp-server@sha256:...` for immutable references
4. **Unified version source**: single `VERSION` file in `wandb-mcp-server` consumed by pyproject.toml, Dockerfile tag, and helm chart default
