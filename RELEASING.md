# Releasing the W&B MCP Server

This document is the source of truth for the public server release. Managed
Cloud Run deployment details live in the private
[`wandb/wandb-mcp-server-test`](https://github.com/wandb/wandb-mcp-server-test)
repository. Dedicated and Self-Managed packaging lives in
[`wandb/helm-charts`](https://github.com/wandb/helm-charts).

The three repositories produce separate artifacts:

1. `wandb-mcp-server`: reviewed Python source and release commit.
2. `wandb-mcp-server-test`: container image and managed deployment.
3. `helm-charts`: operator chart that pins a published image tag.

Do not promote an artifact merely because another repository has merged. Record
and verify the exact source SHA, image tag or digest, and chart version at every
handoff.

## 1. Prepare the release branch

Create `staging/<version>` from a refreshed `origin/main` and immediately open a
draft `release: v<version>` PR to `main`.

Set the version consistently in:

- `pyproject.toml`
- `src/wandb_mcp_server/__init__.py`
- the root package entry in `uv.lock`

Regenerate the lockfile with `uv lock`; never edit lock metadata by hand.

Use semantic versioning:

- Patch: compatible fixes and internal hardening.
- Minor: additive public behavior or an intentional MCP tool/configuration
  migration.
- Major: broad compatibility break.

## 2. Integrate component PRs

Target each release component PR at the staging branch. Merge in dependency
order with merge commits.

Before every merge:

1. Refresh the PR head against the current staging head.
2. Reinspect the diff so cumulative branches contain only their intended layer.
3. Resolve conflicts in the component branch.
4. Run the relevant focused tests.
5. Require the current-head CI and security checks to pass.
6. Merge without an administrative branch-protection bypass.

Keep the release PR draft while components are still being added. Do not delete
stack branches until all dependent PRs are integrated.

## 3. Validate the exact release candidate

Record the final staging SHA and run:

```bash
uv lock --check
uv sync --frozen --extra test --extra http --python 3.12
uv run --no-sync ruff check src/ tests/
uv run --no-sync ruff format --check src/ tests/
uv run --no-sync pytest tests/ -m "not integration" -x -v --tb=short -q
uv run --no-sync bandit -q -r src -ll
uv build
```

The release PR must also pass:

- Full non-integration tests on Python 3.11 and 3.12.
- A fresh-wheel import and server-construction smoke test.
- Default and feature-gated tool-registration smoke tests.
- Compatibility tests against the latest supported W&B SDK.
- Grype, Bandit, and Socket Security.

Test counts are not release criteria; successful execution of the current suite
is. Do not put fixed test or tool counts in durable release documentation.

## 4. Validate managed staging

The managed wrapper must pin the exact public release-candidate SHA. Deploy that
wrapper to Cloud Run staging using the private repository procedure, then
validate:

- Health and unauthenticated rejection.
- Initialize, session continuity, `tools/list`, and representative tool calls.
- Read-only and feature-gated registration.
- Correct client/tool telemetry without raw arguments or credentials.
- Rate limiting, admission control, deadlines, and retryable overload behavior.
- Representative load without unexpected 5xx responses or material W&B
  application degradation.

Do not substitute a moving branch name for the tested SHA.

## 5. Prepare the Helm release

Open a separate draft Helm PR that:

- Pins the eventual MCP image tag.
- Renders the public `WANDB_BASE_URL`.
- Renders a namespace-local `WANDB_INTERNAL_BASE_URL` for backend W&B traffic.
- Selects the Dedicated workload profile and intended concurrency settings.
- Passes dependency build, render, lint, schema, and snapshot tests.

Do not install or upgrade a real cluster during repository-only validation.
Keep the chart PR draft until the referenced image exists.

## 6. Approve and publish

Production actions require explicit approval:

1. Mark the public release PR ready and obtain required review.
2. Merge the release PR to `main`.
3. Record the resulting `main` merge SHA.
4. Build and publish the container from that exact SHA.
5. Verify the immutable image digest.
6. Update and merge the Helm chart PR with the published tag or digest.
7. Promote managed Cloud Run from the tested staging artifact.
8. Run post-deployment health, authentication, registration, telemetry, and
   representative read checks.

Never rebuild a different source state under an already validated release tag.

## 7. Release communication

Add customer-facing notes under `docs/releases/` and include:

- The user-visible outcome.
- Breaking changes and exact migration steps.
- New operator settings and defaults.
- Security and privacy changes.
- Known limitations.
- Availability by hosted, Dedicated, and Self-Managed deployment type.

Do not announce production availability until the corresponding artifact has
been published and verified.

## Rollback

Keep the previous known-good image digest and chart version available.

- Managed: route production back to the previous verified revision.
- Dedicated/Self-Managed: restore the previous image/chart pin.
- Public source: fix forward through a reviewed PR; do not rewrite `main`.

After rollback, preserve the failing SHA, workflow run, logs, and reproduction
details for the incident review.
