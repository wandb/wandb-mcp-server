---
name: develop-wandb-mcp-tools
description: Safely implement, fix, refactor, or review W&B MCP tools and their shared runtime behavior. Use for changes to public tool schemas, descriptions, registration, feature/read-only gates, W&B or Weave reads and writes, ARIA integration, admission cost, response limits, error handling, privacy, or MCP client compatibility in this repository.
---

# Develop W&B MCP Tools

## Establish the contract

1. Read `CONTRIBUTING.md` and the relevant tool, registration, and tests before editing.
2. Prefer extending an existing tool. Add a new public tool only when the request explicitly requires a distinct capability.
3. Record the intended public contract before coding:
   - input schema and validation;
   - read, write, or read/write behavior;
   - idempotency and ambiguous-timeout behavior;
   - feature and `WANDB_MCP_READ_ONLY` registration;
   - bounded request, iteration, and response behavior;
   - stable error codes and retry guidance.
4. Keep caller-supplied GraphQL out of normal tools. Use documented SDK APIs or fixed, application-owned query-only projections.

## Implement through shared boundaries

- Put tool behavior in `src/wandb_mcp_server/mcp_tools/` and register it through `register_tools()`.
- Reuse the shared API configuration, URL resolution, error sanitizer, response budget, deadline, and `InstrumentedFastMCP` boundary.
- Classify the tool in `admission.py`; an unclassified tool is deliberately treated as heavy.
- Preserve public links on `WANDB_BASE_URL`; never expose a server-side route.
- Never log raw arguments, prompts, queries, filters, credentials, resource identifiers, or upstream response bodies.
- Add only bounded, low-cardinality telemetry dimensions.
- Do not retry non-idempotent writes. Return an explicit unknown outcome when a timeout makes write completion ambiguous.
- For optional external services, require an explicit feature flag, validate the endpoint, bound input and polling, disable redirects when credentials could cross origins, and keep the group off by default.

## Test the behavior users actually invoke

Add focused tests for:

- valid, invalid, missing, null, large, cyclic, and non-finite inputs or results;
- authentication, authorization, not-found, timeout, rate-limit, overload, and malformed-upstream errors;
- default, feature-enabled, and read-only registration manifests;
- cancellation, permit cleanup, pagination, truncation, and concurrency bounds;
- credential, prompt, customer-name, and internal-host canaries in responses, logs, and telemetry;
- the installed wheel through an official MCP client when transport or registration changes.

Run the smallest relevant tests while iterating, then run the repository checks from `CONTRIBUTING.md`. For a release-targeted change, also run the complete release gate in `RELEASING.md`.

## Handle live credentials safely

Mock-based tests are the default and need no API key. Run live tests only when the user explicitly authorizes them and supplies an isolated fixture.

- Receive credentials through the environment; never inspect, print, echo, hash for display, copy, or report metadata about an `.env` file.
- Source a user-approved environment file only when necessary and only with shell tracing disabled; prefer injected CI secrets.
- Use a dedicated test entity/project, never customer data.
- Prefer read-only operations. For authorized write tests, use uniquely named fixtures and delete them in `finally`; cleanup failure fails the test.
- Sanitize logs and artifacts before attaching them to a PR.

## Finish the change

Update the README, query capability matrix, release notes, and environment example only when their public contract changed. In the PR, state the exact behavior, compatibility/security impact, tests run, and any unverified live behavior. Never present mocked coverage as live proof.
