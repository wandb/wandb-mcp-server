# Observability

The MCP server emits product analytics (Segment) and operational telemetry (Datadog).
This document covers the two supported Datadog collection modes, the environment
variables that control them, and when to pick each.

## TL;DR

| Deployment target | Recommended mode | What to set |
|---|---|---|
| Managed Kubernetes (dedicated cloud, self-managed) | **Agent mode** | Install DD Agent DaemonSet once per cluster; set `MCP_LOG_FORMAT=json` on the pod. No DD credentials on the workload. |
| Serverless | **Forwarder mode** | `MCP_DATADOG_FORWARD=true` plus a secret-backed `DD_API_KEY` |
| Local dev | No Datadog | Leave `MCP_DATADOG_FORWARD` unset; `MCP_LOG_FORMAT` defaults to `rich` |

## Two collection modes

### Agent mode (preferred on Kubernetes)

A Datadog Agent DaemonSet on every node:

- **Logs**: tails `/var/log/pods/**` (container stdout/stderr) via kubelet and forwards with
  `containerCollectAll: true`. Per-pod Unified Service Tagging labels
  (`tags.datadoghq.com/{service,env,version}`) auto-join logs to APM traces.

The agent holds the single DD API key (typically from a `datadog-secrets` Secret in the
`datadog` namespace, managed once by cluster infra). Workloads hold no DD credentials.
This repository does not embed `ddtrace` or a DogStatsD client; APM and
infrastructure metrics require separate cluster-level instrumentation.

Configure through the
[`operator-wandb` chart](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb):

```yaml
mcp-server:
  observability:
    provider: datadog-agent
    privacy: standard
# Agent mode uses structured container logs and no workload DD_API_KEY.
```

### Forwarder mode (serverless only)

The MCP server itself POSTs analytics events to
`https://http-intake.logs.$DD_SITE/api/v2/logs` using the
[Datadog HTTP Logs Intake API](https://docs.datadoghq.com/api/latest/logs/#send-logs),
mapping each internal event to a structured log entry (`@duration`, `@http.status_code`,
`@error.kind`, `@usr.id`, etc).

Use this on Cloud Run, Lambda, or any environment where you cannot run a node-local agent.
It is **not recommended on Kubernetes** because it duplicates what the agent already does
and requires a `DD_API_KEY` on the workload.

Required environment variables:

| Var | Purpose |
|---|---|
| `MCP_DATADOG_FORWARD=true` | Enables the in-app HTTP forwarder. |
| `DD_API_KEY` | API key. Read from env first, then GCP Secret Manager if `MCP_SERVER_SECRETS_PROVIDER=gcp` is set. |
| `DD_SITE` | Datadog site (default `datadoghq.com`). |
| `DD_SERVICE`, `DD_ENV`, `DD_VERSION` | Unified Service Tagging on every forwarded event. |
| `MCP_SERVER_SECRETS_PROVIDER=gcp` | Optional: resolve `DD_API_KEY` from the configured secret provider instead of plain environment configuration. |
| `MCP_SERVER_SECRETS_PROJECT` | Required when `MCP_SERVER_SECRETS_PROVIDER=gcp`. |

If `MCP_DATADOG_FORWARD=true` but `DD_API_KEY` resolves empty, the forwarder disables
itself. When `DD_AGENT_HOST` is also set (agent mode is active) this is logged at
`DEBUG` because a local agent is already handling observability; otherwise it's a
`WARNING` since it indicates misconfiguration.

## Environment variables (cross-reference)

| Variable | Default | Modes | Purpose |
|---|---|---|---|
| `MCP_LOG_FORMAT` | `rich` | both | `json` for structured one-line-per-record output (preferred in containers / behind DD Agent). `rich` for pretty local dev. |
| `MCP_DATADOG_FORWARD` | `false` | forwarder | Enable the in-app HTTP intake forwarder. |
| `DD_AGENT_HOST` | unset | agent | Optional deployment signal that a node-local Datadog Agent is present. The in-app forwarder uses it only to classify a missing workload API key as expected agent mode. |
| `DD_SERVICE` | `wandb-mcp-server` | both | UST service name. Chart and Cloud Run deploy both set this. |
| `DD_ENV` | `production` | both | UST environment tag. |
| `DD_VERSION` | image tag | both | Service version; the in-app forwarder keeps it as an attribute rather than a high-cardinality tag. |
| `DD_SITE` | `datadoghq.com` | forwarder | Datadog site; controls the intake URL. |
| `DD_API_KEY` | unset | forwarder | DD API key. Workload must not hold this in agent mode. |
| `MCP_SERVER_SECRETS_PROVIDER` | unset | forwarder | Set to `gcp` to resolve `DD_API_KEY` (and other secrets) from GCP Secret Manager. |
| `MCP_SERVER_SECRETS_PROJECT` | unset | forwarder | GCP project id for Secret Manager when provider is `gcp`. |

## Log format: `MCP_LOG_FORMAT=json`

The default `rich` format produces human-readable lines like:

```
[2026-04-24 15:00:40] INFO     GET / -> 200 (0.001s)
```

Datadog's auto-detection misclassifies many of these as `status:error` due to the
rich formatter's ANSI codes and level placement. Setting `MCP_LOG_FORMAT=json` emits:

```json
{"timestamp":"2026-04-24T15:00:40Z","level":"info","logger":"wandb_mcp_server.server","message":"GET / -> 200 (0.001s)"}
```

Datadog extracts the `level`, `timestamp`, `logger`, and `message` fields automatically;
`status:error` misclassification disappears.

### Scope of JSON mode: root + third-party loggers

When `MCP_LOG_FORMAT=json` is set, the server calls `configure_process_logging()` at
startup. This installs the JSON handler on:

- the Python **root logger**,
- `uvicorn`, `uvicorn.access`, `uvicorn.error`,
- `mcp` (the MCP SDK; covers `mcp.server.streamable_http.*`).

As a result, **every log line emitted by the process is structured JSON**, not just
the ones from `wandb_mcp_server.*` modules that go through `get_rich_logger()`. This
is what makes uvicorn's `GET /mcp/health 200` access lines parse cleanly in Datadog
instead of being auto-classified as `status:error` due to rich-formatter text shape.

One explicit exclusion: `wandb_mcp_server.analytics` is intentionally NOT reconfigured.
It already uses its own `_StructuredJsonFormatter` ([`src/wandb_mcp_server/analytics.py`](../src/wandb_mcp_server/analytics.py))
whose schema downstream structured-log consumers depend on. Changing it without
a coordinated schema migration would break analytics ingestion.

For MCP stdio transport, stdout is the JSON-RPC wire. The CLI reconfigures
`wandb_mcp_server.analytics` to write structured analytics to stderr in stdio
mode, while HTTP/container deployments may keep structured logs on stdout for
their configured collector.

### Defensive analytics propagation lock

`analytics.py` sets `analytics_logger.propagate = False` at module import time so that
the analytics record is emitted only by its own `_StructuredJsonFormatter` handler
and never reaches the root logger. In HTTP/container deployments this handler
writes to stdout for Cloud Logging; in stdio deployments it writes to stderr so
stdout remains pure MCP JSON-RPC. However, when the server boots under
`uvicorn`, `logging.config.dictConfig` can reset propagation on existing loggers,
silently re-enabling propagation. In that case every analytics event is emitted
twice: once via the rich `_StructuredJsonFormatter` payload on the analytics
stream, and a minimal duplicate via the root `_JsonLogFormatter` on stderr.

To prevent this, `configure_process_logging()` re-asserts
`logging.getLogger("wandb_mcp_server.analytics").propagate = False` after the
third-party logger reconfiguration loop. This guard runs only in JSON mode (rich
mode returns early).

## Privacy levels: `MCP_LOG_PRIVACY_LEVEL`

`MCP_LOG_PRIVACY_LEVEL` controls verbose application logs and the remaining
identity compatibility fields. Product telemetry never contains raw tool
arguments at any level. Public tool events contain only allowlisted
`usage_dimensions` such as booleans, stable enums, counts, and numeric buckets.

An unset privacy level defaults to `off`. Explicitly empty or invalid values
fail startup without logging the supplied value. Valid levels are case-insensitive
and ignore surrounding whitespace.

| Level | Product tool telemetry | Identity compatibility fields | Verbose request-body logs |
|---|---|---|---|
| `off` (default) | compact `usage_dimensions` only | pass-through | INFO |
| `standard` | compact `usage_dimensions` only | pass-through | demoted to DEBUG |
| `strict` | compact `usage_dimensions` only | hashed -> `<h:sha256_prefix>` | demoted to DEBUG |

Sensitive key-name redaction (`api_key`, `token`, `secret`, `password`,
`credential`, `auth`) runs at every level. Truncation of strings >200 chars
runs at every level.

### Recommended defaults per deployment target

| Deployment | Recommended level | How it's set |
|---|---|---|
| Local dev | `off` (unset) | env-var default |
| Managed serverless | Deployment-selected | Set explicitly in the managed deployment configuration |
| Customer K8s via helm chart | `standard` | chart injects from `mcp-server.observability.privacy` (default `standard`) |
| Regulated / privacy-sensitive K8s | `strict` | set `mcp-server.observability.privacy: strict` |

### Why the split

Schema 1.1 uses a pseudonymous `actor_id` derived from the API-key digest and
compact usage dimensions, so product analysis does not require raw resource
identifiers or tool arguments. Kubernetes agent mode keeps collection within
the operator-selected logging path. Compact dimensions reduce storage and
indexing cost in either topology. `standard` remains the safe application-log
default; `strict` also hashes legacy identity fields for regulated deployments.

### Datadog product dimensions

Datadog receives the same bounded `usage_dimensions` as the canonical event and
never receives `params`. Only deployment, harness, method, public tool, success,
and error class are tags. Actor IDs, session IDs, versions, durations, and error
messages remain attributes to avoid high-cardinality indexing costs.

## MCP client harness dimensions

MCP analytics include a small client dimension for operational and product
analysis. These fields are derived from allowlisted MCP
signals such as `initialize.params.clientInfo`, session metadata, and
`User-Agent` fallback. They are untrusted analytics dimensions only and must not
be used for authentication, authorization, rate-limit bypasses, or protocol
branching.

Schema 1.1 canonical fields:

| Field | Purpose |
|---|---|
| `agent_harness` | Exact detected product, such as `codex`, `claude_code`, `cursor`, `lechat`, or `gemini_cli`. |
| `client_vendor` | Vendor bucket, such as `openai`, `anthropic`, `cursor`, `google`, or `mistral`. |
| `call_type` | Exact MCP JSON-RPC method, such as `initialize`, `tools/list`, or `tools/call`. |
| `tool_name` | Public MCP tool name; emitted exactly once per public invocation. |
| `actor_id` | Pseudonymous `wandb_key:<24 hex chars>` cohort identifier. |
| `mcp_client_family` | One-release compatibility alias for the previous family field. |
| `mcp_client_app` | One-release compatibility alias for the previous app field. |
| `mcp_client_source` | Signal used for classification: `initialize_client_info`, `meta_client_info`, `session_metadata`, `user_agent`, or `unknown`. |
| `mcp_protocol_version` | MCP protocol version observed on the request. |
| `mcp_jsonrpc_method` | JSON-RPC method such as `initialize`, `tools.list`, or `tools.call`. |

Debug-only fields such as raw-ish client names, versions, and user-agent product
tokens are disabled by default. If enabled for classifier maintenance, they must
remain sampled, bounded, and excluded from Datadog tags and default Hex
dashboards.

Recommended Datadog views:

- Request error rate by `agent_harness`.
- Tool error rate by `agent_harness` and `tool_name`.
- p95 latency by `agent_harness` and `tool_name`.
- Unknown-client rate by `mcp_client_source`.
- Initialize failures by `mcp_protocol_version` and `agent_harness`.

Recommended product analyses:

- Daily active actors, sessions, and tool calls by `client_vendor` and
  `agent_harness`.
- Tool adoption and tool mix by client app.
- Success rate, error rate, and latency by client app and tool.
- Initialize to tools/list to tools/call funnel health by client app.
- Unknown-client coverage and classifier candidates.
- Release-over-release regressions by client app and `release_version`.

### Identifier hashing at `strict`

`<h:sha256_prefix>` uses the first 12 hex chars of `sha256(value)`.
Deterministic (the same entity name always hashes to the same digest), so
legacy identity joins remain possible during the schema transition. New
product dashboards should use `actor_id` and `usage_dimensions`. The hash is
not reversible without a rainbow table over known W&B entity names,
which is out of scope for legal defensibility (the retained data is no
longer plaintext customer identifiers).

## Managed serverless deployments

Managed serverless deployments use forwarder mode because they cannot run a
node-local agent. The deployment supplies `MCP_DATADOG_FORWARD`, `DD_SITE`,
`DD_SERVICE`, and a secret-backed `DD_API_KEY`. Environment names, project IDs,
revision names, and secret resource names are deployment details and are
intentionally not encoded in this durable application document.
