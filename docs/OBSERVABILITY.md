# Observability

The MCP server emits product analytics (Segment) and operational telemetry (Datadog).
This document covers the two supported Datadog collection modes, the environment
variables that control them, and when to pick each.

## TL;DR

| Deployment target | Recommended mode | What to set |
|---|---|---|
| Managed Kubernetes (dedicated cloud, self-managed) | **Agent mode** | Install DD Agent DaemonSet once per cluster; set `MCP_LOG_FORMAT=json` on the pod. No DD credentials on the workload. |
| Serverless (Cloud Run, Lambda) | **Forwarder mode** | `MCP_DATADOG_FORWARD=true` + `DD_API_KEY` (env or GCP Secret Manager) |
| Local dev | No Datadog | Leave `MCP_DATADOG_FORWARD` unset; `MCP_LOG_FORMAT` defaults to `rich` |

## Two collection modes

### Agent mode (preferred on Kubernetes)

A Datadog Agent DaemonSet on every node:

- **Logs**: tails `/var/log/pods/**` (container stdout/stderr) via kubelet and forwards with
  `containerCollectAll: true`. Per-pod Unified Service Tagging labels
  (`tags.datadoghq.com/{service,env,version}`) auto-join logs to APM traces.
- **APM traces**: the MCP server's `ddtrace` library sends spans to `$DD_AGENT_HOST:8126`
  (node IP, via downward API). No app code change required.
- **Metrics**: DogStatsD on `$DD_AGENT_HOST:8125`; system/container metrics via kubelet.

The agent holds the single DD API key (typically from a `datadog-secrets` Secret in the
`datadog` namespace, managed once by cluster infra). Workloads hold no DD credentials.

Configure via the helm chart (see
[wandb/helm-charts operator-wandb 0.42.2+](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)):

```yaml
mcp-server:
  datadog:
    enabled: true              # injects DD_SERVICE/ENV/VERSION + DD_AGENT_HOST + MCP_LOG_FORMAT=json
    # mode: agent              # default; no MCP_DATADOG_FORWARD env, no workload DD_API_KEY
    deploymentType: dedicated-cloud
    customer: acme
    extraTags: ["team:ml-platform", "region:us-west1"]
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
| `DD_SITE` | Datadog site (default `datadoghq.com`; W&B uses `us5.datadoghq.com`). |
| `DD_SERVICE`, `DD_ENV`, `DD_VERSION` | Unified Service Tagging on every forwarded event. |
| `MCP_SERVER_SECRETS_PROVIDER=gcp` | Optional: if set, `DD_API_KEY` is fetched from `mcp-server-datadog-api-key` in GCP Secret Manager instead of env. |
| `MCP_SERVER_SECRETS_PROJECT` | Required when `MCP_SERVER_SECRETS_PROVIDER=gcp`. |

If `MCP_DATADOG_FORWARD=true` but `DD_API_KEY` resolves empty, the forwarder disables
itself. When `DD_AGENT_HOST` is also set (agent mode is active) this is logged at
`DEBUG` because a local agent is already handling observability; otherwise it's a
`WARNING` since it indicates misconfiguration.

## Environment variables (cross-reference)

| Variable | Default | Modes | Purpose |
|---|---|---|---|
| `MCP_LOG_FORMAT` | `rich` | both | `json` for structured one-line-per-record output (preferred in containers / behind DD Agent). `rich` for pretty local dev. Chart 0.42.2+ sets `json` when `datadog.enabled=true`. |
| `MCP_DATADOG_ENABLED` | `false` | informational | Marker emitted by the chart; doesn't gate behavior today. |
| `MCP_DATADOG_FORWARD` | `false` | forwarder | Enable the in-app HTTP intake forwarder. |
| `DD_AGENT_HOST` | unset | agent | Node IP where the DD Agent runs; used by `ddtrace` and DogStatsD. Chart sets from `status.hostIP`. |
| `DD_TRACE_AGENT_HOSTNAME` | unset | agent | Same as `DD_AGENT_HOST`, for trace-agent clients that read this name. |
| `DD_SERVICE` | `wandb-mcp-server` | both | UST service name. Chart and Cloud Run deploy both set this. |
| `DD_ENV` | `production` | both | UST environment tag. |
| `DD_VERSION` | image tag | both | UST version tag. |
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
whose schema downstream GCP Cloud Logging -> BigQuery pipelines depend on. Touching
it would silently break analytics ingestion.

For MCP stdio transport, stdout is the JSON-RPC wire. The CLI reconfigures
`wandb_mcp_server.analytics` to write structured analytics to stderr in stdio
mode, while HTTP/container deployments keep stdout for Cloud Logging ingestion.
If an older stdio build writes analytics JSON to stdout, set
`MCP_ANALYTICS_DISABLED=true` to suppress analytics as a workaround.

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
mode returns early), so Cloud Run today is unaffected. Behavior was observed live
on Cloud Run staging revision `wandb-mcp-server-staging-00084-p8s`.

## Privacy levels: `MCP_LOG_PRIVACY_LEVEL`

`MCP_LOG_PRIVACY_LEVEL` controls how aggressively customer-supplied content
is redacted before it reaches any log sink (Cloud Logging, Segment, Datadog
forwarder). Applied uniformly by `AnalyticsTracker._sanitise_params`, by the
identity-hashing helper in `analytics.py`, and by verbose-log-site gates in
`tools_utils.py` and `weave_api/client.py`.

| Level | Free-text params (`query`, `prompt`, `description`, ...) | Identifiers (`entity_name`, `project_name`, `run_id`, `user_id`, `email_domain`) | Verbose `ToolCall` / raw-body logs |
|---|---|---|---|
| `off` (default) | pass-through | pass-through | INFO |
| `standard` | redacted -> `<redacted: text len=N>` | pass-through | demoted to DEBUG |
| `strict` | redacted -> `<redacted: text len=N>` | hashed -> `<h:sha256_prefix>` | demoted to DEBUG |

Sensitive key-name redaction (`api_key`, `token`, `secret`, `password`,
`credential`, `auth`) runs at every level. Truncation of strings >200 chars
runs at every level.

### Recommended defaults per deployment target

| Deployment | Recommended level | How it's set |
|---|---|---|
| Local dev | `off` (unset) | env-var default |
| Cloud Run (W&B-managed, feeds BigQuery analytics) | `off` explicit | `deploy.sh` passes `MCP_LOG_PRIVACY_LEVEL=off` |
| Customer K8s via helm chart | `standard` | chart injects from `mcp-server.privacy.logLevel` (default `standard`) |
| Regulated / privacy-sensitive K8s | `strict` | override chart value to `strict` |

### Why the split

Cloud Run's analytics logger emits directly to GCP Cloud Logging, which pipes
to BigQuery for product analytics. That pipeline depends on plaintext
`user_id`, `email_domain`, and `params` fields for cohort analysis. `off`
preserves this byte-for-byte.

Customer K8s installs do NOT feed W&B's BigQuery; their analytics logger goes
to the container log stream for the local Datadog Agent to collect into the customer's own
Datadog tenant. There's no business need to retain plaintext free-text
params there, and redaction reduces legal exposure if customer logs are
subpoenaed, exported, or retained longer than needed. `standard` is the safe
default. `strict` goes one step further for regulated customers.

### Datadog params privacy

Cloud Run uses `MCP_LOG_PRIVACY_LEVEL=off` so the W&B-managed BigQuery product
analytics sink keeps its historical cohort fields. The Datadog HTTP forwarder is
an operational sink, so it applies a separate params privacy level:
`MCP_DATADOG_PARAM_PRIVACY_LEVEL`, defaulting to `standard`.

This gives Cloud Run Datadog logs semantic parity with Helm/agent-ingested
`ANALYTICS_EVENT` logs without forwarding raw free text. Cloud Run Datadog
payloads include sanitized fields under `attributes.params.*`, corresponding to
Helm's `custom.params.*` fields. Free-text values such as GraphQL queries,
prompts, descriptions, report text, and messages are redacted to
`<redacted: text len=N>` by default; secret-like keys are always redacted.

Only use `MCP_DATADOG_PARAM_PRIVACY_LEVEL=off` for short-lived debugging in a
controlled environment. Do not set it as a production default.

## MCP client harness dimensions

Hosted MCP analytics include a small client dimension for Datadog incident
analysis and Hex adoption analysis. These fields are derived from allowlisted MCP
signals such as `initialize.params.clientInfo`, session metadata, and
`User-Agent` fallback. They are untrusted analytics dimensions only and must not
be used for authentication, authorization, rate-limit bypasses, or protocol
branching.

Default fields:

| Field | Purpose |
|---|---|
| `mcp_client_family` | Broad client bucket, such as `claude`, `cursor`, `openai`, `mistral`, `gemini`, `linear`, or `unknown`. |
| `mcp_client_app` | More specific client bucket, such as `claude_code`, `codex_cli`, `cursor`, `lechat`, or `gemini_cli`. |
| `mcp_client_source` | Signal used for classification: `initialize_client_info`, `meta_client_info`, `session_metadata`, `user_agent`, or `unknown`. |
| `mcp_protocol_version` | MCP protocol version observed on the request. |
| `mcp_jsonrpc_method` | JSON-RPC method such as `initialize`, `tools.list`, or `tools.call`. |

Debug-only fields such as raw-ish client names, versions, and user-agent product
tokens are disabled by default. If enabled for classifier maintenance, they must
remain sampled, bounded, and excluded from Datadog tags and default Hex
dashboards.

Recommended Datadog views:

- Request error rate by `mcp_client_app`.
- Tool error rate by `mcp_client_app` and `tool_name`.
- p95 latency by `mcp_client_app` and `tool_name`.
- Unknown-client rate by `mcp_client_source`.
- Initialize failures by `mcp_protocol_version` and `mcp_client_app`.

Recommended Hex analyses:

- Daily active users, sessions, and tool calls by `mcp_client_family` and
  `mcp_client_app`.
- Tool adoption and tool mix by client app.
- Success rate, error rate, and latency by client app and tool.
- Initialize to tools/list to tools/call funnel health by client app.
- Unknown-client coverage and classifier candidates.
- Release-over-release regressions by client app and `release_version`.

### Identifier hashing at `strict`

`<h:sha256_prefix>` uses the first 12 hex chars of `sha256(value)`.
Deterministic (the same entity name always hashes to the same digest), so
cohort analytics (tool adoption by entity, error rates by project) still
work. Not reversible without a rainbow table over known W&B entity names,
which is out of scope for legal defensibility (the retained data is no
longer plaintext customer identifiers).

## What about Cloud Run today?

Cloud Run production (see [`deploy.sh`](https://github.com/wandb/wandb-mcp-server-test/blob/main/deploy.sh))
sets `MCP_DATADOG_FORWARD=true`, `DD_SITE=us5.datadoghq.com`, `DD_SERVICE=wandb-mcp-server`,
and pulls `DD_API_KEY` via `MCP_SERVER_SECRETS_PROVIDER=gcp` from the
`mcp-server-datadog-api-key` secret in `wandb-mcp-production`. That configuration is
unchanged by this PR: the forwarder path is preserved, `DD_AGENT_HOST` is not set
(so the WARN behavior for a misconfigured forwarder-without-key stays on Cloud Run),
and `MCP_LOG_FORMAT` defaults to `rich` until Cloud Run explicitly opts in.
