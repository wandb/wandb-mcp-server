# Agent Lens tools

Fifteen tools (13 read, 2 write) over the [Agent Lens](https://docs.coreweave.com/products/agent-lens)
official `v1`, Insights, conversation-tag, instrumentation-health, and judge-guideline APIs.
Agent Lens classifies agent turns into intent and failure categories and clusters,
manages human- and judge-applied conversation tags, stores project judge guidelines,
and checks spans against its 16 OpenTelemetry GenAI instrumentation rules. None of
that is visible to classic Weave calls, so these tools pair directly with the
`*_weave_agent_*` family on `trace.wandb.ai`.

## Enabling them

The tools are not enabled by default. W&B-hosted operators add the
`agent-lens` extension to the `models-weave-agents` base profile (the legacy
`models-weave-agents-agent-lens` profile name still resolves to the same
selection) and name the reviewed origin:

```sh
export WANDB_MCP_TOOL_PROFILE=models-weave-agents
export WANDB_MCP_TOOL_EXTENSIONS=agent-lens
export WANDB_MCP_ACCESS_MODE=read-write
export MCP_WORKLOAD_PROFILE=shared
export AGENT_LENS_BASE_URL=https://<your-agent-lens-host>
```

`AGENT_LENS_BASE_URL` has no default. It must be an absolute HTTPS origin with
no credentials, path, query, or fragment — the same bar `WB_AGENT_BASE_URL`
clears, because both forward the caller's W&B credential to a non-W&B origin.
The server refuses to start if the variable is missing or malformed.

The profile is allowed on managed `shared` workloads and in local development.
It is rejected on managed `dedicated` workloads. Setting
`WANDB_MCP_ACCESS_MODE=read-only` keeps all 13 Agent Lens read tools while
omitting `manage_agent_lens_tags_tool` and `manage_agent_lens_guidelines_tool`
(alongside `manage_weave_agent_feedback_tool`, `create_wandb_report_tool`, and
`log_analysis_to_wandb`).

## Authentication and API prefixes

Agent Lens accepts the caller's W&B API key as a bearer token (`Authorization: Bearer <key>`)
with `X-Wandb-Entity` and `X-Wandb-Project` headers and derives project
authorization server-side:

- an explicit `Authorization` header never falls back to a browser cookie identity;
- `UseAdminPrivileges` is only reachable on the cookie path, so a bearer caller
  cannot escalate.

Official public `v1` endpoints (`GET/POST /api/v1/tags`, `GET/POST/DELETE /api/v1/tags/{id}`)
return `{"data": ...}` envelopes and `{"message", "param"}` validation errors;
Insights, tag-assignment, guideline, instrumentation-health, and conversation-summary
endpoints route through `/api/*` on the same origin.

## The tools

| Tool | Access | Endpoint |
|---|---|---|
| `render_agent_lens_insights_tool` | `read` | Parallel gather across `latest-week`, `clustering-status`, `{failure,intent}/breakdowns`, `clusters`, and optional `failure-attributions/query` (`ui://wandb/agent-lens-insights.html` MCP App) |
| `get_agent_lens_insights_coverage_tool` | `read` | `GET /api/insights/latest-week` |
| `get_agent_lens_clustering_status_tool` | `read` | `GET /api/insights/clustering-status` |
| `get_agent_lens_category_breakdowns_tool` | `read` | `GET /api/insights/{type}/breakdowns` |
| `list_agent_lens_clusters_tool` | `read` | `GET /api/insights/clusters` (+ optional `GET /api/insights/{type}/volume`) |
| `list_agent_lens_category_example_turns_tool` | `read` | `GET /api/insights/{type}/categories/{id}/example-turns` |
| `get_agent_lens_failure_attributions_tool` | `read` | `POST /api/insights/failure-attributions/query` |
| `list_agent_lens_tags_tool` | `read` | `GET /api/v1/tags` / `GET /api/v1/tags/{id}` |
| `get_agent_lens_conversation_tags_tool` | `read` | `POST /api/conversation-tags/query` |
| `list_agent_lens_tagged_conversations_tool` | `read` | `POST /api/conversation-tags/conversations/query` or `GET /api/tags/{id}/assignments` |
| `get_agent_lens_tag_distribution_tool` | `read` | `POST /api/conversation-tags/distribution` |
| `get_agent_lens_guidelines_tool` | `read` | `GET /api/guidelines?type=project` or `POST /api/conversations/{id}/summarize` |
| `diagnose_agent_lens_instrumentation_tool` | `read` | `GET /api/instrumentation/health` |
| `manage_agent_lens_tags_tool` | `write` | `POST /api/v1/tags`, `POST/DELETE /api/v1/tags/{id}`, `POST/DELETE /api/conversation-tags` |
| `manage_agent_lens_guidelines_tool` | `write` | `POST /api/guidelines`, `PATCH/DELETE /api/guidelines/{id}` |

Read/write selection comes from the packaged runtime contract's `access` field,
never from the HTTP method.

## Recommended workflows

### 1. Cluster discovery, stratified sampling, and root-cause analysis

1. `get_agent_lens_insights_coverage_tool` — find a populated `first_week`..`latest_week` range.
2. `get_agent_lens_category_breakdowns_tool` (`signature_type="both"`, `top_n_clusters=10`)
   or `list_agent_lens_clusters_tool` (`include_volume=True` for trend spikes) —
   inspect top user-intent and failure categories and clusters.
3. `list_agent_lens_category_example_turns_tool` with `topic_ids=[...]` and
   `per_topic_limit=5` — stratified-sample representative turns concurrently
   across every cluster in the category so high-volume clusters never crowd out
   smaller ones.
4. `get_agent_lens_failure_attributions_tool` — pull `failure_reason`,
   `failure_cluster`, `failure_severity`, and `failure_evidence_span_ids` for
   sampled `trace_ids`.
5. `get_weave_agent_conversation_tool` (`sort_direction="asc"`, `include_feedback=True`)
   or `get_weave_agent_conversation_spans_tool` — read full multi-turn
   trajectories chronologically from Turn 1 or batch-load span trees across up to
   100 conversations.

### 2. Human feedback, tags, and judge guidelines

- Search free-text human notes/corrections (`feedback_type="wandb.note.1"`) and
  reactions (`feedback_type="wandb.agent_user_feedback"`) across the project via
  `query_weave_agent_feedback_tool`, or add/delete them in `read-write` mode via
  `manage_weave_agent_feedback_tool`.
- Inspect the `v1` tag catalog via `list_agent_lens_tags_tool`, page turn-level
  assignments with `list_agent_lens_tagged_conversations_tool(include_assignments=True)`,
  and create/update/assign tags via `manage_agent_lens_tags_tool`.
- Inspect and tune project-wide Insights judge rules (`1`–`4,000` characters)
  via `get_agent_lens_guidelines_tool` and `manage_agent_lens_guidelines_tool`.

### 3. Instrumentation loop

`diagnose_agent_lens_instrumentation_tool` lets a coding agent instrument an
app without a person reading traces: run the app, call the tool (usually with
`conversation_id="latest"`), fix `issues[0]`, and repeat until `verdict` is
`clean`. Span rules update immediately; trace rules wait until a trace has been
quiet for 10 minutes (`verdict: settling` while `unsettled_trace_count > 0`).
Zero matching spans return `verdict: no_spans` along with exporter reachability
diagnostics, never a false clean verdict.

## Bounds and transport safety

- Ranged Insights queries are bounded to **30 days** (`start_at < end_at`);
  volume queries cap at **1,440 buckets** (`bucket_seconds` `60`–`86,400`).
- `conversation_ids` ≤ 5,000, `tag_ids` ≤ 100, `category_ids` ≤ 100,
  `topic_ids` ≤ 20, `per_topic_limit` 1–25, failure-attribution `trace_ids` ≤ 500,
  example-turn `limit` 1–50, and tag-assignment `limit` 1–500.
- `v1` tag slugs match `^[a-z0-9]+(?:-[a-z0-9]+)*$` (1–64 chars), tag `criteria`
  ≤ 1,600 chars, and project `guidelines` content ≤ 4,000 chars.
- Oversized responses trim to the configured token budget with `_truncation`
  metadata; paginated example-turn pages fail closed rather than dropping rows
  behind `next_cursor`. Downloads are capped at 4 MiB, redirects are disabled,
  one absolute deadline covers connect + headers + body, and low-level HTTPX
  request-line logs are suppressed while an Agent Lens call is active.

## Errors

| `error` | Meaning |
|---|---|
| `agent_lens_not_configured` | `AGENT_LENS_BASE_URL` missing or malformed |
| `auth_required` | No W&B API key in context or environment |
| `agent_lens_forbidden` | 401/403 — the key cannot access this project |
| `agent_lens_unavailable` | 404 — origin does not expose the endpoint |
| `agent_lens_invalid_request` | Rejected locally, or upstream 400/404/409/422 validation or resource error |
| `agent_lens_query_failed` | Transport failure, unexpected status, malformed JSON, or oversized download |
| `tool_timeout` | The request exhausted the current MCP tool deadline |

429 and 503 raise retryable server-busy errors through the shared handler.

## Maintenance

Verify live contracts against `/api/v1/openapi.json` and `/api/openapi.json` on
the Agent Lens service and `/openapi.json` on `trace.wandb.ai`, and run
`scripts/agent_lens_smoke.py` against approved populated fixture IDs.
