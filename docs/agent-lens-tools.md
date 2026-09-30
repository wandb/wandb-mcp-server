# Agent Lens tools

Nine read tools over the [Agent Lens](https://github.com/wandb/agent-lens) Insights
and conversation-tag APIs. Agent Lens classifies agent turns into intent and
failure categories and clusters, and carries human- and judge-applied
conversation tags. None of that is visible to the Weave calls or Agents data
planes, so these tools complement rather than duplicate `query_weave_traces_tool`
and the `*_weave_agent_*` family.

## Enabling them

The tools are not enabled by default. W&B-hosted operators may select the
shared-only profile and name the reviewed origin:

```sh
export WANDB_MCP_TOOL_PROFILE=models-weave-agents-agent-lens
export MCP_WORKLOAD_PROFILE=shared
export AGENT_LENS_BASE_URL=https://<your-agent-lens-host>
```

`AGENT_LENS_BASE_URL` has no default. It must be an absolute HTTPS origin with
no credentials, path, query, or fragment — the same bar `WB_AGENT_BASE_URL`
clears, because both forward the caller's W&B credential to a non-W&B origin.
The server refuses to start if the variable is missing or malformed, so a
misconfiguration can never silently fall back to a different host.

The profile is allowed on managed `shared` workloads and in local development.
It is rejected on managed `dedicated` workloads. The default `models-weave`
profile remains unchanged, and the endpoint alone never registers these tools.

## Authentication

Agent Lens accepts the caller's W&B API key as a bearer token and derives
project authorization from it server-side. Two properties of
`internal/auth/middleware.go` matter here:

- an explicit `Authorization` header never falls back to a browser cookie identity;
- `UseAdminPrivileges` is only reachable on the cookie path, so a bearer caller
  cannot escalate.

An MCP caller therefore reads exactly the projects its own key can reach. Every
request additionally carries the project in `X-Wandb-Entity` / `X-Wandb-Project`.

## Read-only by construction

All nine tools are declared `access: read` in the runtime contract, so the group
is identical under `WANDB_MCP_ACCESS_MODE=read-only` (39 tools read-write, 37
read-only — the 2 dropped are models-group writes). Agent Lens mutations
(creating tags, views, or alignment examples) are deliberately not exposed.

## The tools

| Tool | Endpoint |
|---|---|
| `get_agent_lens_insights_coverage_tool` | `GET /insights/latest-week` |
| `get_agent_lens_clustering_status_tool` | `GET /insights/clustering-status` |
| `get_agent_lens_category_breakdowns_tool` | `GET /insights/intent-category-breakdowns` |
| `list_agent_lens_category_example_turns_tool` | `GET /insights/{type}/categories/{id}/example-turns` |
| `get_agent_lens_failure_attributions_tool` | `POST /insights/failure-attributions/query` |
| `list_agent_lens_tags_tool` | `GET /tags` |
| `get_agent_lens_conversation_tags_tool` | `POST /conversation-tags/query` |
| `list_agent_lens_tagged_conversations_tool` | `POST /conversation-tags/conversations/query` |
| `get_agent_lens_tag_distribution_tool` | `POST /conversation-tags/distribution` |

Four of these are `POST` endpoints that perform reads — Agent Lens uses a request
body where the filter would not fit in a query string. Read/write selection
comes from the contract's `access` field, never from the HTTP method.

## Suggested call order

Insights come from a periodic classification job, so a project can hold plenty of
traces and no classified turns, and the usable range rarely reaches today:

1. `get_agent_lens_insights_coverage_tool` — find a populated range.
2. `get_agent_lens_category_breakdowns_tool` — the aggregate picture over that range.
3. `list_agent_lens_category_example_turns_tool` returns paged identifiers and
   bounded classification/failure metadata for a category, optionally refined
   with Agent Lens topic IDs.
4. Pass one or more returned `trace_id` values to
   `get_agent_lens_failure_attributions_tool` for bounded failure attribution,
   or to `get_weave_agent_trace_tool` / `query_weave_traces_tool` for the trace
   content itself.

The Insights tools return identifiers and attribution, never message bodies.
In particular, the example-turn projection removes upstream `agent_message`
and `message` fields before token truncation.

For tags, call `list_agent_lens_tags_tool` first. It returns the project tag
catalog, including each tag's UUID and display name. Pass UUIDs—not names—to
`list_agent_lens_tagged_conversations_tool`; assignments returned by
`get_agent_lens_conversation_tags_tool` include both `tag_id` and `tag`.

## Bounds

Arguments that the server bounds are checked locally first, so an oversized
request fails with an actionable message instead of a bare `422`:

- MCP limits ranged Insights reads to **30 days** as a response/work safety
  policy, even though current Agent Lens accepts any nonempty range.
- `signature_type` accepts exactly `intent` or `failure`.
- `conversation_ids` ≤ 5000, `tag_ids` ≤ 100, `topic_ids` ≤ 20, and
  failure-attribution `trace_ids` ≤ 500.
- `limit` 1–50 for example turns; `time_bucket_seconds` 1–86400.
- Tag filters must be UUIDs. Failure-attribution reads require at least one
  trace ID.

Insights bounds are RFC 3339 timestamps; the tag distribution uses epoch
milliseconds, matching each endpoint's own contract.

Oversized responses are trimmed to the token budget and annotated with
`_truncation`, reporting the field trimmed and the original count so a partial
answer is never mistaken for a complete one. Each download is also capped at
4 MiB or the lower configured accumulated-byte limit before JSON decoding.
Requests reject redirects. One absolute deadline covers connection setup,
response headers, and the complete streamed body, so a response cannot remain
open indefinitely by continuing to send small chunks.

Within the MCP process, low-level HTTPX and HTTPCore request-line records are
suppressed while an Agent Lens request is active. This keeps caller-derived
category identifiers, topic identifiers, and pagination cursors out of those
client logs without muting unrelated HTTP client activity. Application failure
telemetry remains bounded and sanitized. Reverse-proxy and platform logging are
separate operator concerns.

## Errors

| `error` | Meaning |
|---|---|
| `agent_lens_not_configured` | `AGENT_LENS_BASE_URL` missing or malformed |
| `auth_required` | No W&B API key in context or environment |
| `agent_lens_forbidden` | 401/403 — the key cannot read this project |
| `agent_lens_unavailable` | 404 — origin does not expose the endpoint |
| `agent_lens_invalid_request` | Rejected locally, or a 422 with a sanitized validation detail |
| `agent_lens_query_failed` | Transport failure, unexpected status, malformed JSON, or oversized response |
| `tool_timeout` | The request exhausted the current MCP tool deadline |

429 and 503 are raised as retryable server-busy errors through the shared
handler. Reads are never retried in-process; retrying belongs to the MCP caller.

## Maintenance

Agent Lens generates an OpenAPI document at `/api/openapi.json` (Huma v2). To
check this client against the live contract, diff the paths and field names
above against that document and run `scripts/agent_lens_smoke.py` with approved,
populated fixture IDs. The response shapes here are transcribed from
`internal/api/insights.go`, `internal/api/tags.go`, and
`internal/api/conversation_tags.go`; a field rename on that side is the most
likely source of drift.
