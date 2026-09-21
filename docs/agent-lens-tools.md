# Agent Lens tools

Nine read tools over the [Agent Lens](https://github.com/wandb/agent-lens) Insights
and conversation-tag APIs. Agent Lens classifies agent turns into intent and
failure categories and clusters, and carries human- and judge-applied
conversation tags. None of that is visible to the Weave calls or Agents data
planes, so these tools complement rather than duplicate `query_weave_traces_tool`
and the `*_weave_agent_*` family.

## Enabling them

The tools are absent from every managed profile and are not enabled by default.
Select the profile and name the origin:

```sh
export WANDB_MCP_TOOL_PROFILE=models-weave-agent-lens
export AGENT_LENS_BASE_URL=https://<your-agent-lens-host>
```

`AGENT_LENS_BASE_URL` has no default. It must be an absolute HTTPS origin with
no credentials, path, query, or fragment — the same bar `WB_AGENT_BASE_URL`
clears, because both forward the caller's W&B credential to a non-W&B origin.
The server refuses to start if the variable is missing or malformed, so a
misconfiguration can never silently fall back to a different host.

The profile is rejected on managed `shared` and `dedicated` workloads. It runs
in `local` only.

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
is identical under `WANDB_MCP_ACCESS_MODE=read-only` (31 tools read-write, 29
read-only — the 2 dropped are models-group writes). Agent Lens mutations
(creating tags, views, or alignment examples) are deliberately not exposed.

## The tools

| Tool | Endpoint |
|---|---|
| `get_agent_lens_insights_coverage_tool` | `GET /insights/latest-week` |
| `get_agent_lens_clustering_status_tool` | `GET /insights/clustering-status` |
| `get_agent_lens_category_breakdowns_tool` | `GET /insights/intent-category-breakdowns` |
| `list_agent_lens_category_example_turns_tool` | `GET /insights/{type}/categories/{id}/example-turns` |
| `list_agent_lens_matching_turns_tool` | `GET /insights/matching-turns` |
| `list_agent_lens_conversation_tag_names_tool` | `GET /conversation-tags` |
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
3. `list_agent_lens_matching_turns_tool` (failure detail per turn) or
   `list_agent_lens_category_example_turns_tool` (paged identifiers) to drill in.
4. Pass the returned `trace_id` to `get_weave_agent_trace_tool` or
   `query_weave_traces_tool` for the trace content itself. The Insights tools
   return identifiers and attribution, never message bodies.

For tags, call `list_agent_lens_conversation_tag_names_tool` first: tag names are
project-defined free text, so guessing a spelling usually returns nothing.

## Bounds

Arguments that the server bounds are checked locally first, so an oversized
request fails with an actionable message instead of a bare `422`:

- Ranged Insights reads span at most **30 days** (`insights.Window.Validate`).
- `conversation_ids` ≤ 5000, `tags` ≤ 100, `cluster_ids` ≤ 20.
- `limit` 1–50 for example turns; `time_bucket_seconds` 1–86400.

Insights bounds are RFC 3339 timestamps; the tag distribution uses epoch
milliseconds, matching each endpoint's own contract.

Oversized responses are trimmed to the token budget and annotated with
`_truncation`, reporting the field trimmed and the original count so a partial
answer is never mistaken for a complete one.

## Errors

| `error` | Meaning |
|---|---|
| `agent_lens_not_configured` | `AGENT_LENS_BASE_URL` missing or malformed |
| `auth_required` | No W&B API key in context or environment |
| `agent_lens_forbidden` | 401/403 — the key cannot read this project |
| `agent_lens_unavailable` | 404 — origin does not expose the endpoint |
| `agent_lens_invalid_request` | Rejected locally, or a 422 with the server's detail |
| `agent_lens_query_failed` | Transport failure, unexpected status, or invalid JSON |

429 and 503 are raised as retryable server-busy errors through the shared
handler. Reads are never retried in-process; retrying belongs to the MCP caller.

## Maintenance

Agent Lens generates an OpenAPI document at `/api/openapi.json` (Huma v2), but
does not publish it as an artifact. To check this client against the live
contract, run Agent Lens and diff the paths and field names above against that
document. The response shapes here are transcribed from `internal/api/insights.go`
and `internal/api/conversation_tags.go`; a field rename on that side is the most
likely source of drift.
